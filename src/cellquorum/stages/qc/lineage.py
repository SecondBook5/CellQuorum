# Pipeline step (order=20): qc — provisional lineages, so severity is judged within a type.
"""Provisional lineages: the discriminator between damage and unusual baseline biology.

This module exists because of a measured defect. With severity fitted against a *sample-wide*
null, a synthetic cohort containing 50 perfectly healthy cells whose constitutive biology is
low-complexity and high-mitochondrial — the neutrophil / erythrocyte / plasma-cell profile —
was quarantined at **50 out of 50**, while 0 of 950 ordinary cells were. On the real 201,923
cell validation cohort, 2,091 cells carry that same signature and all 2,091 are barred from
fitting anything.

The cause is not a bad threshold. It is that the concordance requirement, which exists to stop
a single noisy axis condemning a cell, is *satisfied by construction* for these cells:

    capture_complexity  0.946      "far fewer genes than the sample median"
    metabolic_stress    0.974      "far more mitochondrial reads than the sample median"
    -> 2 concordant families -> quarantine

Both statements are true, and neither means the cell is damaged. A dying cell is
low-complexity and high-mito because it is dying; a neutrophil is low-complexity and high-mito
because it is a neutrophil. Against a sample-wide null the two are **geometrically identical**,
so no amount of threshold tuning separates them.

## What does separate them

Coherence. A rare cell type is a *group*: many cells share the same unusual profile and
resemble each other in genes that are not QC metrics. Damage is idiosyncratic: each damaged
cell degrades toward a different random subset, and damaged cells of different types resemble
each other only in the QC metrics themselves.

So severity is fitted **within a provisional transcriptional grouping**. A cell that looks
unusual next to the whole sample but ordinary next to its own lineage is a rare cell type. A
cell that looks unusual next to its own lineage is damaged. That reduces the question from
"is this profile unusual?" — which rarity alone answers yes to — to "is this cell unusual
*for what it is?*", which is the question QC actually means to ask.

## Chicken and egg, and why this stays inside the QC stage

Grouping needs a manifold and a manifold needs QC. The loop is broken by doing a deliberately
cheap, deliberately permissive grouping here: per-cell normalization (which fits nothing, so it
cannot leak), a coarse PCA, and Leiden at low resolution. It only has to be good enough to put
neutrophils next to neutrophils. Keeping it inside the stage means no new stage ordering and no
refit of the real manifold, and the downstream chain still receives final masks rather than
provisional ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cellquorum.stages.qc._types import ExpressionMatrix

if TYPE_CHECKING:  # pragma: no cover - import cost
    import anndata as ad

logger = logging.getLogger(__name__)

#: obs column holding the provisional grouping, kept for provenance and figures. Named
#: "provisional" in full because it must never be mistaken for the analysis clustering.
LINEAGE_COLUMN = "qc_provisional_lineage"

#: obs column recording which grouping level supplied each cell's null.
NULL_LEVEL_COLUMN = "qc_null_group_level"

#: Label for cells no provisional grouping could place.
UNASSIGNED = "unassigned"

#: obsm key holding the provisional embedding. Kept rather than discarded because the
#: archetype audit and the QC figures both need an embedding, and recomputing one would be
#: a second PCA over the whole cohort for no new information.
PROVISIONAL_EMBEDDING = "X_qc_provisional"


def provisional_lineages(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    resolution: float = 0.5,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    n_top_genes: int = 2000,
    min_genes: int = 50,
    random_state: int = 0,
) -> pd.Series:
    """Group cells transcriptionally, coarsely, before any QC verdict exists.

    Deliberately permissive and deliberately cheap. Its only job is to put cells of the same
    identity together so severity can be judged within identity, so a coarse resolution is
    correct: splitting a lineage in two costs nothing, merging two lineages weakens the null
    slightly, and both are far better than judging everything against one sample-wide median.

    Cells below ``min_genes`` are excluded from the grouping and returned as
    :data:`UNASSIGNED`. That floor is an absolute one, not a cohort statistic: a barcode with
    almost no genes cannot be a rare cell type, and including it would let true empties anchor
    a group.

    Args:
        adata: The object being QC'd. Gains ``obsm[PROVISIONAL_EMBEDDING]``; nothing else
            is modified.
        layer: Counts layer to group on. ``None`` uses ``X``.
        resolution: Leiden resolution. Low on purpose.
        n_neighbors: Neighbours for the provisional graph.
        n_pcs: Components for the provisional PCA.
        n_top_genes: HVGs for the provisional embedding.
        min_genes: Absolute floor below which a barcode is not grouped.
        random_state: Seed.

    Returns:
        A per-cell lineage label, indexed like ``adata.obs``, with :data:`UNASSIGNED` for
        cells that could not be placed.
    """
    import scanpy as sc

    matrix = adata.layers[layer] if layer and layer in adata.layers else adata.X
    genes_per_cell = np.asarray((matrix > 0).sum(axis=1)).ravel()
    groupable = genes_per_cell >= min_genes

    labels = pd.Series(UNASSIGNED, index=adata.obs_names, dtype=object)
    if int(groupable.sum()) < max(n_neighbors + 1, 50):
        logger.warning(
            "Provisional lineage grouping skipped: only %d cells clear the %d-gene floor.",
            int(groupable.sum()),
            min_genes,
        )
        return labels

    # A throwaway object, so nothing here can touch the caller's layers or obs.
    work = _provisional_object(adata, matrix, groupable)

    # Per-cell normalization only. target_sum is a constant, so no cohort quantity is
    # estimated and this grouping cannot leak information between cells.
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)

    n_top = int(min(n_top_genes, max(1, work.n_vars - 1)))
    sc.pp.highly_variable_genes(work, n_top_genes=n_top, flavor="seurat")

    comps = int(min(n_pcs, work.n_obs - 1, int(work.var["highly_variable"].sum()) - 1))
    if comps < 2:
        logger.warning("Provisional lineage grouping skipped: too few usable components.")
        return labels

    sc.pp.pca(work, n_comps=comps, mask_var="highly_variable", random_state=random_state)
    sc.pp.neighbors(
        work,
        n_neighbors=int(min(n_neighbors, work.n_obs - 1)),
        use_rep="X_pca",
        random_state=random_state,
    )
    sc.tl.leiden(
        work,
        resolution=resolution,
        random_state=random_state,
        key_added="_lineage",
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    # Keep the embedding: the archetype audit needs one, and computing a second PCA over the
    # cohort to rediscover the same coordinates would be pure waste. Cells below the gene floor
    # get NaN, which is honest — they were never placed.
    embedding = np.full((adata.n_obs, work.obsm["X_pca"].shape[1]), np.nan, dtype=np.float32)
    embedding[groupable] = work.obsm["X_pca"]
    adata.obsm[PROVISIONAL_EMBEDDING] = embedding

    labels.loc[work.obs_names] = ["L" + str(value) for value in work.obs["_lineage"]]
    n_groups = int(pd.Series(labels).nunique())
    logger.info(
        "Provisional lineages: %d groups over %d cells (%d unassigned below the %d-gene floor).",
        n_groups,
        int(groupable.sum()),
        int((~groupable).sum()),
        min_genes,
    )
    return labels


def _provisional_object(
    adata: ad.AnnData, matrix: ExpressionMatrix, groupable: np.ndarray
) -> ad.AnnData:
    """A minimal throwaway AnnData for the provisional embedding."""
    import anndata as ad_module

    return ad_module.AnnData(
        X=matrix[groupable].copy(),
        obs=pd.DataFrame(index=adata.obs_names[groupable]),
        var=pd.DataFrame(index=adata.var_names),
    )


@dataclass(frozen=True)
class NullGrouping:
    """The grouping each cell's severity null is fitted within, plus how it was chosen.

    Carries the key at *every* level rather than only the chosen one, because a fallback null
    must be **estimated over all cells at that level** and merely *applied* to the cells that
    fell back to it.

    Getting that wrong is not subtle. An earlier version stored one key per cell, so cells that
    fell back to the library level formed a group containing only each other. Since the cells
    that fall back are disproportionately the damaged ones, their null was estimated from
    damage — and detection of real damage collapsed from 75% to 10%. The nesting is the whole
    point of a fallback: a wider reference class, not a separate one.

    Args:
        level: Per-cell name of the hierarchy level that supplied the null.
        keys: Group key at each level, for every cell, whether or not it uses that level.
    """

    level: pd.Series
    keys: dict[str, pd.Series]

    def summary(self) -> dict[str, int]:
        """Cells resolved at each hierarchy level, for provenance."""
        return {str(name): int(count) for name, count in self.level.value_counts().items()}

    def levels_used(self) -> list[str]:
        """Hierarchy levels that actually claimed at least one cell."""
        return [name for name in self.keys if bool((self.level == name).any())]


def resolve_null_groups(
    obs: pd.DataFrame,
    *,
    sample_key: str | None,
    lineage: pd.Series | None,
    min_cells: int = 25,
) -> NullGrouping:
    """Choose the finest grouping that still supports a null, per cell.

    The hierarchy, finest first::

        sample x lineage    the ideal: same library, same cell identity
        lineage             a rare type borrows strength across libraries
        sample              no usable lineage, fall back to today's behaviour
        pooled              single-library cohorts

    Resolution is *per cell* rather than one level for the whole dataset, so a rare lineage
    that cannot support its own per-sample null borrows a coarser one while abundant lineages
    keep theirs. That mirrors the ``fallback_groupby`` machinery the legacy mixture path used,
    which was the only place in the codebase that modelled cell types separately.

    Falling back is the conservative direction: a coarser null is wider, so severity is lower,
    so a cell is less likely to be condemned on evidence its own group could not support.

    Args:
        obs: The observation frame.
        sample_key: Library/sample column, or None.
        lineage: Provisional lineage labels, or None to disable lineage conditioning.
        min_cells: Smallest group that can support a null.

    Returns:
        The per-cell grouping and the level that produced it.
    """
    index = obs.index
    sample = (
        obs[sample_key].astype(str)
        if sample_key and sample_key in obs.columns
        else pd.Series("__pooled__", index=index)
    )

    candidates: list[tuple[str, pd.Series]] = []
    if lineage is not None:
        usable = lineage.reindex(index).astype(str)
        # An unassigned cell has no lineage, so lineage levels must not claim it. It still
        # receives a key at the coarser levels, which is how it ends up with a null at all.
        placed = usable != UNASSIGNED
        candidates.append(("sample_x_lineage", (sample + "|" + usable).where(placed)))
        candidates.append(("lineage", usable.where(placed)))
    candidates.append(("sample", sample))
    candidates.append(("pooled", pd.Series("__pooled__", index=index)))

    level = pd.Series(pd.NA, index=index, dtype=object)
    keys_by_level: dict[str, pd.Series] = {}

    for name, keys in candidates:
        keys_by_level[name] = keys
        unresolved = level.isna()
        if not bool(unresolved.any()):
            continue

        # Group size is counted over every cell carrying the key, not only the unresolved
        # ones, because the null is estimated from all of them.
        sizes = keys.groupby(keys).transform("size")
        claimable = unresolved & keys.notna() & (sizes >= min_cells)
        level.loc[claimable] = name

    # The coarsest level is unconditional: pooling always yields one group, so it either
    # works for everyone or for nobody, and leaving a cell without a null would make absent
    # evidence read as health.
    level = level.fillna("pooled")

    grouping = NullGrouping(level=level.astype(object), keys=keys_by_level)
    logger.info("Severity nulls resolved per cell at: %s", grouping.summary())
    return grouping


def lineage_coherence(
    matrix: ExpressionMatrix,
    lineage: pd.Series,
    *,
    detection_fraction: float = 0.5,
) -> pd.Series:
    """Per-lineage consistency of *which* genes are detected — a population versus debris.

    The measure that separates a rare cell type from a debris cluster, and the only one that
    can: on the absolute severity scale the two are identical, and on a within-lineage scale
    both look unremarkable. What differs is coherence.

    A real population expresses the *same* genes in cell after cell. Damage is idiosyncratic —
    each dying cell retains a different random handful — so almost no gene is detected
    consistently across a debris cluster. Measured on the fixture that exposed the defect:

        ordinary lineages    0.96 - 0.98
        rare healthy type    0.990        <- highest of all, despite the lowest RNA content
        debris               0.040

    Note it is deliberately about *which* genes and not *how many*. An erythrocyte-like
    population detecting only fifty genes still scores highly, because every cell detects the
    same fifty. That is what makes the measure safe for low-RNA cell types, which is the whole
    population at risk here.

    The absolute value is depth- and sparsity-dependent, so real cohorts will score far lower
    throughout; compare a lineage against the cohort's own median rather than a fixed bar.

    Args:
        matrix: Cells x genes counts.
        lineage: Per-cell lineage labels.
        detection_fraction: Fraction of a lineage's cells that must detect a gene for it to
            count as consistently detected.

    Returns:
        Coherence in [0, 1] per lineage label.
    """
    import scipy.sparse as sp

    labels = lineage.astype(str)
    scores: dict[str, float] = {}
    for label in sorted(set(labels)):
        rows = (labels == label).to_numpy()
        block = matrix[rows]
        detected = block > 0
        prevalence = (
            np.asarray(detected.mean(axis=0)).ravel()
            if sp.issparse(block)
            else np.asarray(detected).mean(axis=0)
        )
        scores[label] = float((prevalence >= detection_fraction).mean())
    return pd.Series(scores, dtype=float)


def audit_lineages(
    lineage: pd.Series,
    absolute_severity: pd.DataFrame,
    excluded_from_fit: pd.Series,
    probable_multiplet: pd.Series | None = None,
    *,
    suspect_severity: float = 0.667,
    vulnerable_fraction: float = 0.50,
) -> pd.DataFrame:
    """Per-lineage report: is this group a real population, or is it debris?

    Judging severity within lineage fixes the rare-cell defect and introduces one failure of
    its own that must not be left implicit: **a lineage that is uniformly damaged is exonerated
    by its own uniformity.** Every cell in a debris cluster looks ordinary next to its
    neighbours, because its neighbours are also debris.

    That failure cannot be resolved per cell — it is a statement about a group — so it is
    surfaced as a group-level judgement instead. Two flags, meaning opposite things:

    ``suspect``
        The lineage's *absolute* severity is high across the board. Nothing was condemned
        because nothing stood out locally, but the group as a whole looks like debris. A human
        or a downstream annotation step should decide.

    ``vulnerable``
        Most of the lineage is excluded from fitting **for damage reasons**. If the group is real
        biology, this is the rare-population loss this whole module exists to prevent, and it is
        exactly the situation that deleted 50/50 of a healthy synthetic population.

        Multiplet-driven exclusion is deliberately factored out, because conflating the two
        produces a confident false alarm. On the validation cohort a 2,111-cell lineage was
        flagged vulnerable at 83% excluded while carrying the *lowest* absolute severity of any
        lineage (0.127) — it was a doublet cluster: 50.5% called doublets against 1.9%
        cohort-wide, scDblFinder 0.693 vs 0.110, 3,668 genes vs 2,051, and mitochondrial content
        *below* average. Those are excellent libraries that simply are not one cell each.
        Excluding them is QC working, not a population being lost.

    Args:
        lineage: Per-cell provisional lineage.
        absolute_severity: Per-cell, per-family severity computed against *sample-wide* nulls,
            i.e. without lineage conditioning. The absolute scale is what makes "this whole
            group is bad" expressible.
        excluded_from_fit: Per-cell mask of cells barred from fitting.
        probable_multiplet: Per-cell multiplet flag. Supplied so multiplet-driven exclusion can
            be separated from damage-driven exclusion; without it a doublet cluster reads as a
            lost population.
        suspect_severity: Median absolute severity above which a lineage looks like debris.
        vulnerable_fraction: Damage-driven excluded fraction above which a lineage is being lost.

    Returns:
        One row per lineage: size, median absolute severity, excluded fractions, multiplet rate,
        and the flags.
    """
    labels = lineage.astype(str)
    worst = absolute_severity.max(axis=1)
    excluded = excluded_from_fit.astype(bool)
    multiplet = (
        probable_multiplet.reindex(labels.index).fillna(False).astype(bool)
        if probable_multiplet is not None
        else pd.Series(False, index=labels.index)
    )

    frame = pd.DataFrame(
        {
            "n_cells": labels.groupby(labels).size(),
            "median_absolute_severity": worst.groupby(labels).median(),
            "excluded_fraction": excluded.groupby(labels).mean(),
            "multiplet_fraction": multiplet.groupby(labels).mean(),
        }
    )

    # Exclusion among cells that are not multiplets: the share attributable to damage, which is
    # the only kind that can mean a population is being lost.
    not_multiplet = ~multiplet
    damage_excluded = (excluded & not_multiplet).groupby(labels).sum()
    eligible = not_multiplet.groupby(labels).sum()
    frame["damage_excluded_fraction"] = (damage_excluded / eligible.replace(0, np.nan)).fillna(0.0)

    frame["suspect"] = frame["median_absolute_severity"] >= suspect_severity
    frame["vulnerable"] = frame["damage_excluded_fraction"] >= vulnerable_fraction
    frame = frame.sort_values("median_absolute_severity", ascending=False)

    for label, row in frame[frame["suspect"] | frame["vulnerable"]].iterrows():
        logger.warning(
            "Lineage %s (n=%d): median absolute severity %.2f, %.0f%% excluded "
            "(%.0f%% of non-multiplets, %.0f%% called multiplets)%s%s",
            label,
            int(row["n_cells"]),
            row["median_absolute_severity"],
            100.0 * row["excluded_fraction"],
            100.0 * row["damage_excluded_fraction"],
            100.0 * row["multiplet_fraction"],
            " [SUSPECT: may be debris]" if row["suspect"] else "",
            " [VULNERABLE: may be a real population being lost]" if row["vulnerable"] else "",
        )
    return frame


__all__ = [
    "LINEAGE_COLUMN",
    "PROVISIONAL_EMBEDDING",
    "NULL_LEVEL_COLUMN",
    "UNASSIGNED",
    "NullGrouping",
    "audit_lineages",
    "provisional_lineages",
    "resolve_null_groups",
]
