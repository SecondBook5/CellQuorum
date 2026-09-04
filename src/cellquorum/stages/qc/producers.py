# Pipeline step (order=20): qc — turn measured metrics into graded evidence axes.
"""Evidence producers: metrics in, :class:`EvidenceTable` out.

The one place that decides how a measured number becomes a severity, and the reason the
adjudicator can compare families at all.

    1. The severity definition   what 0.7 means, identically on every axis
    2. Robust nulls              estimating the healthy mode, and knowing when we cannot
    3. Axis builders             one per evidence family
    4. Assembly                  build_evidence_table

## Severity is a tail statement about the healthy population

    severity ~= P(a healthy cell in this group looks this bad or worse on this axis)

That definition is what makes one ``concern_severity`` bar meaningful. The first version of
this module scaled each metric from its median to its 99.9th percentile, which is
*relative*: the same nominal 0.50 flagged 13% of cells on complexity and 1.5% on stress —
a 9x difference in stringency produced by the mapping rather than by any decision. Every
relative scaling has that defect, and plain percentiles have a worse one, because they
force a uniform distribution and so condemn the worst few per cent of even a pristine
sample.

Here severity is a saturating function of a **robust z against the healthy mode**, so it
carries the same meaning on every axis and on every dataset. A cell inside the healthy mode
scores near zero however many cells share its sample; a genuinely damaged cell scores high
because it sits many robust deviations out, not because it occupies some sample's tail.

Median and MAD are the estimators precisely because they describe the *healthy mode* of a
contaminated distribution rather than the whole spread — the property that makes MAD the
wrong tool for a threshold and the right tool for a null.

## When the null cannot be estimated, say so

A group with no usable spread, or too few cells, yields no severity. Those cells become
``COMPUTATION_FAILED``, never a comfortable zero. The adjudicator already withholds quarantine
as coverage falls, so an axis that cannot speak makes the system more conservative rather
than more permissive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

from cellquorum.stages.qc.evidence import (
    AxisEvidence,
    Direction,
    EvidenceAvailability,
    EvidenceFamily,
    EvidenceTable,
    build_axis,
)
from cellquorum.stages.qc.lineage import NullGrouping

# ─── 1. The severity definition ─────────────────────────────────────────────────────

#: Robust z at which severity reaches 0.5 — "as likely abnormal as not". The single knob
#: for the whole scale, and a statement about outlyingness rather than about any particular
#: metric, so it transfers across assays and tissues. Three robust deviations from the
#: healthy mode is the conventional line between "unusual" and "probably not this
#: population".
DEFAULT_HALF_SEVERITY_Z = 3.0

#: MAD -> sigma for a normal healthy mode. Without it a "robust z" is not on a sigma scale
#: and the shared bar stops meaning anything.
MAD_TO_SIGMA = 1.4826

#: Upper-quantile fallbacks for the scale, tried in order when MAD is exactly zero. Each
#: pair is (quantile, the normal z at that quantile) so the estimate stays on a sigma scale.
#: Zero-inflated fractions such as MALAT1 or the stress programme have a median of 0 and a
#: MAD of 0 — and often a q75 of 0 as well — which is a property of the metric rather than a
#: reason to abandon the axis. Walking up the quantiles finds the first one with spread.
QUANTILE_SCALE_FALLBACKS: tuple[tuple[float, float], ...] = (
    (0.75, 0.6745),
    (0.90, 1.2816),
    (0.99, 2.3263),
)

#: Smallest group that can support a null. Below this the location and scale are noise.
MIN_CELLS_FOR_NULL = 25


def _saturating_severity(z: pd.Series, half_severity_z: float) -> pd.Series:
    """Map a one-sided robust z to severity in ``[0, 1)``.

    ``z / (z + k)``: zero at the healthy mode, 0.5 at ``k`` deviations, asymptotic to 1.
    Monotone and bounded, so a single extreme outlier cannot dominate a family rollup, and
    no distributional assumption is needed beyond the healthy mode having a location and a
    scale.

    Args:
        z: One-sided robust z; negative means "not concerning".
        half_severity_z: z at which severity is 0.5.
    """
    positive = z.clip(lower=0.0)
    return positive / (positive + half_severity_z)


# ─── 2. Robust nulls ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RobustNull:
    """Per-cell location and scale of the healthy mode of its group.

    Args:
        location: Group median, broadcast per cell.
        scale: Robust sigma of the group's healthy mode; NaN where inestimable.
    """

    location: pd.Series
    scale: pd.Series

    def z(self, values: pd.Series, *, direction: Direction) -> pd.Series:
        """One-sided robust z, oriented so positive always means "concerning"."""
        signed = (values - self.location) / self.scale
        return signed if direction is Direction.UPPER_TAIL else -signed


def fit_robust_null(values: pd.Series, groups: pd.Series | None) -> RobustNull:
    """Estimate the healthy mode's location and scale, per group.

    Scale is ``1.4826 * MAD``, walking up :data:`QUANTILE_SCALE_FALLBACKS` when MAD is
    exactly zero — the normal case for a zero-inflated fraction, where over half the cells
    share one value. A group supporting none of them, or smaller than
    :data:`MIN_CELLS_FOR_NULL`, gets a NaN scale, which propagates to NaN severity and so to
    ``COMPUTATION_FAILED`` rather than to a false "no concern".

    Args:
        values: The metric, per cell.
        groups: Grouping to fit within, normally the cohort sample key. None pools, which
            is only correct for a single library.
    """
    if groups is None:
        groups = pd.Series("__pooled__", index=values.index)

    grouped = values.groupby(groups)
    location = grouped.transform("median")

    absolute_deviation = (values - location).abs()
    scale = absolute_deviation.groupby(groups).transform("median") * MAD_TO_SIGMA

    # Zero MAD reflects zero inflation, not failure. Walk up the quantiles until one shows
    # spread; a fraction that is zero for 90% of cells still separates its tail.
    for quantile, normal_z in QUANTILE_SCALE_FALLBACKS:
        if bool((scale > 0).all()):
            break
        upper = grouped.transform(lambda group, q=quantile: group.quantile(q))
        scale = scale.where(scale > 0, (upper - location) / normal_z)

    # A group too small for a stable median, or with genuinely no spread, is not a null.
    n_cells = grouped.transform("size")
    scale = scale.where((scale > 0) & (n_cells >= MIN_CELLS_FOR_NULL))

    return RobustNull(location=location, scale=scale)


def tail_severity(
    values: pd.Series,
    groups: pd.Series | None,
    *,
    direction: Direction,
    log_scale: bool = False,
    half_severity_z: float = DEFAULT_HALF_SEVERITY_Z,
) -> pd.Series:
    """Severity for one metric: robust z against its healthy mode, saturated.

    Args:
        values: The metric, per cell.
        groups: Grouping to fit the null within.
        direction: Which tail is concerning.
        log_scale: Fit the null on ``log1p`` values. Correct for count-like metrics, whose
            healthy mode is right-skewed on the raw scale, so a symmetric robust z there
            would systematically over-flag the low side.
        half_severity_z: z at which severity is 0.5.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    prepared = np.log1p(numeric.clip(lower=0.0)) if log_scale else numeric

    null = fit_robust_null(prepared, groups)
    return _saturating_severity(null.z(prepared, direction=direction), half_severity_z)


def nested_tail_severity(
    values: pd.Series,
    grouping: NullGrouping,
    *,
    direction: Direction,
    log_scale: bool = False,
    half_severity_z: float = DEFAULT_HALF_SEVERITY_Z,
) -> pd.Series:
    """Severity where each cell is scored against the reference class it was assigned.

    The null at every level is estimated over **all** cells carrying that level's key, and only
    then applied to the cells assigned to that level. That nesting is the entire point of a
    fallback and is easy to get wrong in a way that silently inverts the result.

    The wrong version handed one key per cell to the fitter, so cells that fell back to the
    library level were compared only against each other. Cells fall back precisely when their
    own group is too small or unusable, which selects for damaged barcodes — so their null was
    estimated from damage, and detection of real damage fell from 75% to 10% while every
    rare-population test still passed. A coarser reference class must be *wider*, never
    *separate*.

    Args:
        values: The metric, per cell.
        grouping: Level assignment plus each level's keys for every cell.
        direction: Which tail is concerning.
        log_scale: Fit the null on ``log1p`` values, for count-like metrics.
        half_severity_z: z at which severity is 0.5.

    Returns:
        Severity per cell, NaN where the assigned level could not support a null.
    """
    severity = pd.Series(np.nan, index=values.index, dtype=float)
    for level in grouping.levels_used():
        assigned = (grouping.level == level).reindex(values.index, fill_value=False)
        if not bool(assigned.any()):
            continue
        at_level = tail_severity(
            values,
            grouping.keys[level].reindex(values.index),
            direction=direction,
            log_scale=log_scale,
            half_severity_z=half_severity_z,
        )
        severity[assigned] = at_level[assigned]
    return severity


def axis_from_severity(
    *,
    name: str,
    family: EvidenceFamily,
    direction: Direction,
    severity: pd.Series,
    weight: float = 1.0,
) -> AxisEvidence:
    """Build an axis whose availability follows from whether a severity was produced.

    Availability must be a consequence of the computation, not a separate claim about it.
    Deriving the two independently is how they drift: an earlier version asserted
    ``AVAILABLE_VALID`` from ``value.notna()`` while the severity formula returned NaN for
    158,414 cells whose group had a degenerate spread. The engine rejected it, correctly.
    """
    usable = severity.notna()
    return build_axis(
        name=name,
        family=family,
        direction=direction,
        severity=severity,
        availability=pd.Series(
            np.where(
                usable,
                str(EvidenceAvailability.AVAILABLE_VALID),
                # COMPUTATION_FAILED, not MODEL_UNSTABLE: the latter means "we produced a
                # number but the fit is shaky", so it counts as usable and must carry a
                # value. No number at all is a different state, and conflating them puts a
                # NaN behind a usable flag — which the engine rejects outright.
                str(EvidenceAvailability.COMPUTATION_FAILED),
            ),
            index=severity.index,
        ),
        weight=weight,
    )


# ─── 3. Axis builders ───────────────────────────────────────────────────────────────

#: Nuclear-retained lncRNA. A cell that has leaked cytoplasm is enriched for it relative to
#: its remaining transcriptome. Assay-dependent: expected high in single-nucleus data.
NUCLEAR_RETAINED_GENE = "MALAT1"

#: Immediate-early and heat-shock dissociation-stress programme (van den Brink et al. 2017
#: and the widely reused core set). Elevated by dissociation, but also genuinely elevated in
#: inflamed tissue — hence supporting evidence that can never condemn a cell alone.
DISSOCIATION_STRESS_GENES: tuple[str, ...] = (
    "FOS",
    "FOSB",
    "JUN",
    "JUNB",
    "JUND",
    "EGR1",
    "ATF3",
    "IER2",
    "HSPA1A",
    "HSPA1B",
    "HSPB1",
    "HSPH1",
    "DNAJB1",
    "DNAJA1",
    "SOCS3",
    "ZFP36",
    "DUSP1",
    "KLF6",
    "NR4A1",
    "PPP1R15A",
)

#: Doublet detectors, most informative first. Scores are combined only after each has been
#: put on its own healthy-mode scale.
DOUBLET_SCORE_COLUMNS: tuple[str, ...] = (
    "doublet_score_scdblfinder",
    "doublet_score_scrublet",
    "doublet_score",
)

#: Count-like metrics whose healthy mode is right-skewed, so their nulls are fitted on
#: ``log1p``.
_CAPTURE_METRICS: tuple[str, ...] = ("n_genes_by_counts", "total_counts")


def gene_fraction(
    adata: AnnData,
    genes: tuple[str, ...],
    total: pd.Series,
    *,
    layer: str | None = None,
) -> pd.Series | None:
    """Fraction of a cell's counts falling in ``genes``, or None if none are present.

    Sums via a sparse matrix-vector product against a 0/1 gene indicator rather than
    slicing the object. ``adata[:, genes]`` looks equivalent and is not: it runs scipy
    fancy-indexing over CSR for ``X`` *and every layer*, which at cohort scale
    (201,923 x 36,601, three matrices) segfaulted on int32 index arithmetic. A matvec
    touches each nonzero once and allocates one output vector.
    """
    positions = {name: index for index, name in enumerate(adata.var_names)}
    columns = [positions[gene] for gene in genes if gene in positions]
    if not columns:
        return None

    matrix = adata.layers[layer] if layer and layer in adata.layers else adata.X
    indicator = np.zeros((matrix.shape[1], 1), dtype=np.float64)
    indicator[columns, 0] = 1.0
    summed = np.asarray(matrix @ indicator).ravel()

    fraction = pd.Series(summed, index=pd.Index(adata.obs_names), dtype=float) / total
    return fraction.replace([np.inf, -np.inf], np.nan)


def multiplet_agreement_severity(
    obs: pd.DataFrame,
    groups: pd.Series | None,
    *,
    half_severity_z: float = DEFAULT_HALF_SEVERITY_Z,
) -> pd.Series | None:
    """Multiplet severity requiring detectors to agree, on comparable scales.

    Each detector is put on its own healthy-mode scale *first*, then the **minimum** across
    detectors is taken. The order is the whole point: on the lymphedema cohort scDblFinder
    ran at median 0.110 and Scrublet at 0.031, so a minimum of the *raw* scores returned
    Scrublet verbatim — not agreement, merely whichever detector was systematically more
    conservative — and its maximum of 0.816 never reached the bar, so the family flagged
    nothing across 201,923 cells. Normalised first, the minimum means what it should: both
    detectors rate this cell extreme for their own distribution.
    """
    present = [column for column in DOUBLET_SCORE_COLUMNS if column in obs.columns]
    if not present:
        return None

    # `doublet_score` is normally a copy of whichever detector ran; counting it beside its
    # own source would let one detector satisfy the agreement requirement twice.
    if len(present) > 1 and "doublet_score" in present:
        present = [column for column in present if column != "doublet_score"]

    severities = [
        tail_severity(
            obs[column],
            groups,
            direction=Direction.UPPER_TAIL,
            half_severity_z=half_severity_z,
        )
        for column in present
    ]

    # skipna=False so a detector that could not score a cell leaves the agreement unknown
    # rather than silently deferring to the other detector.
    return pd.concat(severities, axis=1).min(axis=1, skipna=False)


# ─── 4. Assembly ────────────────────────────────────────────────────────────────────


def build_evidence_table(
    adata: AnnData,
    cell_metrics: pd.DataFrame,
    *,
    group_key: str | None = None,
    layer: str | None = None,
    mito_posterior: pd.Series | None = None,
    nuclear_axis_applicable: bool = True,
    half_severity_z: float = DEFAULT_HALF_SEVERITY_Z,
    grouping: NullGrouping | None = None,
    lineage_conditional: bool = False,
) -> EvidenceTable:
    """Assemble every evidence axis this dataset supports.

    Axes with no usable input are simply absent, which lowers evidence coverage and makes
    the adjudicator more conservative — the honest behaviour when we know less.

    Args:
        adata: The QC AnnData, used for gene-level axes.
        cell_metrics: Per-cell QC metrics, indexed like ``adata.obs``.
        group_key: ``obs`` column to fit nulls within, normally the cohort sample key. Used
            only when ``grouping`` is not supplied.
        layer: Layer holding the counts the gene-fraction axes should use.
        mito_posterior: Per-cell compromised probability from the mixture model. Used
            directly: it already estimates the quantity severity approximates, from a model
            that captures the joint mito-complexity relationship of a dying cell rather
            than treating mitochondrial fraction as unimodal noise.
        nuclear_axis_applicable: False for single-nucleus assays.
        half_severity_z: Robust z at which severity is 0.5.
        grouping: Per-cell reference classes from
            :func:`cellquorum.stages.qc.lineage.resolve_null_groups`, with each level's null
            estimated over every cell at that level. Overrides ``group_key``.
        lineage_conditional: True when ``grouping`` carries cell identity as well as library.
            Recorded for provenance only — it deliberately changes no behaviour here. An earlier
            version used it to re-scale the mitochondrial posterior within lineage, which
            corrupted an already-calibrated probability; see the metabolic axis below. Calibrating
            the posterior for cell identity is the mixture model's job, not this module's.
    """
    obs = adata.obs
    groups = obs[group_key] if group_key and group_key in obs.columns else None

    def severity_of(
        values: pd.Series,
        *,
        direction: Direction,
        log_scale: bool = False,
    ) -> pd.Series:
        """Score one metric against each cell's reference class.

        One entry point so no axis can accidentally skip the nested fallback and be scored
        against a group made only of the cells that fell back to it.
        """
        if grouping is not None:
            return nested_tail_severity(
                values,
                grouping,
                direction=direction,
                log_scale=log_scale,
                half_severity_z=half_severity_z,
            )
        return tail_severity(
            values,
            groups,
            direction=direction,
            log_scale=log_scale,
            half_severity_z=half_severity_z,
        )

    axes: list[AxisEvidence] = []

    def add(
        name: str,
        family: EvidenceFamily,
        direction: Direction,
        severity: pd.Series,
        weight: float = 1.0,
    ) -> None:
        axes.append(
            axis_from_severity(
                name=name,
                family=family,
                direction=direction,
                severity=severity,
                weight=weight,
            )
        )

    # --- capture / complexity ------------------------------------------------------- #
    for metric in _CAPTURE_METRICS:
        if metric in cell_metrics:
            add(
                metric,
                EvidenceFamily.CAPTURE_COMPLEXITY,
                Direction.LOWER_TAIL,
                severity_of(
                    cell_metrics[metric],
                    direction=Direction.LOWER_TAIL,
                    log_scale=True,
                ),
            )

    # --- metabolic / stress --------------------------------------------------------- #
    if mito_posterior is not None:
        # Used as an ABSOLUTE probability, never re-scored — not even when lineages are
        # available. That distinction was got wrong once and cost 22,541 cells.
        #
        # The posterior is already a calibrated statement: "probability this cell is
        # compromised". Passing it through a robust z answers a different question — "is this
        # cell's compromise-probability unusual for its lineage?" — and the two diverge badly
        # because the posterior is sharply peaked near zero. On the validation cohort its median
        # was 0.035 with a MAD of 0.015, so a cell at posterior 0.10 sat 4.3 sigma out and scored
        # severity 0.59, while the model itself put the chance of anything being wrong at 10%.
        # Cells reaching >= 0.50 on this axis went from 10.7% to 18.8% of the cohort.
        #
        # The problem re-scoring was reaching for is real: a constitutively high-mitochondrial
        # cell type receives a high posterior on biology alone. But the fix belongs in the
        # mixture model, which already supports fitting per group — give it the provisional
        # lineage and the posterior comes back calibrated within lineage, needing no rescaling
        # anywhere. Never reshape a calibrated probability to compensate for it having been
        # fitted on the wrong population.
        add(
            "mito_mixture_posterior",
            EvidenceFamily.METABOLIC_STRESS,
            Direction.UPPER_TAIL,
            mito_posterior.reindex(cell_metrics.index).astype(float),
        )
    elif "pct_counts_mito" in cell_metrics:
        add(
            "pct_counts_mito",
            EvidenceFamily.METABOLIC_STRESS,
            Direction.UPPER_TAIL,
            severity_of(
                cell_metrics["pct_counts_mito"],
                direction=Direction.UPPER_TAIL,
            ),
        )

    total_counts = cell_metrics.get("total_counts")
    if total_counts is not None:
        total_counts = total_counts.astype(float)

        stress = gene_fraction(adata, DISSOCIATION_STRESS_GENES, total_counts, layer=layer)
        if stress is not None:
            add(
                "dissociation_stress",
                EvidenceFamily.METABOLIC_STRESS,
                Direction.UPPER_TAIL,
                severity_of(
                    stress,
                    direction=Direction.UPPER_TAIL,
                ),
                # Tracks disease biology as much as damage, so reduced weight even inside
                # its own family.
                weight=0.6,
            )

        # --- nuclear / cytoplasmic integrity --------------------------------------- #
        if nuclear_axis_applicable:
            nuclear = gene_fraction(adata, (NUCLEAR_RETAINED_GENE,), total_counts, layer=layer)
            if nuclear is not None:
                add(
                    "malat1_fraction",
                    EvidenceFamily.NUCLEAR_INTEGRITY,
                    Direction.UPPER_TAIL,
                    severity_of(
                        nuclear,
                        direction=Direction.UPPER_TAIL,
                    ),
                )

    # --- multiplet ------------------------------------------------------------------ #
    #
    # Deliberately scored per library and NOT per lineage, unlike every damage axis above.
    # Doublets frequently form a cluster of their own, so a within-lineage null would let a
    # doublet cluster exonerate itself — and unlike a rare cell type, "doublet" is not an
    # identity that deserves its own reference class. The asymmetry is the point: lineage
    # conditioning protects cells for *being what they are*, which is exactly what a doublet
    # is not.
    multiplet = multiplet_agreement_severity(obs, groups, half_severity_z=half_severity_z)
    if multiplet is not None:
        add("doublet_agreement", EvidenceFamily.MULTIPLET, Direction.UPPER_TAIL, multiplet)

    return EvidenceTable(axes=tuple(axes), obs_names=pd.Index(adata.obs_names))


__all__ = [
    "DEFAULT_HALF_SEVERITY_Z",
    "DISSOCIATION_STRESS_GENES",
    "DOUBLET_SCORE_COLUMNS",
    "MIN_CELLS_FOR_NULL",
    "NUCLEAR_RETAINED_GENE",
    "RobustNull",
    "axis_from_severity",
    "build_evidence_table",
    "fit_robust_null",
    "gene_fraction",
    "multiplet_agreement_severity",
    "tail_severity",
]
