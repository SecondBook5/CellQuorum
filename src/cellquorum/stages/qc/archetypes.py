# Pipeline step (order=20): qc — archetype audit, asking whether QC is eating a population.
"""Archetype audit: is there a coherent population that QC is quietly removing?

Clustering answers "where are the dense blobs". Archetypal analysis answers "where are the
extremes", and those are different questions with different blind spots. Leiden needs density,
so a fifty-cell population is exactly what it merges into a neighbour; a polytope vertex does
not care how few cells support it. Rare populations are the case clustering is *structurally*
bad at, which is why this exists alongside the lineage grouping rather than instead of it.

## What this can and cannot conclude

It cannot decide damage. On the severity axes a rare cell type and a dying cell are both
extreme, and the measurement that motivated this whole area showed they are geometrically
identical. So a vertex is evidence of **coherent extremeness**, never of health.

What it can do is audit QC from the outside: if an archetype's supporting cells are
disproportionately barred from fitting, either QC is removing a real population or the
archetype is debris — and those two are distinguishable, because debris is incoherent. The
audit therefore reports both the exclusion rate and the coherence, and leaves the call to a
human. That is deliberate: this is the one question the automated verdict cannot ask about
itself.

## Optional by construction

partipy is GPL-3 and CellQuorum is BSD-3, so it runs in an isolated environment through a
subprocess backend and is never imported here. Absent that environment the audit reports
itself unavailable and nothing else in the run changes.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.stages.qc._types import ExpressionMatrix, IsolatedBackend

logger = logging.getLogger(__name__)

#: obs column holding each cell's dominant archetype.
ARCHETYPE_COLUMN = "qc_archetype"

#: How far above uniform a cell's weight must sit for it to count as *supporting* an archetype,
#: as a multiple of ``1 / n_archetypes``. Relative rather than absolute because the weights are
#: normalised across however many vertices were fitted: with eight archetypes a cell spread
#: evenly carries 0.125 each, so a fixed bar of 0.5 demands four times uniform and effectively
#: excludes everyone.
#:
#: That is not hypothetical — a fixed 0.5 produced an archetype with 6,482 dominant cells and
#: zero supporting ones on the validation cohort, making its exclusion rate NaN and the audit
#: silent about it. Twice uniform keeps the meaning ("distinctly nearer this vertex than an
#: average cell") at any polytope size.
SUPPORT_MULTIPLE_OF_UNIFORM = 2.0


@dataclass(frozen=True)
class ArchetypeAudit:
    """Result of the archetype audit.

    Args:
        available: False when the partipy environment is absent; every other field is then
            empty and the run is unaffected.
        reason: Why the audit is unavailable, when it is.
        n_archetypes: Vertices fitted.
        t_ratio: Polytope tightness. Higher means the simplex describes the data better.
        t_ratio_pvalue: Permutation p-value for the t-ratio, or None when unavailable. Without
            it, "we found archetypes" is not a finding — a polytope can always be fitted.
        polytope_supported: False when the p-value says the simplex does not describe the
            data. Nothing is flagged in that case, and the distinction matters: "no
            population is being lost" and "this method could not tell" are different
            statements and must not both surface as silence.
        dominant: Per-cell dominant archetype label.
        table: One row per archetype: support size, exclusion rate, coherence, flags.
    """

    available: bool
    reason: str | None = None
    n_archetypes: int = 0
    t_ratio: float | None = None
    t_ratio_pvalue: float | None = None
    polytope_supported: bool = True
    dominant: pd.Series | None = None
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def flagged(self) -> pd.DataFrame:
        """Archetypes worth a human's attention."""
        if self.table.empty:
            return self.table
        return self.table[self.table["losing_a_population"] | self.table["probably_debris"]]


def audit_archetypes(
    embedding: np.ndarray,
    obs_names: pd.Index,
    excluded_from_fit: pd.Series,
    counts: ExpressionMatrix | None = None,
    *,
    backend: IsolatedBackend | None = None,
    n_archetypes_min: int = 3,
    n_archetypes_max: int = 10,
    bootstrap: int = 0,
    seed: int = 0,
    scratch_dir: str | Path | None = None,
    excluded_fraction_bar: float = 0.50,
    max_cells: int = 10_000,
    n_restarts: int = 1,
    timeout_seconds: int = 900,
    max_pvalue: float = 0.05,
) -> ArchetypeAudit:
    """Fit archetypes and report which ones QC is removing.

    Args:
        embedding: Cells x components, the provisional embedding QC already computed. Reused
            rather than recomputed, so the audit costs one subprocess call and no new PCA.
        obs_names: Cell names, for aligning the per-cell result.
        excluded_from_fit: Per-cell mask of cells barred from fitting anything.
        counts: Optional cells x genes counts. When given, coherence is computed *per
            archetype* from it, which is what separates "a real population is being lost" from
            "this vertex is debris" — the two situations that produce an identical exclusion
            rate and demand opposite responses.
        backend: A :class:`~cellquorum.backends.partipy_backend.PartipyBackend`, or None to
            build the default one.
        n_archetypes_min: Smallest polytope considered.
        n_archetypes_max: Largest polytope considered.
        bootstrap: Bootstrap replicates for vertex stability. 0 skips it.
        seed: Seed passed through to partipy.
        scratch_dir: Directory for file exchange with the isolated environment.
        excluded_fraction_bar: Exclusion rate above which an archetype is flagged.
        max_cells: Cap on cells entering the fit; above it a uniform random subsample is used.
            Necessary, not merely prudent: archetypal analysis is a nonnegative least-squares
            problem per cell per iteration, and an uncapped fit on the 201,923-cell validation
            cohort sat at 0% CPU for 27 minutes and blocked the run. A uniform subsample leaves
            exclusion rates unbiased and still contains ~2,000 cells of a population at 10%
            frequency, which is ample for the question being asked.
        n_restarts: Restarts per candidate archetype count. The selection sweep is one fit per
            candidate, so restarts multiply the whole sweep.
        timeout_seconds: Hard cap on the subprocess. An audit must never be able to hang a run;
            exceeding it degrades to unavailable.
        max_pvalue: Permutation p-value above which the polytope is treated as unsupported and
            **no archetype is flagged**. A polytope can be fitted to anything, so without this
            the audit invents findings on data that has no simplex structure: two Gaussian blobs
            produced a t-ratio p-value of 0.92 and still raised two "a real population is being
            removed" alarms. The vertices of a polytope that does not describe the data are
            artifacts of the fit, not populations.

    Returns:
        The audit. ``available=False`` when the partipy environment is absent.
    """
    from cellquorum.backends.partipy_backend import ARCHETYPE_HELPER, build_partipy_backend

    partipy = backend if backend is not None else build_partipy_backend()
    status = partipy.status()
    if not status.available:
        reason = (
            f"partipy environment unavailable (missing: {', '.join(status.missing) or 'unknown'})"
        )
        logger.info("Archetype audit skipped: %s", reason)
        return ArchetypeAudit(available=False, reason=reason)

    scratch = Path(scratch_dir) if scratch_dir is not None else Path(tempfile.gettempdir())
    scratch.mkdir(parents=True, exist_ok=True)

    # Subsample for the fit. Deterministic, so a rerun audits the same cells.
    n_cells = int(embedding.shape[0])
    if n_cells > max_cells:
        picked = np.sort(
            np.random.default_rng(seed).choice(n_cells, size=int(max_cells), replace=False)
        )
        logger.info(
            "Archetype audit fitting on a %d-cell uniform subsample of %d.", max_cells, n_cells
        )
    else:
        picked = np.arange(n_cells)

    fit_embedding = embedding[picked]
    fit_names = obs_names[picked]

    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        tmp_path = Path(tmp)
        embedding_path = tmp_path / "embedding.npy"
        weights_path = tmp_path / "weights.npy"
        meta_path = tmp_path / "meta.json"
        np.save(embedding_path, np.ascontiguousarray(fit_embedding, dtype=np.float64))

        try:
            result = partipy.run_helper(
                ARCHETYPE_HELPER,
                [
                    "fit",
                    str(embedding_path),
                    str(weights_path),
                    str(meta_path),
                    "--n-archetypes-min",
                    str(int(n_archetypes_min)),
                    "--n-archetypes-max",
                    str(int(n_archetypes_max)),
                    "--seed",
                    str(int(seed)),
                    "--bootstrap",
                    str(int(bootstrap)),
                    "--n-restarts",
                    str(int(n_restarts)),
                ],
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            reason = f"partipy exceeded {timeout_seconds}s and was abandoned"
            logger.warning("Archetype audit skipped: %s", reason)
            return ArchetypeAudit(available=False, reason=reason)
        if result.returncode != 0 or not weights_path.is_file():
            reason = f"partipy helper failed: {result.stderr.strip()[:300] or 'no stderr'}"
            logger.warning("Archetype audit skipped: %s", reason)
            return ArchetypeAudit(available=False, reason=reason)

        weights = np.load(weights_path)
        meta = json.loads(meta_path.read_text())

    labels = [f"A{index}" for index in range(weights.shape[1])]
    # Indexed by the fitted cells only. Cells outside the subsample carry no archetype, which
    # the caller records as "unsampled" rather than inventing an assignment for them.
    dominant = pd.Series(
        [labels[index] for index in weights.argmax(axis=1)], index=fit_names, dtype=object
    )

    # Only cells that genuinely sit near a vertex count toward it. An interior cell is a
    # mixture and says nothing about any single extreme phenotype. The bar scales with the
    # polytope size, since "near a vertex" means something different with 3 vertices than 10.
    support_bar = SUPPORT_MULTIPLE_OF_UNIFORM / float(weights.shape[1])
    supports = weights.max(axis=1) >= support_bar
    excluded = excluded_from_fit.reindex(fit_names).astype(bool).to_numpy()

    # Coherence per archetype, not per lineage: a vertex is the unit being judged here.
    # Consistency of *which* genes are detected is the only signal that separates a rare
    # population from debris, since both are extreme and both are excluded.
    coherence_by_archetype = None
    if counts is not None:
        from cellquorum.stages.qc.lineage import lineage_coherence

        coherence_by_archetype = lineage_coherence(counts[picked], dominant)

    rows = []
    for index, label in enumerate(labels):
        near = supports & (weights.argmax(axis=1) == index)
        n_support = int(near.sum())
        rows.append(
            {
                "archetype": label,
                "n_supporting": n_support,
                "excluded_fraction": float(excluded[near].mean()) if n_support else float("nan"),
                "coherence": (
                    float(coherence_by_archetype.get(label, float("nan")))
                    if coherence_by_archetype is not None
                    else float("nan")
                ),
            }
        )

    table = pd.DataFrame(rows).set_index("archetype")
    over_bar = table["excluded_fraction"] >= excluded_fraction_bar

    # Coherence, when available, splits the flag in two. Debris is incoherent by nature, so an
    # excluded-but-incoherent vertex is QC working; excluded-and-coherent is QC failing.
    if coherence_by_archetype is not None and table["coherence"].notna().any():
        median_coherence = float(table["coherence"].median())
        coherent = table["coherence"] >= 0.35 * median_coherence
    else:
        coherent = pd.Series(True, index=table.index)

    # A polytope that does not describe the data has no meaningful vertices, so nothing is
    # flagged. The table is still returned, so a reader can see what was measured and why it
    # was disregarded.
    pvalue = meta.get("t_ratio_pvalue")
    supported = pvalue is None or float(pvalue) <= max_pvalue
    if not supported:
        logger.info(
            "Archetype audit found no significant polytope (t-ratio p=%.3g > %.3g); "
            "reporting the table but flagging nothing.",
            float(pvalue),
            max_pvalue,
        )

    table["losing_a_population"] = over_bar & coherent & supported
    table["probably_debris"] = over_bar & ~coherent & supported
    table = table.sort_values("excluded_fraction", ascending=False)

    audit = ArchetypeAudit(
        available=True,
        n_archetypes=int(meta.get("n_archetypes", weights.shape[1])),
        t_ratio=meta.get("t_ratio"),
        t_ratio_pvalue=meta.get("t_ratio_pvalue"),
        polytope_supported=bool(supported),
        dominant=dominant,
        table=table,
    )

    for archetype_label, row in audit.flagged().iterrows():
        logger.warning(
            "Archetype %s (n=%d): %.0f%% excluded from fitting — %s",
            str(archetype_label),
            int(row["n_supporting"]),
            100.0 * row["excluded_fraction"],
            (
                "a coherent extreme population is being removed; check it is not real biology"
                if row["losing_a_population"]
                else "incoherent, so most likely debris QC is correctly removing"
            ),
        )
    return audit


__all__ = [
    "ARCHETYPE_COLUMN",
    "SUPPORT_MULTIPLE_OF_UNIFORM",
    "ArchetypeAudit",
    "audit_archetypes",
]
