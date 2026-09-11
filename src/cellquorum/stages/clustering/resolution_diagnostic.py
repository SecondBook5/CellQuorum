"""Bootstrap cluster-stability diagnostic across a Leiden resolution sweep.

Answers a narrower, more defensible question than "how many clusters are there": at a
given resolution, is the partition robust to resampling noise, or would a slightly
different draw of the same cells split, merge, or dissolve it? That robustness is the
closest available proxy to ground truth when no labeled reference exists -- but it is a
proxy for statistical robustness, not biological correctness. A batch effect can be just
as reproducible as real biology; this diagnostic answers "is this structure real
(reproducible)", not "is this structure meaningful (a true population)". The latter is
what downstream marker-gene validation and adjudication are for.

## Method

Bootstrap-subsample stability, the approach behind chooseR (Patterson-Cross, Levine &
Bhaduri, 2021) and similar to scclusteval. At each resolution: fit a REFERENCE partition
on every fit-eligible cell, then repeatedly subsample a fraction of those cells and
recluster independently. Each reference cluster is scored by the Jaccard overlap to its
best-matching cluster in the subsample -- matching by cell-membership overlap, not by
label, since Leiden's own integer labels are not stable identifiers across independent
runs. High, stable Jaccard across bootstraps means the partition survives resampling.

## Adaptive allocation

A full N-resolution x M-bootstrap sweep is expensive, and most of that budget is wasted
on resolutions that are already obviously stable or obviously not. Two rounds instead:

    Round 1: a small fraction of the bootstrap budget at EVERY resolution, to get a
             rough stability curve (median + spread per resolution).
    Round 2: the remaining budget, weighted toward resolutions that are either
             UNCERTAIN (wide spread in round 1 -- the estimate itself isn't trustworthy
             yet) or sit next to a TRANSITION (a large jump in median stability between
             neighboring resolutions -- is that a real elbow, or noise). Resolutions
             already flat and confident get little or no extra budget.

This is not classic successive-halving (which discards candidates to find one winner):
every resolution keeps at least its round-1 estimate, since the deliverable is a full
curve, not a single selected value. Nothing here selects a resolution automatically --
the figure this feeds is a diagnostic for a human to read, matching how this pipeline
treats every other ambiguous call (e.g. adjudication proposes evidence, never decides
identity on its own).

GPU-routed via `run_neighbors_leiden_routed`, shared with LeidenMethod so the diagnostic
cannot drift from what the real clustering run actually does.
"""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.clustering.neighbors_leiden import run_neighbors_leiden_routed

#: obs column the diagnostic writes its own Leiden runs into; never left on the object.
_DIAG_KEY = "_resolution_diagnostic_leiden"


def _cluster_membership(labels: pd.Series) -> dict[str, set]:
    """cluster_id -> set of cell names, from a labels Series indexed by cell name."""
    groups: dict[str, set] = {}
    for cell, label in labels.items():
        groups.setdefault(str(label), set()).add(cell)
    return groups


def _best_match_jaccard(reference_cells: set, subsample_clusters: dict[str, set]) -> float | None:
    """Jaccard between ``reference_cells`` and its best-matching subsample cluster.

    Returns:
        The best Jaccard score, or None when ``reference_cells`` is empty -- every
        member of that reference cluster was resampled out, so this bootstrap says
        nothing about it (not zero, which would claim "destroyed" rather than
        "not evaluable this draw").
    """
    if not reference_cells:
        return None
    best = 0.0
    for candidate in subsample_clusters.values():
        union = reference_cells | candidate
        if not union:
            continue
        jaccard = len(reference_cells & candidate) / len(union)
        if jaccard > best:
            best = jaccard
    return best


def _run_one_clustering(
    target: ad.AnnData,
    *,
    context: object,
    n_neighbors: int,
    use_rep: str,
    resolution: float,
    random_state: int,
) -> pd.Series:
    """Neighbors+Leiden on ``target`` (mutated), returning labels indexed by cell name."""
    run_neighbors_leiden_routed(
        target,
        context=context,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        resolution=resolution,
        random_state=random_state,
        key_added=_DIAG_KEY,
    )
    return target.obs[_DIAG_KEY].astype(str)


def _bootstrap_round(
    adata: ad.AnnData,
    *,
    context: object,
    n_neighbors: int,
    use_rep: str,
    resolution: float,
    random_state: int,
    reference_clusters: dict[str, set],
    n_draws: int,
    subsample_fraction: float,
    seed_offset: int,
) -> list[dict]:
    """Run ``n_draws`` independent subsample-and-recluster bootstraps.

    Returns:
        Long-form rows: one per (reference cluster, bootstrap) pair that was evaluable.
    """
    rows: list[dict] = []
    if n_draws <= 0:
        return rows

    n = adata.n_obs
    subsample_size = max(2, int(round(n * subsample_fraction)))
    for b in range(n_draws):
        rng = np.random.default_rng(random_state + seed_offset + b)
        idx = rng.choice(n, size=min(subsample_size, n), replace=False)
        sub = adata[idx].copy()
        sub_labels = _run_one_clustering(
            sub,
            context=context,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
            resolution=resolution,
            random_state=random_state,
        )
        subsample_set = set(sub.obs_names)
        subsample_clusters = _cluster_membership(sub_labels)
        for ref_cluster, ref_cells in reference_clusters.items():
            score = _best_match_jaccard(ref_cells & subsample_set, subsample_clusters)
            if score is not None:
                rows.append(
                    {
                        "resolution": resolution,
                        "cluster": ref_cluster,
                        "bootstrap": seed_offset + b,
                        "jaccard": score,
                    }
                )
    return rows


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]; an all-equal series normalizes to all zeros."""
    spread = series.max() - series.min()
    if spread <= 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / spread


def _round2_allocation(
    long_df: pd.DataFrame,
    resolutions: Sequence[float],
    remaining_budget: int,
) -> dict[float, int]:
    """Extra bootstraps per resolution: weighted toward uncertain or transitioning points.

    Priority = normalized IQR (round-1 estimate uncertainty) + normalized transition
    magnitude (the larger of the two neighbor-to-neighbor deltas in median stability).
    Falls back to an even split when round 1 produced no signal to prioritize by (e.g.
    every resolution scored identically, or there is only one resolution).
    """
    if remaining_budget <= 0 or long_df.empty:
        return dict.fromkeys(resolutions, 0)

    grouped = long_df.groupby("resolution")["jaccard"]
    summary = grouped.agg(
        median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
    )
    summary = summary.reindex(resolutions).fillna(0.0)

    iqr = summary["q75"] - summary["q25"]
    medians = summary["median"]
    transition = pd.Series(0.0, index=list(resolutions))
    for i, resolution in enumerate(resolutions):
        deltas = []
        if i > 0:
            deltas.append(abs(medians.iloc[i] - medians.iloc[i - 1]))
        if i < len(resolutions) - 1:
            deltas.append(abs(medians.iloc[i] - medians.iloc[i + 1]))
        transition.loc[resolution] = max(deltas) if deltas else 0.0

    priority = _normalize(iqr) + _normalize(transition)
    total = priority.sum()

    if total <= 0:
        # No signal to prioritize by (every resolution scored identically in round 1):
        # split evenly, dropping any remainder rather than over-allocating past budget.
        even = remaining_budget // len(resolutions)
        return dict.fromkeys(resolutions, even)

    raw = (priority / total * remaining_budget).round().astype(int)
    return {r: int(raw.loc[r]) for r in resolutions}


def compute_resolution_stability(
    adata: ad.AnnData,
    *,
    use_rep: str,
    n_neighbors: int,
    resolutions: Sequence[float],
    n_bootstraps: int,
    subsample_fraction: float,
    random_state: int,
    context: object = None,
    adaptive: bool = True,
    initial_fraction: float = 0.3,
) -> tuple[pd.DataFrame, dict[float, int]]:
    """Bootstrap cluster-stability across a resolution sweep.

    Args:
        adata: The fit-eligible cohort to sweep over (already restricted to core cells
            by the caller -- this function clusters every cell it is given).
        use_rep, n_neighbors: Mirror the real clustering run's own settings, so the
            diagnostic describes what that run would actually do.
        resolutions: Candidate resolutions, swept in order (order matters: it defines
            "neighboring" resolutions for the adaptive transition score).
        n_bootstraps: Target bootstrap count per resolution (adaptive mode spends this
            unevenly; non-adaptive mode spends exactly this many per resolution).
        subsample_fraction: Fraction of cells kept per bootstrap draw.
        random_state: Base seed; combined with a per-bootstrap offset for reproducible,
            distinct draws.
        context: Pipeline context, forwarded to GPU routing.
        adaptive: Whether to use the two-round allocation (see module docstring) or
            spend n_bootstraps evenly at every resolution.
        initial_fraction: Fraction of n_bootstraps spent in round 1 when adaptive.

    Returns:
        ``(long_df, n_clusters_by_resolution)`` -- long_df has one row per evaluable
        (resolution, cluster, bootstrap) triple with its Jaccard score; the dict gives
        the reference cluster count at each resolution.
    """
    resolutions = list(resolutions)
    long_rows: list[dict] = []
    n_clusters_by_resolution: dict[float, int] = {}
    reference_by_resolution: dict[float, dict[str, set]] = {}

    initial_n = max(1, int(round(n_bootstraps * initial_fraction))) if adaptive else n_bootstraps

    for resolution in resolutions:
        reference = adata.copy()
        reference_labels = _run_one_clustering(
            reference,
            context=context,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
            resolution=resolution,
            random_state=random_state,
        )
        reference_clusters = _cluster_membership(reference_labels)
        reference_by_resolution[resolution] = reference_clusters
        n_clusters_by_resolution[resolution] = len(reference_clusters)

        long_rows.extend(
            _bootstrap_round(
                adata,
                context=context,
                n_neighbors=n_neighbors,
                use_rep=use_rep,
                resolution=resolution,
                random_state=random_state,
                reference_clusters=reference_clusters,
                n_draws=min(initial_n, n_bootstraps),
                subsample_fraction=subsample_fraction,
                seed_offset=0,
            )
        )

    if adaptive and n_bootstraps > initial_n:
        remaining_budget = (n_bootstraps - initial_n) * len(resolutions)
        allocation = _round2_allocation(pd.DataFrame(long_rows), resolutions, remaining_budget)
        for resolution in resolutions:
            long_rows.extend(
                _bootstrap_round(
                    adata,
                    context=context,
                    n_neighbors=n_neighbors,
                    use_rep=use_rep,
                    resolution=resolution,
                    random_state=random_state,
                    reference_clusters=reference_by_resolution[resolution],
                    n_draws=allocation.get(resolution, 0),
                    subsample_fraction=subsample_fraction,
                    seed_offset=initial_n,
                )
            )

    long_df = pd.DataFrame(long_rows, columns=["resolution", "cluster", "bootstrap", "jaccard"])
    return long_df, n_clusters_by_resolution


def summarize_resolution_stability(
    long_df: pd.DataFrame, n_clusters_by_resolution: dict[float, int]
) -> pd.DataFrame:
    """Per-resolution summary: cluster count, median stability, and its spread.

    Args:
        long_df: As returned by :func:`compute_resolution_stability`.
        n_clusters_by_resolution: As returned by :func:`compute_resolution_stability`.

    Returns:
        One row per resolution (sorted), columns: resolution, n_clusters,
        median_jaccard, q25, q75, n_bootstrap_observations.
    """
    resolutions = sorted(n_clusters_by_resolution)
    if long_df.empty:
        return pd.DataFrame(
            {
                "resolution": resolutions,
                "n_clusters": [n_clusters_by_resolution[r] for r in resolutions],
                "median_jaccard": [float("nan")] * len(resolutions),
                "q25": [float("nan")] * len(resolutions),
                "q75": [float("nan")] * len(resolutions),
                "n_bootstrap_observations": [0] * len(resolutions),
            }
        )

    grouped = long_df.groupby("resolution")["jaccard"]
    summary = grouped.agg(
        median_jaccard="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        n_bootstrap_observations="count",
    ).reindex(resolutions)
    summary["n_clusters"] = [n_clusters_by_resolution[r] for r in resolutions]
    summary["n_bootstrap_observations"] = summary["n_bootstrap_observations"].fillna(0).astype(int)
    summary = summary.reset_index()
    return summary[
        ["resolution", "n_clusters", "median_jaccard", "q25", "q75", "n_bootstrap_observations"]
    ]


__all__ = ["compute_resolution_stability", "summarize_resolution_stability"]
