# Pipeline step (order=105): query_projection — place borderline cells against core.
"""Frozen-reference query projection and its pipeline stage.

Built once, as an engine primitive, because the same neighbourhood question is asked in
QC rescue, annotation, reference mapping and diagnostics, and the design forbids
reimplementing it four times (``docs/design/qc-graded-adjudication.md`` §5). The primitive
is deliberately biology-free and I/O-free: it takes coordinates and labels and returns
per-query-cell metrics. The stage decides where the coordinates come from and where the
results are written.

``marker_vote`` cannot answer this — it assigns labels by ``clusters.map(assignments)``,
so a cell that never entered clustering has no path through it — and nothing resembling
``sc.tl.ingest`` exists in the repo. Hence a primitive.

The frozen-reference contract is satisfied *upstream*, not here: the integration stage
fits its latent model with ``fit_scope=CORE`` and then encodes every cell through the
trained model, so ``obsm['X_scvi']`` already holds borderline cells projected onto a
manifold that only core cells shaped. This function only reads that space; it never
retrains anything, which is what keeps the core manifold immutable
(``docs/design/qc-graded-adjudication.md`` §5, "Reference immutability").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.context_access import resolve_stage_config
from cellquorum.stages.qc.config import QueryProjectionConfig


@dataclass(frozen=True)
class QueryProjection:
    """Per-query-cell projection metrics, each array aligned to the query row order.

    Attributes:
        top_label: The label most represented among a query cell's reference neighbours.
        top_label_probability: Fraction of neighbours carrying ``top_label`` — the
            neighbourhood purity for the winning label.
        second_label_probability: Fraction carrying the runner-up label.
        margin: ``top_label_probability - second_label_probability``. A decisive
            neighbourhood has a wide margin; a cell on a boundary between two populations
            has a narrow one.
        neighbor_label_entropy: Shannon entropy of the neighbour label distribution.
            Low means the neighbours agree; high means the cell sits among a mixture.
        nearest_reference_distance: Distance to the single closest reference cell.
        mean_neighbor_distance: Mean distance to the ``k`` nearest reference cells.
        ood_score: Out-of-distribution score in [0, 1]: this cell's mean neighbour
            distance as a quantile of the reference's own mean-neighbour-distance
            distribution. Near 1 means the cell sits further from the reference than
            almost any reference cell sits from its own neighbours — it fits nowhere.
        effective_neighbor_count: Number of equally weighted reference neighbors used.
    """

    top_label: np.ndarray
    top_label_probability: np.ndarray
    second_label_probability: np.ndarray
    margin: np.ndarray
    neighbor_label_entropy: np.ndarray
    nearest_reference_distance: np.ndarray
    mean_neighbor_distance: np.ndarray
    ood_score: np.ndarray
    effective_neighbor_count: np.ndarray


def project_query_cells(
    reference_coords: np.ndarray,
    query_coords: np.ndarray,
    reference_labels: np.ndarray,
    *,
    k: int = 15,
) -> QueryProjection:
    """Project query cells against a frozen reference in a shared coordinate space.

    Args:
        reference_coords: ``(n_ref, d)`` coordinates of the reference (e.g. QC-core)
            cells in the frozen representation.
        query_coords: ``(n_query, d)`` coordinates of the query (e.g. borderline) cells
            in the SAME representation. Must share ``d`` with ``reference_coords``.
        reference_labels: ``(n_ref,)`` labels for the reference cells, in row order.
        k: Neighbours per query cell. Clamped to ``n_ref - 1`` so the reference's
            leave-one-out distances use the same number of neighbours as each query.

    Returns:
        A :class:`QueryProjection` whose arrays are aligned to ``query_coords`` rows.

    Raises:
        ValueError: If the reference is empty, the dimensions disagree, or the label
            count does not match the reference row count — misuse that would otherwise
            surface as a confusing downstream error.
    """
    reference_coords = np.asarray(reference_coords, dtype=float)
    query_coords = np.asarray(query_coords, dtype=float)
    reference_labels = np.asarray(reference_labels)

    if reference_coords.ndim != 2 or query_coords.ndim != 2:
        raise ValueError("reference_coords and query_coords must both be 2-D.")
    if reference_coords.shape[0] == 0:
        raise ValueError("reference_coords is empty; nothing to project against.")
    if reference_coords.shape[0] < 2:
        raise ValueError("At least two reference cells are required to calibrate OOD distances.")
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or k < 1:
        raise ValueError("k must be a positive integer.")
    if reference_coords.shape[1] != query_coords.shape[1]:
        raise ValueError(
            f"dimension mismatch: reference has {reference_coords.shape[1]} dims, "
            f"query has {query_coords.shape[1]}."
        )
    if reference_labels.ndim != 1 or len(reference_labels) != reference_coords.shape[0]:
        raise ValueError(
            "reference_labels must be one-dimensional with one label per reference cell."
        )
    if pd.isna(reference_labels).any():
        raise ValueError("reference_labels contains missing labels.")

    n_query = query_coords.shape[0]
    k_eff = int(min(k, reference_coords.shape[0] - 1))

    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(reference_coords)

    ref_dist, _ = nn.kneighbors()
    ref_mean_neighbor = ref_dist.mean(axis=1)
    ref_scale_sorted = np.sort(ref_mean_neighbor)

    q_dist, q_idx = nn.kneighbors(query_coords)
    neighbor_labels = reference_labels[q_idx]

    top_label = np.empty(n_query, dtype=reference_labels.dtype)
    top_prob = np.zeros(n_query)
    second_prob = np.zeros(n_query)
    label_entropy = np.zeros(n_query)

    for i in range(n_query):
        values, counts = np.unique(neighbor_labels[i], return_counts=True)
        probs = counts / counts.sum()
        order = np.argsort(counts)[::-1]
        top_label[i] = values[order[0]]
        top_prob[i] = probs[order[0]]
        second_prob[i] = probs[order[1]] if len(order) > 1 else 0.0
        label_entropy[i] = entropy(probs)

    nearest = q_dist[:, 0]
    mean_neighbor = q_dist.mean(axis=1)

    ood = np.searchsorted(ref_scale_sorted, mean_neighbor, side="right") / len(ref_scale_sorted)

    return QueryProjection(
        top_label=top_label,
        top_label_probability=top_prob,
        second_label_probability=second_prob,
        margin=top_prob - second_prob,
        neighbor_label_entropy=label_entropy,
        nearest_reference_distance=nearest,
        mean_neighbor_distance=mean_neighbor,
        ood_score=ood,
        effective_neighbor_count=np.full(n_query, k_eff, dtype=float),
    )


def neighborhood_label_entropy(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell label mixing of each cell's own neighbourhood in a shared manifold.

    Args:
        coords: ``(n, d)`` cell coordinates in the manifold.
        labels: ``(n,)`` per-cell labels (e.g. coarse cell type).
        k: Neighbours per cell, including the cell itself; clamped to ``n``.

    Returns:
        ``(entropy, effective_labels)``, both ``(n,)`` aligned to ``coords``.
        ``effective_labels = exp(entropy)`` is the number of labels the neighbourhood
        effectively spans — 1 for a pure neighbourhood, higher as it mixes — and reads
        more naturally as a threshold than nats of entropy do.

    Raises:
        ValueError: If shapes disagree or the input is empty.
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels)
    if coords.ndim != 2 or coords.shape[0] == 0:
        raise ValueError("coords must be a non-empty 2-D array.")
    if labels.shape[0] != coords.shape[0]:
        raise ValueError(f"labels has {labels.shape[0]} entries for {coords.shape[0]} cells.")

    n = coords.shape[0]
    k_eff = int(min(k, n))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(coords)
    _, idx = nn.kneighbors(coords)

    codes, _ = pd.factorize(labels)
    n_labels = int(codes.max()) + 1 if codes.size and codes.max() >= 0 else 1
    neighbour_codes = codes[idx]

    ent = np.zeros(n, dtype=float)
    for i in range(n):
        counts = np.bincount(neighbour_codes[i], minlength=n_labels)
        ent[i] = entropy(counts / counts.sum())
    return ent, np.exp(ent)


_DEFAULT_REP_CANDIDATES = ("X_scvi", "X_pca")


_DEFAULT_LABEL_CANDIDATES = ("cell_type", "qc_provisional_lineage", "leiden")

_STATE_COLUMN = "qc_state_initial"


@register_stage(
    name="query_projection",
    order=105,
    config_flag="query_projection",
    config_field="query_projection",
)
class QueryProjectionStage:
    """Project borderline cells onto the frozen core manifold and record the evidence."""

    def _resolve_rep(self, adata: object, config: dict) -> str | None:
        requested = config.get("use_rep")
        if requested and requested not in adata.obsm:
            raise ValueError(f"Requested query-projection representation '{requested}' is missing.")
        candidates = (requested,) if requested else _DEFAULT_REP_CANDIDATES
        for key in candidates:
            if key and key in adata.obsm:
                return key
        return None

    def _resolve_label_column(self, adata: object, config: dict) -> str | None:
        requested = config.get("label_column")
        if requested and requested not in adata.obs.columns:
            raise ValueError(f"Requested query-projection label column '{requested}' is missing.")
        candidates = (requested,) if requested else _DEFAULT_LABEL_CANDIDATES
        for col in candidates:
            if col and col in adata.obs.columns:
                return col
        return None

    def run(self, context: object) -> StageResult:
        """Execute the query-projection stage."""
        adata = context.require_adata()
        config = QueryProjectionConfig.model_validate(
            resolve_stage_config(context, "query_projection")
        ).model_dump()

        if _STATE_COLUMN not in adata.obs.columns:
            return StageResult.skipped(
                adata=adata,
                reason=f"no {_STATE_COLUMN} column; QC graded adjudication did not run",
                warnings=[
                    f"query_projection needs obs['{_STATE_COLUMN}']; is the graded QC "
                    "stage enabled?"
                ],
            )

        state = adata.obs[_STATE_COLUMN].astype(str)
        core_mask = (state == "core").to_numpy()
        query_mask = (state == "borderline").to_numpy()

        if not query_mask.any():
            return StageResult.skipped(
                adata=adata,
                reason="no borderline cells to project",
                metrics={"n_borderline": 0, "n_core": int(core_mask.sum())},
            )
        if core_mask.sum() < 2:
            return StageResult.skipped(
                adata=adata,
                reason="fewer than 2 core cells to project against",
                warnings=["query_projection: the core reference is too small to project onto."],
                metrics={"n_borderline": int(query_mask.sum()), "n_core": int(core_mask.sum())},
            )

        rep = self._resolve_rep(adata, config)
        if rep is None:
            return StageResult.skipped(
                adata=adata,
                reason="no frozen representation in obsm (tried X_scvi/X_pca)",
                warnings=[
                    "query_projection needs a core-fit embedding; run integration or "
                    "dimensionality first."
                ],
            )
        label_column = self._resolve_label_column(adata, config)
        if label_column is None:
            return StageResult.skipped(
                adata=adata,
                reason="no reference label column (tried cell_type/qc_provisional_lineage/leiden)",
                warnings=["query_projection needs core-cell labels; run annotation first."],
            )

        k = config["k"]
        coords = np.asarray(adata.obsm[rep])
        reference_labels = adata.obs.loc[core_mask, label_column]
        if reference_labels.isna().any():
            raise ValueError(
                f"Reference label column '{label_column}' contains missing core labels."
            )
        labels = reference_labels.astype(str).to_numpy()

        projection = project_query_cells(
            reference_coords=coords[core_mask],
            query_coords=coords[query_mask],
            reference_labels=labels,
            k=k,
        )

        n = adata.n_obs
        qi = np.flatnonzero(query_mask)

        def _fill_num(values: np.ndarray) -> np.ndarray:
            out = np.full(n, np.nan, dtype=float)
            out[qi] = values
            return out

        top_label_full = np.array([""] * n, dtype=object)
        top_label_full[qi] = projection.top_label.astype(str)

        new_cols = {
            "query_top_label": top_label_full,
            "query_top_label_probability": _fill_num(projection.top_label_probability),
            "query_second_label_probability": _fill_num(projection.second_label_probability),
            "query_label_margin": _fill_num(projection.margin),
            "query_neighbor_label_entropy": _fill_num(projection.neighbor_label_entropy),
            "query_nearest_reference_distance": _fill_num(projection.nearest_reference_distance),
            "query_mean_neighbor_distance": _fill_num(projection.mean_neighbor_distance),
            "query_ood_score": _fill_num(projection.ood_score),
            "query_effective_neighbor_count": _fill_num(projection.effective_neighbor_count),
            "query_effective_label_count": _fill_num(np.exp(projection.neighbor_label_entropy)),
        }
        adata.obs = pd.concat(
            [
                adata.obs.drop(columns=list(new_cols), errors="ignore"),
                pd.DataFrame(new_cols, index=adata.obs_names),
            ],
            axis=1,
        )

        adata.uns.setdefault("cellquorum", {})["query_projection"] = {
            "representation": rep,
            "label_column": label_column,
            "k": k,
            "n_borderline": int(query_mask.sum()),
            "n_core_reference": int(core_mask.sum()),
        }

        median_ood = float(np.median(projection.ood_score))
        median_support = float(np.median(projection.top_label_probability))
        return StageResult(
            adata=adata,
            notes=[
                f"query_projection projected {int(query_mask.sum()):,} borderline cells onto "
                f"{int(core_mask.sum()):,} core cells in '{rep}' (labels from "
                f"'{label_column}', k={k}).",
                f"median neighbourhood support={median_support:.2f}, median OOD={median_ood:.2f}.",
            ],
            metrics={
                "representation": rep,
                "label_column": label_column,
                "k": k,
                "n_borderline": int(query_mask.sum()),
                "n_core": int(core_mask.sum()),
                "median_top_label_probability": median_support,
                "median_ood_score": median_ood,
            },
        )


__all__ = [
    "QueryProjection",
    "project_query_cells",
    "neighborhood_label_entropy",
    "QueryProjectionStage",
]
