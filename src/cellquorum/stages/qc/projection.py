"""Query projection primitive: place query cells against a frozen reference manifold.

Built once, as an engine primitive, because the same neighbourhood question is asked in
QC rescue, annotation, reference mapping and diagnostics, and the design forbids
reimplementing it four times (``docs/design/qc-graded-adjudication.md`` §5). This module
is deliberately biology-free and I/O-free: it takes coordinates and labels and returns
per-query-cell metrics. The stage layer decides where the coordinates come from and where
the results are written.

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
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors


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
        effective_neighbor_count: ``exp(neighbor_label_entropy)`` — the number of labels
            the neighbourhood effectively spans (1 = unanimous, k = maximally mixed).
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
        k: Neighbours per query cell. Clamped to ``n_ref`` when the reference is smaller,
            so a tiny reference degrades gracefully instead of raising.

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
    if reference_coords.shape[1] != query_coords.shape[1]:
        raise ValueError(
            f"dimension mismatch: reference has {reference_coords.shape[1]} dims, "
            f"query has {query_coords.shape[1]}."
        )
    if reference_labels.shape[0] != reference_coords.shape[0]:
        raise ValueError(
            f"reference_labels has {reference_labels.shape[0]} entries for "
            f"{reference_coords.shape[0]} reference cells."
        )

    n_query = query_coords.shape[0]
    k_eff = int(min(k, reference_coords.shape[0]))

    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(reference_coords)

    # Reference-internal neighbour distances calibrate the OOD scale: a query cell is
    # out-of-distribution relative to how far reference cells sit from THEIR neighbours,
    # not against an absolute distance whose meaning changes with the representation.
    ref_dist, _ = nn.kneighbors(reference_coords)
    # Column 0 is the cell itself (distance 0); mean over the real neighbours.
    ref_mean_neighbor = ref_dist[:, 1:].mean(axis=1) if k_eff > 1 else ref_dist[:, 0]
    ref_scale_sorted = np.sort(ref_mean_neighbor)

    q_dist, q_idx = nn.kneighbors(query_coords)
    neighbor_labels = reference_labels[q_idx]  # (n_query, k_eff)

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
    # OOD score: quantile position of each query cell's mean-neighbour distance within
    # the reference's own distribution. searchsorted gives the count of reference cells
    # at least as close; dividing by n maps it to [0, 1].
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
        effective_neighbor_count=np.exp(label_entropy),
    )


__all__ = ["QueryProjection", "project_query_cells"]
