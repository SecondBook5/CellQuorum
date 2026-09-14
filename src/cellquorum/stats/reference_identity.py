"""Reference-mapping identity-confidence diagnostics.

Generalizes a real forensic diagnostic from an external analysis into
reusable CellQuorum code: reference_mapping's per-cell class probabilities
(``refprob_*`` obs columns / the ``{key_added}_probabilities`` obsm matrix)
carry more information than the single label they get collapsed to. A cell
can win a label by a hair over a crowded field (low top probability -- an
"ambiguous identity" cell, independent of whether the three-way consensus
check in ``annotation_consensus`` happens to agree with the assignment), or
the classifier's mass can concentrate on a DIFFERENT coarse group than the
one the cell was actually assigned to (a "reassembly candidate").

Both checks are read-only characterizations of an already-finished object,
not a stage -- there is nothing here to compute upstream of, and nothing
that reassigns a label. Consumed manually or from a diagnostic script, the
same way ``cellquorum.stats.ambient_specificity`` is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def top_reference_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Return each cell's top and second-highest reference-class probability.

    Args:
        probabilities: Cells (rows) x reference classes (columns) probability
            matrix, e.g. built from every ``refprob_*`` obs column.

    Returns:
        DataFrame indexed like ``probabilities``, with ``top_label``,
        ``top_prob``, ``second_label``, ``second_prob`` columns. A high
        second-highest probability alongside a middling top one is the
        classic doublet-like signature: two identities, neither dominant.
    """
    values = probabilities.to_numpy(dtype=float)
    columns = np.asarray(probabilities.columns)

    # argsort ascending; the last two columns are the top two per row.
    order = np.argsort(values, axis=1, kind="mergesort")
    top_idx = order[:, -1]
    second_idx = order[:, -2] if values.shape[1] >= 2 else order[:, -1]

    rows = np.arange(values.shape[0])
    return pd.DataFrame(
        {
            "top_label": columns[top_idx],
            "top_prob": values[rows, top_idx],
            "second_label": columns[second_idx],
            "second_prob": values[rows, second_idx],
        },
        index=probabilities.index,
    )


def ambiguous_identity_mask(top_prob: pd.Series, *, threshold: float = 0.5) -> pd.Series:
    """Flag cells whose best reference-class probability falls below a floor.

    Args:
        top_prob: Per-cell top reference-class probability (e.g.
            ``top_reference_probabilities(...)["top_prob"]``).
        threshold: Probability floor; below it, no class dominates.

    Returns:
        Boolean Series, same index as ``top_prob``.
    """
    return top_prob < threshold


def blob_composition(obs: pd.DataFrame, blob_mask: pd.Series, cell_type_col: str) -> pd.Series:
    """Return the assigned-cell-type composition of the flagged (blob) cells.

    Args:
        obs: Obs DataFrame carrying ``cell_type_col``.
        blob_mask: Boolean Series (same index as ``obs``) marking blob cells.
        cell_type_col: Column identifying each cell's assigned type.

    Returns:
        Series of fractions (summing to 1.0) of assigned cell types among
        blob cells, only the types actually present -- descending.
    """
    counts = obs.loc[blob_mask.to_numpy(), cell_type_col].value_counts()
    return (counts / counts.sum()).astype(float)


def blob_qc_comparison(
    obs: pd.DataFrame, blob_mask: pd.Series, qc_columns: Sequence[str]
) -> pd.DataFrame:
    """Compare blob cells against the rest on a set of QC/uncertainty metrics.

    Args:
        obs: Obs DataFrame carrying every column in ``qc_columns``.
        blob_mask: Boolean Series (same index as ``obs``) marking blob cells.
        qc_columns: Numeric obs columns to compare (e.g. doublet score,
            n_genes, pct_mito, second-highest reference probability).

    Returns:
        DataFrame indexed by column name, with ``blob_mean``/``rest_mean``.
        A blob that runs hot on doublet score and cold on n_genes reads as
        real low-quality/multiplet cells; a blob that matches the rest on
        every QC axis reads as a genuinely intermediate cell state instead.
    """
    mask = blob_mask.to_numpy()
    blob_means = obs.loc[mask, qc_columns].mean()
    rest_means = obs.loc[~mask, qc_columns].mean()
    return pd.DataFrame({"blob_mean": blob_means, "rest_mean": rest_means})


def coarse_reassembly_candidates(
    probabilities: pd.DataFrame,
    fine_labels: Sequence[str],
    fine_to_coarse: Mapping[str, str],
    assigned_coarse: pd.Series,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Find cells the reference classifier assigns to a DIFFERENT coarse group.

    Generalizes a real, previously hardcoded-to-one-cell-type check: for
    EVERY coarse group the hierarchy defines (not just one picked by hand),
    sum each cell's fine-label probability mass that maps to that group, and
    flag cells whose mass exceeds ``threshold`` for a group they were not
    actually assigned to.

    Args:
        probabilities: Cells (rows) x fine reference classes (columns).
        fine_labels: Fine label columns of ``probabilities`` to consider
            (lets a caller restrict to columns that exist on both sides).
        fine_to_coarse: Fine label -> coarse group, e.g. from the same
            hierarchy ``reference_mapping`` itself derived.
        assigned_coarse: Per-cell assigned coarse group, same index as
            ``probabilities``.
        threshold: Probability-mass floor a foreign group must clear.

    Returns:
        DataFrame with one row per (cell, candidate coarse group) that
        qualifies: ``cell``, ``candidate_group``, ``candidate_prob``,
        ``assigned_group``. Empty (with these columns) when nothing
        qualifies.
    """
    columns = ["cell", "candidate_group", "candidate_prob", "assigned_group"]
    coarse_groups: dict[str, list[str]] = {}
    for label in fine_labels:
        group = fine_to_coarse.get(label)
        if group is None:
            continue
        coarse_groups.setdefault(group, []).append(label)

    rows: list[dict[str, object]] = []
    for group, members in coarse_groups.items():
        group_prob = probabilities[members].sum(axis=1)
        qualifies = group_prob > threshold
        not_assigned = assigned_coarse.reindex(probabilities.index) != group
        candidates = qualifies & not_assigned
        for cell in probabilities.index[candidates.to_numpy()]:
            rows.append(
                {
                    "cell": cell,
                    "candidate_group": group,
                    "candidate_prob": float(group_prob.loc[cell]),
                    "assigned_group": assigned_coarse.loc[cell],
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "ambiguous_identity_mask",
    "blob_composition",
    "blob_qc_comparison",
    "coarse_reassembly_candidates",
    "top_reference_probabilities",
]
