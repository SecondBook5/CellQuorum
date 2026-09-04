"""Label a data-driven partition by its dominant module signature.

The subclustering stage produces clusters; a manuscript needs *named* states. The
naming here is deliberately cluster-level rather than per-cell: a cluster is one
label, chosen by which signature scores highest across its cells. Per-cell argmax
on noisy AUC scores speckles every cluster with a minority label, which then
propagates into the per-subtype statistics as a handful of cells drawn from the
wrong biology. Averaging first is the more stable estimator and the one the
design spec asks for.

Nothing here knows what the signatures mean. A signature is a name plus a list of
program columns, both supplied by config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.stats import signature_argmax_labels

UNASSIGNED = "unassigned"


def signature_scores(
    scores: pd.DataFrame, signatures: dict[str, list[str]]
) -> tuple[pd.DataFrame, list[str]]:
    """Collapse program scores into one column per signature (mean of members).

    Args:
        scores: Cells x programs score frame.
        signatures: Signature name -> member program names. Members absent from
            ``scores`` are dropped, and a signature left with no members is
            skipped entirely.

    Returns:
        ``(frame, notes)`` — cells x signatures, plus one note per signature that
        lost members or was skipped. A signature silently reduced to a single
        surviving gene program is the kind of thing that turns into a
        misinterpreted figure months later, so the loss is always recorded.
    """
    notes: list[str] = []
    columns: dict[str, pd.Series] = {}
    for name, members in signatures.items():
        present = [m for m in members if m in scores.columns]
        missing = [m for m in members if m not in scores.columns]
        if missing:
            notes.append(
                f"signature '{name}': {len(missing)}/{len(members)} program(s) not scored "
                f"({', '.join(sorted(missing)[:5])})"
            )
        if not present:
            notes.append(f"signature '{name}' skipped: none of its programs were scored")
            continue
        columns[name] = scores[present].mean(axis=1)
    return pd.DataFrame(columns, index=scores.index), notes


def label_clusters(
    scores: pd.DataFrame,
    cluster_labels: pd.Series,
    *,
    signatures: dict[str, list[str]] | None = None,
    min_margin: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Assign every cluster one signature label; return per-cell labels too.

    Args:
        scores: Cells x programs score frame.
        cluster_labels: Per-cell cluster assignment, indexed like ``scores``.
        signatures: Signature name -> member programs. ``None`` or empty treats
            each program as its own candidate signature, which is the sensible
            default when a caller has not decided on named states yet.
        min_margin: Minimum standardized best-minus-second gap for a label to be
            assigned; below it the cluster is ``unassigned``.

    Returns:
        ``(per_cell_labels, cluster_table, notes)``. ``cluster_table`` is the
        replot source: cluster, label, top_signature, top_z, second_z, margin.
    """
    notes: list[str] = []
    if signatures:
        sig_frame, sig_notes = signature_scores(scores, signatures)
        notes.extend(sig_notes)
    else:
        sig_frame = scores
    if sig_frame.empty or sig_frame.shape[1] == 0:
        raise ValueError("no signature columns could be built from the score matrix")

    table = signature_argmax_labels(sig_frame, cluster_labels, min_margin=min_margin)

    n_unassigned = int((table["label"] == UNASSIGNED).sum())
    if n_unassigned:
        notes.append(
            f"{n_unassigned}/{len(table)} cluster(s) left '{UNASSIGNED}': "
            f"top-vs-second signature margin below min_margin={min_margin:g}"
        )

    mapping = dict(zip(table["cluster"], table["label"], strict=True))
    per_cell = pd.Series(
        [mapping.get(c, UNASSIGNED) for c in np.asarray(cluster_labels)],
        index=scores.index,
        dtype="object",
    )
    return per_cell, table, notes


__all__ = ["UNASSIGNED", "label_clusters", "signature_scores"]
