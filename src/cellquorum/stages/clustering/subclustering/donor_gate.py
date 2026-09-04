"""Donor-reproducibility gatekeeper for subclustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier

if TYPE_CHECKING:
    from anndata import AnnData


def donor_reproducibility(
    adata: AnnData,
    cluster_key: str,
    group_key: str,
    *,
    min_groups: int = 3,
    min_cells_per_group: int = 0,
    max_group_frac: float | None = 0.8,
    do_lodo: bool = True,
    do_classifier: bool = True,
    embedding_key: str = "X_pca",
    random_state: int = 0,
) -> dict:
    """
    Compute per-cluster donor-reproducibility metrics.

    For each cluster in obs[cluster_key], compute:
    - n_groups: number of distinct donors/groups contributing cells.
    - max_group_frac: fraction of cells from the single most-represented donor
      (one-donor-dominated detector).
    - n_cells: number of cells in the cluster.
    - lodo_stability: leave-one-donor-out stability (if do_lodo=True).
      For each donor held out, train a kNN classifier on other donors'
      cells predicting cluster labels, then predict held-out donor's cells.
      LODO stability = mean agreement (held-out predictions match assigned
      cluster) across held-out donors.
    - classifier_sep: donor-blocked one-vs-rest separability (if do_classifier).
      Train a RandomForestClassifier on cells from a subset of donors,
      test on held-out donors. Use GroupKFold or LeaveOneGroupOut to ensure
      train/test donors are DISJOINT (never split by cell → would leak donor).
    - qc_pass: bool, PASS if n_groups >= min_groups AND (max_group_frac is None
      OR max_group_frac <= threshold) AND (lodo_stability is None OR above floor).
    - qc_reason: str, explanation for FAIL verdict.

    Args:
        adata: AnnData with obs[cluster_key], obs[group_key], obsm[embedding_key].
        cluster_key: obs column with cluster labels.
        group_key: obs column for donor/group identity.
        min_groups: minimum number of groups required for PASS.
        min_cells_per_group: minimum cells a group must contribute to count as a
            supporting group. Groups below this are excluded from the effective
            group count used for the min_groups PASS check. 0 disables the floor.
        max_group_frac: max fraction of cells from one donor (default 0.8).
            None = skip one-donor-dominated check.
        do_lodo: whether to compute LODO stability.
        do_classifier: whether to compute donor-blocked classifier separability.
        embedding_key: obsm key for embedding (features for classifier).
        random_state: random seed for reproducibility.

    Returns:
        dict with:
        - clusters: {cluster_id: {n_groups, max_group_frac, n_cells,
          lodo_stability, classifier_sep, qc_pass, qc_reason}}
        - summary: {n_pass, n_fail}
    """
    # Validate inputs.
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"cluster_key '{cluster_key}' not found in adata.obs")
    if group_key not in adata.obs.columns:
        raise ValueError(f"group_key '{group_key}' not found in adata.obs")
    if embedding_key not in adata.obsm.keys():
        raise ValueError(f"embedding_key '{embedding_key}' not found in adata.obsm")

    # Extract cluster labels, group labels, embedding.
    clusters = adata.obs[cluster_key].to_numpy()
    groups = adata.obs[group_key].to_numpy()
    embedding = adata.obsm[embedding_key]

    # Unique clusters.
    unique_clusters = np.unique(clusters)

    results = {}

    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cluster_groups = groups[cluster_mask]
        cluster_embedding = embedding[cluster_mask]

        # Basic metrics.
        n_cells = int(cluster_mask.sum())
        unique_groups_in_cluster = np.unique(cluster_groups)
        n_groups = len(unique_groups_in_cluster)

        # Max group fraction (one-donor-dominated detector).
        group_counts = pd.Series(cluster_groups).value_counts()
        max_group_frac_val = float(group_counts.max() / n_cells)

        # Effective group count: only groups meeting the per-group cell floor
        # count as supporting evidence. A cluster propped up by many groups that
        # each contribute a handful of cells is not donor-reproducible.
        if min_cells_per_group > 0:
            n_supporting_groups = int((group_counts >= min_cells_per_group).sum())
        else:
            n_supporting_groups = n_groups

        # LODO stability.
        lodo_stability = None
        if do_lodo and n_groups >= 2:
            lodo_stability = _compute_lodo_stability(
                cluster_embedding,
                cluster_groups,
                embedding,
                groups,
                clusters,
                cluster_id,
                random_state,
            )

        # Classifier separability (donor-blocked).
        classifier_sep = None
        if do_classifier and n_groups >= 2:
            classifier_sep = _compute_classifier_separability(
                embedding,
                groups,
                clusters,
                cluster_id,
                random_state,
            )

        # QC verdict.
        qc_pass, qc_reason = _evaluate_qc(
            n_groups,
            max_group_frac_val,
            lodo_stability,
            min_groups,
            max_group_frac,
            n_supporting_groups=n_supporting_groups,
            min_cells_per_group=min_cells_per_group,
        )

        results[cluster_id] = {
            "n_groups": n_groups,
            "n_supporting_groups": n_supporting_groups,
            "max_group_frac": max_group_frac_val,
            "n_cells": n_cells,
            "lodo_stability": lodo_stability,
            "classifier_sep": classifier_sep,
            "qc_pass": qc_pass,
            "qc_reason": qc_reason,
        }

    # Summary.
    n_pass = sum(1 for r in results.values() if r["qc_pass"])
    n_fail = len(results) - n_pass

    return {"clusters": results, "summary": {"n_pass": n_pass, "n_fail": n_fail}}


def _compute_lodo_stability(
    cluster_embedding: np.ndarray,
    cluster_groups: np.ndarray,
    full_embedding: np.ndarray,
    full_groups: np.ndarray,
    full_clusters: np.ndarray,
    cluster_id: str,
    random_state: int,
) -> float | None:
    """
    Compute leave-one-donor-out (LODO) stability for a cluster.

    For each donor in the cluster, hold out that donor's cells, train a kNN
    classifier on the other donors' cells to predict cluster membership
    (binary: this cluster vs. all others), then predict the held-out donor's
    cells. LODO stability = mean agreement (held-out predictions match the
    cluster label) across held-out donors.

    Args:
        cluster_embedding: embedding for cells in this cluster.
        cluster_groups: group labels for cells in this cluster.
        full_embedding: embedding for all cells in adata.
        full_groups: group labels for all cells.
        full_clusters: cluster labels for all cells.
        cluster_id: the cluster ID being tested.
        random_state: random seed.

    Returns:
        Mean agreement (fraction of held-out cells correctly predicted).
    """
    unique_groups_in_cluster = np.unique(cluster_groups)
    n_groups = len(unique_groups_in_cluster)

    if n_groups < 2:
        return None

    # Binary target: 1 = this cluster, 0 = all others.
    y_binary = (full_clusters == cluster_id).astype(int)

    agreements = []

    for held_out_group in unique_groups_in_cluster:
        # Train mask: all donors EXCEPT held_out_group.
        train_mask = full_groups != held_out_group
        test_mask = full_groups == held_out_group

        X_train = full_embedding[train_mask]
        y_train = y_binary[train_mask]
        X_test = full_embedding[test_mask]
        y_test = y_binary[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Train kNN classifier (k=min(5, n_train)).
        n_neighbors = min(5, len(X_train) - 1)
        if n_neighbors < 1:
            continue

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        # Predict held-out donor's cells.
        y_pred = knn.predict(X_test)

        # Agreement: fraction of held-out cells correctly predicted.
        agreement = (y_pred == y_test).mean()
        agreements.append(agreement)

    if not agreements:
        return None

    return float(np.mean(agreements))


def _compute_classifier_separability(
    embedding: np.ndarray,
    groups: np.ndarray,
    clusters: np.ndarray,
    cluster_id: str,
    random_state: int,
) -> float | None:
    """
    Compute donor-blocked one-vs-rest classifier separability.

    Train a RandomForestClassifier on cells from a subset of donors,
    test on held-out donors. Use GroupKFold to split by donor (NEVER by cell).
    Compute balanced accuracy for one-vs-rest classification (this cluster vs.
    all others).

    Args:
        embedding: embedding for all cells.
        groups: group labels for all cells.
        clusters: cluster labels for all cells.
        cluster_id: the cluster ID being tested.
        random_state: random seed.

    Returns:
        Mean balanced accuracy across donor-blocked folds, or None if
        cannot donor-block (<2 groups).
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        # Cannot donor-block with <2 groups.
        return None

    # Binary target: 1 = this cluster, 0 = all others.
    y_binary = (clusters == cluster_id).astype(int)

    # Use LeaveOneGroupOut (LOGO) if few groups, else GroupKFold.
    if n_groups <= 5:
        splitter = LeaveOneGroupOut()
    else:
        # Use 3-fold GroupKFold (or n_groups if fewer).
        n_splits = min(3, n_groups)
        splitter = GroupKFold(n_splits=n_splits)

    balanced_accs = []

    for train_idx, test_idx in splitter.split(embedding, y_binary, groups):
        X_train, X_test = embedding[train_idx], embedding[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]

        # Skip if test set has no positive or no negative examples.
        if len(np.unique(y_test)) < 2:
            continue

        # Train RandomForestClassifier.
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=random_state,
            n_jobs=1,
        )
        rf.fit(X_train, y_train)

        # Predict test set.
        y_pred = rf.predict(X_test)

        # Balanced accuracy (average of recall for each class).
        from sklearn.metrics import balanced_accuracy_score

        bacc = balanced_accuracy_score(y_test, y_pred)
        balanced_accs.append(bacc)

    if not balanced_accs:
        return None

    return float(np.mean(balanced_accs))


def _evaluate_qc(
    n_groups: int,
    max_group_frac: float,
    lodo_stability: float | None,
    min_groups: int,
    max_group_frac_threshold: float | None,
    *,
    n_supporting_groups: int | None = None,
    min_cells_per_group: int = 0,
) -> tuple[bool, str]:
    """
    Evaluate QC pass/fail for a cluster.

    Args:
        n_groups: number of groups in the cluster.
        max_group_frac: max fraction of cells from one group.
        lodo_stability: LODO stability (or None if not computed).
        min_groups: minimum groups required.
        max_group_frac_threshold: max allowed group fraction (or None to skip).
        n_supporting_groups: number of groups meeting the per-group cell floor.
            Defaults to ``n_groups`` when not provided.
        min_cells_per_group: per-group cell floor (for the FAIL message).

    Returns:
        (qc_pass, qc_reason) tuple.
    """
    reasons = []

    # Effective group count defaults to raw n_groups when no floor was applied.
    effective_groups = n_groups if n_supporting_groups is None else n_supporting_groups

    # Check n_groups against the min, using the per-group-cell-floored count.
    if effective_groups < min_groups:
        if min_cells_per_group > 0 and effective_groups != n_groups:
            reasons.append(
                f"n_groups < min_groups ({effective_groups} < {min_groups}; "
                f"{n_groups} total but only {effective_groups} with "
                f">={min_cells_per_group} cells)"
            )
        else:
            reasons.append(f"n_groups < min_groups ({effective_groups} < {min_groups})")

    # Check one-donor-dominated.
    if max_group_frac_threshold is not None and max_group_frac > max_group_frac_threshold:
        reasons.append(
            f"one-donor-dominated (max_group_frac={max_group_frac:.2f} "
            f"> {max_group_frac_threshold:.2f})"
        )

    # Check LODO stability (optional floor).
    # For now, no floor enforced (LODO is informational).
    # Could add: if lodo_stability is not None and lodo_stability < floor: ...

    qc_pass = len(reasons) == 0
    qc_reason = "; ".join(reasons) if reasons else "PASS"

    return qc_pass, qc_reason


def apply_qc_flags(
    adata: AnnData,
    cluster_key: str,
    gate_result: dict,
    key_added: str,
) -> None:
    """
    Write per-cell QC flags to adata.obs (flag-not-drop).

    For each cell, annotate obs[f"{key_added}_qc_pass"] (bool) and
    obs[f"{key_added}_qc_reason"] (str) based on its cluster's QC verdict.

    This function DOES NOT remove cells (flag-not-drop default). To drop
    cells in failed clusters, filter adata after calling this function:
        adata = adata[adata.obs[f"{key_added}_qc_pass"]].copy()

    Args:
        adata: AnnData with obs[cluster_key].
        cluster_key: obs column with cluster labels.
        gate_result: output of donor_reproducibility().
        key_added: prefix for obs columns (e.g., "donor_qc").

    Side effects:
        Writes obs[f"{key_added}_qc_pass"] and obs[f"{key_added}_qc_reason"].
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"cluster_key '{cluster_key}' not found in adata.obs")

    clusters = adata.obs[cluster_key].to_numpy()

    # Map cluster ID → (qc_pass, qc_reason), keyed by STRING. The ids come out of
    # numpy as int64 here, but h5py cannot name a group with one, so writing this
    # payload to h5ad turns the keys into strings — and a gate_result read back
    # from a checkpoint would then miss on every single cell and mark the whole
    # object "cluster not in gate_result", i.e. failed. Matching on str() costs
    # nothing and makes the lookup indifferent to which side of a write we are on.
    cluster_verdicts = {
        str(cid): (info["qc_pass"], info["qc_reason"])
        for cid, info in gate_result["clusters"].items()
    }

    # Annotate each cell.
    qc_pass = np.zeros(len(clusters), dtype=bool)
    qc_reason = np.empty(len(clusters), dtype=object)

    for i, cluster_id in enumerate(clusters):
        if str(cluster_id) in cluster_verdicts:
            qc_pass[i], qc_reason[i] = cluster_verdicts[str(cluster_id)]
        else:
            # Cluster not in gate_result (should not happen).
            qc_pass[i] = False
            qc_reason[i] = "cluster not in gate_result"

    adata.obs[f"{key_added}_qc_pass"] = qc_pass
    adata.obs[f"{key_added}_qc_reason"] = qc_reason.astype(str)


__all__ = ["donor_reproducibility", "apply_qc_flags"]
