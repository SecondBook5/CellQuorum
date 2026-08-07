"""Donor x condition pseudobulk aggregation for differential expression.

Sums single-cell counts within each (donor, condition) group into pseudo-sample
profiles, using a sparse indicator matrix so the aggregation stays fast on large
objects. Cannibalized and generalized from paired_sc.stats.paired.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class PseudobulkResult:
    """Pseudobulk counts and aligned sample metadata.

    Args:
        counts: Pseudo-samples x genes integer count matrix.
        sample_meta: Per-pseudo-sample metadata aligned to counts.index.
    """

    # Pseudo-samples x genes integer counts.
    counts: pd.DataFrame

    # Per-pseudo-sample metadata (donor, condition, optional covariates).
    sample_meta: pd.DataFrame


def aggregate_pseudobulk(
    adata: ad.AnnData,
    *,
    layer: str,
    donor_col: str,
    condition_col: str,
    extra_obs: list[str] | None = None,
) -> PseudobulkResult:
    """
    Aggregate single cells to donor x condition pseudobulk counts.

    Args:
        adata: Input AnnData with raw counts in ``layer``.
        layer: Layer name holding raw counts.
        donor_col: obs column identifying donors.
        condition_col: obs column holding the condition label.
        extra_obs: Optional obs columns (e.g. covariates) carried into sample_meta;
            each must be constant within a (donor, condition) group.

    Returns:
        PseudobulkResult with integer counts and aligned sample metadata.

    Raises:
        KeyError: If the layer or a required obs column is absent.
    """

    # Confirm the counts layer exists.
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found; available: {list(adata.layers)}.")

    # Confirm required obs columns exist.
    for column in (donor_col, condition_col, *(extra_obs or [])):
        if column not in adata.obs.columns:
            raise KeyError(f"obs column '{column}' not found for pseudobulk aggregation.")

    # Build the pseudo-sample key per cell.
    donor = adata.obs[donor_col].astype(str)
    condition = adata.obs[condition_col].astype(str)
    group_key = donor.to_numpy() + "__" + condition.to_numpy()

    # Stable ordered list of pseudo-samples.
    groups = pd.Index(pd.unique(group_key))

    # Build a groups x cells sparse indicator matrix, then multiply by counts.
    col_index = {g: i for i, g in enumerate(groups)}
    rows = np.fromiter((col_index[g] for g in group_key), dtype=np.int64, count=len(group_key))
    cols = np.arange(len(group_key), dtype=np.int64)
    data = np.ones(len(group_key), dtype=np.float64)
    indicator = sp.csr_matrix((data, (rows, cols)), shape=(len(groups), adata.n_obs))

    # Counts as a sparse matrix; sum within each group.
    counts_matrix = adata.layers[layer]
    counts_matrix = (
        sp.csr_matrix(counts_matrix) if not sp.issparse(counts_matrix) else counts_matrix
    )
    summed = indicator @ counts_matrix

    # Assemble the pseudobulk counts DataFrame (rounded to integers).
    counts_df = pd.DataFrame(
        np.rint(np.asarray(summed.todense())).astype(np.int64),
        index=groups,
        columns=list(adata.var_names),
    )

    # Assemble aligned per-pseudo-sample metadata (first value per group).
    meta_source = adata.obs.assign(_group=group_key)
    meta_cols = [donor_col, condition_col, *(extra_obs or [])]
    sample_meta = meta_source.groupby("_group")[meta_cols].first().reindex(groups)
    sample_meta.index.name = None

    return PseudobulkResult(counts=counts_df, sample_meta=sample_meta)


__all__ = ["PseudobulkResult", "aggregate_pseudobulk"]
