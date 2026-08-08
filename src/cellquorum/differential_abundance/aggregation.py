"""Sample × cell-type count aggregation for differential abundance analysis.

Aggregates single-cell observations into sample (donor × condition) × cell-type
integer count tables, used by abundance testing methods (scCODA, propeller).
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CelltypeCounts:
    """Sample × cell-type counts and aligned sample metadata.

    Attributes:
        counts: Samples (rows) × cell types (columns) integer count matrix.
            Index is the sample identifier (donor_condition).
        sample_meta: Per-sample metadata aligned to counts.index, containing
            donor_col, condition_col, and any extra_obs columns.
    """

    counts: pd.DataFrame
    sample_meta: pd.DataFrame


def aggregate_celltype_counts(
    adata: ad.AnnData,
    *,
    donor_col: str,
    condition_col: str,
    cell_type_col: str,
    extra_obs: tuple[str, ...] = (),
) -> CelltypeCounts:
    """
    Aggregate single cells to sample × cell-type counts.

    Counts cells of each type in each (donor, condition) sample.

    Args:
        adata: Input AnnData object.
        donor_col: obs column identifying donors.
        condition_col: obs column holding the condition label.
        cell_type_col: obs column identifying cell types.
        extra_obs: Optional obs columns (e.g. covariates) carried into sample_meta;
            each must be constant within a (donor, condition) group.

    Returns:
        CelltypeCounts with integer cell counts and aligned sample metadata.

    Raises:
        KeyError: If a required obs column is absent.

    Notes:
        Sample identifier is constructed as f"{donor}_{condition}".
        Both .counts and .sample_meta share the same sample index for alignment.
    """

    # Confirm required obs columns exist.
    for column in (donor_col, condition_col, cell_type_col, *extra_obs):
        if column not in adata.obs.columns:
            raise KeyError(f"obs column '{column}' not found for cell-type count aggregation.")

    # Build the sample key per cell as donor_condition.
    donor = adata.obs[donor_col].astype(str)
    condition = adata.obs[condition_col].astype(str)
    sample_key = donor.to_numpy() + "_" + condition.to_numpy()

    # Get cell types as strings.
    cell_types = adata.obs[cell_type_col].astype(str)

    # Stable ordered list of unique samples.
    samples = pd.Index(pd.unique(sample_key))

    # Build a crosstab: rows = samples, columns = cell types, values = counts.
    ct_data = pd.DataFrame(
        {
            "sample": sample_key,
            "cell_type": cell_types,
        }
    )
    counts_df = pd.crosstab(ct_data["sample"], ct_data["cell_type"], margins=False)

    # Reindex to ensure stable ordering and integer type.
    counts_df = counts_df.reindex(samples, fill_value=0).astype(np.int64)
    counts_df.index.name = None

    # Assemble aligned per-sample metadata (first value per group).
    meta_source = adata.obs.assign(_sample=sample_key)
    meta_cols = [donor_col, condition_col, *extra_obs]
    sample_meta = meta_source.groupby("_sample")[meta_cols].first().reindex(samples)
    sample_meta.index.name = None

    return CelltypeCounts(counts=counts_df, sample_meta=sample_meta)


__all__ = ["CelltypeCounts", "aggregate_celltype_counts"]
