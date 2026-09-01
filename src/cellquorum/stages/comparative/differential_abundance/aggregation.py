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


def build_cell_distribution_summary(
    counts: pd.DataFrame,
    conditions: pd.Series,
    *,
    case: str,
    control: str,
    test_results: pd.DataFrame | None = None,
    cell_type_name: str = "cell_type",
    test_cell_type_col: str = "cell_type",
    pvalue_col: str = "pvalue",
    adj_pvalue_col: str = "fdr",
) -> pd.DataFrame:
    """
    Build a per-cell-type composition summary grouped by condition.

    Produces the "Cell Distribution Summary" table: one row per cell type with
    pooled absolute counts and within-condition relative percentages for the
    case and control arms, plus the differential-abundance p-value and adjusted
    p-value attached to the case arm. This is a purely descriptive pooled view
    (every cell in a condition contributes), distinct from any per-sample mean
    the underlying test may use — which is why it is its own table.

    Nothing here is study-specific: condition labels come from ``case``/``control``
    and column names are generic (``case_*`` / ``control_*``); the display layer
    maps them to the study's condition names.

    Args:
        counts: Samples (rows) × cell types (columns) integer count matrix, as
            produced by :func:`aggregate_celltype_counts`.
        conditions: Per-sample condition labels, indexed by the same sample ids
            as ``counts`` (e.g. ``CelltypeCounts.sample_meta[condition_col]``).
        case: Condition label treated as the case/disease arm.
        control: Condition label treated as the control/normal arm.
        test_results: Optional per-cell-type DA results carrying p-values and
            adjusted p-values to attach to the case arm. When ``None`` the
            p-value columns are emitted as NaN so the schema stays stable.
        cell_type_name: Name of the emitted cell-type column.
        test_cell_type_col: Cell-type column name within ``test_results``.
        pvalue_col: P-value column name within ``test_results``.
        adj_pvalue_col: Adjusted p-value column name within ``test_results``.

    Returns:
        A DataFrame with one row per cell type (alphabetically ordered) and
        columns ``[cell_type_name, case_absolute, case_relative_pct,
        case_pvalue, case_adj_pvalue, control_absolute, control_relative_pct]``.
    """

    # Align condition labels to the count matrix's sample index.
    aligned_conditions = conditions.reindex(counts.index)

    # Cell types are the count columns, presented in a stable alphabetical order.
    cell_types = sorted(str(col) for col in counts.columns)

    # Pooled absolute counts per condition arm (every cell contributes).
    case_mask = (aligned_conditions == case).to_numpy()
    control_mask = (aligned_conditions == control).to_numpy()
    case_absolute = counts.loc[case_mask].sum(axis=0)
    control_absolute = counts.loc[control_mask].sum(axis=0)

    # Within-condition relative percentages (guard against an empty arm).
    case_total = float(case_absolute.sum())
    control_total = float(control_absolute.sum())

    # Optional p / adjusted-p lookup keyed by cell type (case arm only).
    pvalue_lookup: dict[str, float] = {}
    adj_pvalue_lookup: dict[str, float] = {}
    if test_results is not None and not test_results.empty:
        keyed = test_results.set_index(test_results[test_cell_type_col].astype(str))
        if pvalue_col in keyed.columns:
            pvalue_lookup = keyed[pvalue_col].to_dict()
        if adj_pvalue_col in keyed.columns:
            adj_pvalue_lookup = keyed[adj_pvalue_col].to_dict()

    rows = []
    for ct in cell_types:
        case_abs = int(case_absolute.get(ct, 0))
        control_abs = int(control_absolute.get(ct, 0))
        rows.append(
            {
                cell_type_name: ct,
                "case_absolute": case_abs,
                "case_relative_pct": (100.0 * case_abs / case_total) if case_total > 0 else 0.0,
                "case_pvalue": pvalue_lookup.get(ct, np.nan),
                "case_adj_pvalue": adj_pvalue_lookup.get(ct, np.nan),
                "control_absolute": control_abs,
                "control_relative_pct": (
                    100.0 * control_abs / control_total if control_total > 0 else 0.0
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            cell_type_name,
            "case_absolute",
            "case_relative_pct",
            "case_pvalue",
            "case_adj_pvalue",
            "control_absolute",
            "control_relative_pct",
        ],
    )


__all__ = ["CelltypeCounts", "aggregate_celltype_counts", "build_cell_distribution_summary"]
