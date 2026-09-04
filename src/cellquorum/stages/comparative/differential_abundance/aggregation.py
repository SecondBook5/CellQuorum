"""Sample × cell-type count aggregation for differential abundance analysis.

Aggregates single-cell observations into sample (donor × condition) × cell-type
integer count tables, used by abundance testing methods (scCODA, propeller).
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.labels import as_label_strings


@dataclass(frozen=True)
class CelltypeCounts:
    """Sample × cell-type counts and aligned sample metadata.

    Attributes:
        counts: Samples (rows) × cell types (columns) integer count matrix.
            Index is the sample identifier (donor_condition).
        sample_meta: Per-sample metadata aligned to counts.index, containing
            donor_col, condition_col, and any extra_obs columns.
        n_unlabeled: Cells excluded because one of the grouping columns was
            missing for them. Zero on a fully labelled object.
        notes: Human-readable account of any exclusion, for the calling method to
            put in its stage notes. Authored here so all three count-based DA
            methods report the same thing in the same words.
    """

    counts: pd.DataFrame
    sample_meta: pd.DataFrame
    n_unlabeled: int = 0
    notes: tuple[str, ...] = ()


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

        Cells with a missing value in any grouping column are excluded, and the
        exclusion is reported rather than absorbed. See the inline note for why a
        missing label must not become a cell type.
    """

    # Confirm required obs columns exist.
    group_cols = (donor_col, condition_col, cell_type_col, *extra_obs)
    for column in group_cols:
        if column not in adata.obs.columns:
            raise KeyError(f"obs column '{column}' not found for cell-type count aggregation.")

    # Drop cells with a missing grouping value BEFORE anything is counted.
    #
    # ``.astype(str)`` renders a missing label as the literal string "nan", and
    # ``crosstab`` then reports that as a cell type. Cell-state columns routinely
    # carry missing values by design: subclustering leaves NaN for every cell
    # outside the analysed focus -- outside the focus labels, below the per-sample
    # cell floor, or failed by the donor gate -- so a within-lineage abundance test
    # on a subcluster column would acquire a pseudo-type made entirely of
    # technically-excluded cells. That pseudo-type competes for the compositional
    # reference, gets its own effect row, and sits in the denominator of every
    # other type's proportion. Its size is a property of the FILTER rather than of
    # the biology, and filters are rarely balanced across arms, so leaving it in
    # tests the exclusion rule as though it were a cell state.
    obs = adata.obs
    labeled = obs[list(group_cols)].notna().all(axis=1).to_numpy()
    n_unlabeled = int(labeled.size - labeled.sum())
    notes: tuple[str, ...] = ()
    if n_unlabeled:
        per_column = ", ".join(
            f"{column} ({int(obs[column].isna().sum())})"
            for column in dict.fromkeys(group_cols)
            if int(obs[column].isna().sum())
        )
        notes = (
            f"Excluded {n_unlabeled} of {labeled.size} cells "
            f"({n_unlabeled / labeled.size:.1%}) from the abundance counts for a missing "
            f"grouping value: {per_column}. A missing label is an absence of measurement, "
            f"not a cell type; counting it as one would put the exclusion rule itself into "
            f"the composition being tested.",
        )
        obs = obs.loc[labeled]

    # Build the sample key per cell as donor_condition. Canonical label strings,
    # not ``astype(str)``: see :mod:`cellquorum.core.labels` for what "1.0" costs.
    donor = as_label_strings(obs[donor_col])
    condition = as_label_strings(obs[condition_col])
    sample_key = donor.to_numpy().astype(str) + "_" + condition.to_numpy().astype(str)

    # Get cell types as canonical label strings.
    cell_types = as_label_strings(obs[cell_type_col])

    # Nothing survived the label check. Return the empty shape with the note
    # attached: crosstab has no group keys to work with, and a caller that skips
    # still needs to be able to say why.
    if obs.shape[0] == 0:
        return CelltypeCounts(
            counts=pd.DataFrame(),
            sample_meta=pd.DataFrame(columns=[donor_col, condition_col, *extra_obs]),
            n_unlabeled=n_unlabeled,
            notes=notes,
        )

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

    # Assemble aligned per-sample metadata (first value per group). Donor and
    # condition carry their canonical form so the metadata agrees with the sample
    # key built from them -- a method that matches sample_meta[condition_col]
    # against the config's `case` string cannot do it against a float. extra_obs is
    # deliberately left as-is: those are covariates, and a numeric covariate has to
    # stay numeric for the model that consumes it.
    meta_source = obs.assign(_sample=sample_key, **{donor_col: donor, condition_col: condition})
    meta_cols = [donor_col, condition_col, *extra_obs]
    sample_meta = meta_source.groupby("_sample")[meta_cols].first().reindex(samples)
    sample_meta.index.name = None

    return CelltypeCounts(
        counts=counts_df,
        sample_meta=sample_meta,
        n_unlabeled=n_unlabeled,
        notes=notes,
    )


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


def build_composition_proportions(
    counts: pd.DataFrame,
    conditions: pd.Series,
    donors: pd.Series,
    *,
    case: str,
    control: str,
) -> pd.DataFrame:
    """
    Build tidy per-sample cell-type composition proportions.

    Produces the long-format backing table for the composition figure: one row
    per (sample, cell type) for every sample in the case or control arm, with
    the within-sample proportion (a sample's cell-type count divided by that
    sample's total). Proportions sum to 1 within each sample, so the same table
    drives both the per-patient stacked bar and any condition-level summary the
    plotting layer derives from it.

    Nothing here is study-specific: the arms come from ``case``/``control`` and
    the sample/donor identities come from the aligned metadata; samples in any
    other condition are dropped.

    Args:
        counts: Samples (rows) × cell types (columns) integer count matrix, as
            produced by :func:`aggregate_celltype_counts`.
        conditions: Per-sample condition labels, indexed by the same sample ids
            as ``counts`` (e.g. ``CelltypeCounts.sample_meta[condition_col]``).
        donors: Per-sample donor labels, indexed by the same sample ids as
            ``counts`` (e.g. ``CelltypeCounts.sample_meta[donor_col]``).
        case: Condition label treated as the case/disease arm.
        control: Condition label treated as the control/normal arm.

    Returns:
        A tidy DataFrame with columns ``[sample, donor, condition, cell_type,
        count, proportion]``, ordered control arm first then case arm, then by
        donor, then alphabetically by cell type.
    """

    # Align condition / donor labels to the count matrix's sample index.
    aligned_conditions = conditions.reindex(counts.index)
    aligned_donors = donors.reindex(counts.index)

    # Cell types presented in a stable alphabetical order.
    cell_types = sorted(str(col) for col in counts.columns)

    # Within-sample totals guard against an all-zero sample.
    sample_totals = counts.sum(axis=1)

    # Control arm first, then case arm — matches the left→right reading of the
    # Normal-vs-Disease figure. Samples in any other condition are excluded.
    ordered_samples: list[str] = []
    for arm in (control, case):
        arm_samples = [s for s in counts.index if aligned_conditions.get(s) == arm]
        arm_samples.sort(key=lambda s: str(aligned_donors.get(s)))
        ordered_samples.extend(arm_samples)

    rows = []
    for sample in ordered_samples:
        total = float(sample_totals.get(sample, 0))
        for ct in cell_types:
            count = int(counts.at[sample, ct])
            rows.append(
                {
                    "sample": str(sample),
                    "donor": str(aligned_donors.get(sample)),
                    "condition": str(aligned_conditions.get(sample)),
                    "cell_type": ct,
                    "count": count,
                    "proportion": (count / total) if total > 0 else 0.0,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["sample", "donor", "condition", "cell_type", "count", "proportion"],
    )


__all__ = [
    "CelltypeCounts",
    "aggregate_celltype_counts",
    "build_cell_distribution_summary",
    "build_composition_proportions",
]
