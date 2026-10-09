"""Specimen-level pseudobulk aggregation and support auditing for differential expression.

One row of pseudobulk counts is one biological *specimen* (one ``sample_col`` value):
single-cell counts are summed within each specimen using a sparse indicator matrix, so
the aggregation stays fast on large objects. A subject that contributes several
specimens (e.g. two tissues) therefore yields several profiles, never one merged
profile. The legacy donor x condition grouping remains available for pairwise
case/control workflows.

Responsibilities are deliberately separate:

* :func:`aggregate_pseudobulk` is the deterministic count aggregation over the cells it
  is given. It knows nothing about specimens that have no cells.
* :func:`build_specimen_universe` records every specimen of the experiment from the
  FULL object, before any cell-type subsetting.
* :func:`evaluate_pseudobulk_support` compares the aggregate with that universe and
  decides, with an explicit reason, which specimens are usable as expression samples.
  A specimen with zero cells of a population is ``population_absent`` -- an abundance
  observation, never an all-zero expression profile.

Cannibalized and generalized from paired_sc.stats.paired.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.core.exceptions import CellQuorumDataError

# Exclusion reasons written to the support table (stable, machine-readable).
REASON_POPULATION_ABSENT = "population_absent"
REASON_BELOW_MIN_CELLS = "below_min_cells"
REASON_NO_ELIGIBLE_CELLS = "population_present_but_no_de_eligible_cells"
REASON_BELOW_MIN_CELLS_AFTER_QC = "below_min_cells_after_qc"
REASON_ZERO_LIBRARY = "zero_library_size"

# Column names the support table owns; a factor/subject column may not shadow them.
_SUPPORT_RESERVED = frozenset(
    {
        "sample_id",
        "subject_id",
        "n_cells",
        "n_cells_observed",
        "n_cells_de_eligible",
        "n_cells_qc_excluded",
        "library_size",
        "observed_population",
        "included_for_expression",
        "exclusion_reason",
    }
)

# Column the specimen universe adds to the metadata it carries.
_UNIVERSE_CELLS = "n_cells_total"

# How many offending specimens an error message spells out before truncating.
_MAX_REPORTED = 5


@dataclass(frozen=True)
class PseudobulkResult:
    """Pseudobulk counts and aligned sample metadata.

    Args:
        counts: Pseudo-samples x genes integer count matrix.
        sample_meta: Per-pseudo-sample metadata aligned to counts.index.
        n_cells: Cells summed into each pseudo-sample, aligned to counts.index.
            ``None`` only for results built by hand (older call sites).
    """

    # Pseudo-samples x genes integer counts.
    counts: pd.DataFrame

    # Per-pseudo-sample metadata (donor, condition, optional covariates).
    sample_meta: pd.DataFrame

    # Cells summed into each pseudo-sample.
    n_cells: pd.Series | None = None

    @property
    def library_size(self) -> pd.Series:
        """Total counts per pseudo-sample, aligned to ``counts.index``."""
        return self.counts.sum(axis=1)


@dataclass(frozen=True)
class PseudobulkSupport:
    """Per-specimen support audit over the complete specimen universe.

    Args:
        table: One row per specimen of the universe (including specimens with no
            cells of this population) with the columns ``sample_id``,
            ``subject_id`` (when a subject column was given), one column per factor,
            ``n_cells``, ``library_size``, ``observed_population``,
            ``included_for_expression`` and ``exclusion_reason``.
        min_cells_per_sample: The per-specimen cell floor that was applied, if any.
    """

    table: pd.DataFrame
    min_cells_per_sample: int | None = None

    @property
    def included_ids(self) -> list[str]:
        """Specimen ids usable as expression samples, in universe order."""
        keep = self.table["included_for_expression"].to_numpy(dtype=bool)
        return [str(s) for s in self.table.loc[keep, "sample_id"]]

    @property
    def excluded_ids(self) -> list[str]:
        """Specimen ids excluded from expression analysis, in universe order."""
        keep = self.table["included_for_expression"].to_numpy(dtype=bool)
        return [str(s) for s in self.table.loc[~keep, "sample_id"]]

    @property
    def n_included(self) -> int:
        """Number of specimens usable as expression samples."""
        return len(self.included_ids)


def _require_columns(obs: pd.DataFrame, columns: list[str]) -> None:
    """Raise ``KeyError`` naming the first requested obs column that is absent."""
    for column in columns:
        if column not in obs.columns:
            raise KeyError(f"obs column '{column}' not found for pseudobulk aggregation.")


def _reject_null_keys(obs: pd.DataFrame, columns: list[str]) -> None:
    """Fail when a grouping column has missing values (never coerce them to ``"nan"``)."""
    for column in columns:
        n_null = int(obs[column].isna().sum())
        if n_null:
            raise CellQuorumDataError(
                f"obs column '{column}' identifies the pseudobulk unit but is missing for "
                f"{n_null} of {len(obs)} cells; a missing identifier cannot be assigned to "
                "a specimen. Fix or drop those cells before aggregating."
            )


def _group_metadata(
    obs: pd.DataFrame,
    group_key: np.ndarray,
    groups: pd.Index,
    columns: list[str],
    *,
    optional: frozenset[str],
) -> pd.DataFrame:
    """Collect one value per group for each DECLARED column, validating constancy.

    A declared column must take exactly one value within every group AND be fully
    observed there. Partial missingness is always an error (a corrupt specimen);
    complete missingness is an error unless the caller declared the column optional.
    Only ``columns`` are inspected, so a per-cell quantity that was never requested can
    not accidentally become -- or break -- sample metadata.

    Args:
        obs: Cell-level metadata.
        group_key: Group label per cell (aligned to ``obs``).
        groups: Ordered unique group labels; the row order of the result.
        columns: Declared sample-level columns (unique).
        optional: Columns permitted to be entirely missing within a group.

    Returns:
        A frame indexed by ``groups`` holding each group's single value per column.

    Raises:
        CellQuorumDataError: If any declared column is non-constant or missing.
    """

    result: dict[str, pd.Series] = {}
    for column in columns:
        series = obs[column]
        grouped = series.groupby(group_key, sort=False)
        n_cells = grouped.size()
        n_null = series.isna().astype(int).groupby(group_key, sort=False).sum()
        n_unique = grouped.nunique()

        multi = n_unique > 1
        partial = (n_null > 0) & (n_null < n_cells)
        empty = (n_null == n_cells) & (column not in optional)
        offenders = n_cells.index[(multi | partial | empty).to_numpy()]
        if len(offenders):
            lines = []
            for group in list(offenders)[:_MAX_REPORTED]:
                values = series[group_key == group]
                observed = sorted({str(v) for v in values.dropna().unique()})
                if values.isna().any():
                    observed.append("<missing>")
                lines.append(
                    f"specimen '{group}': column '{column}' has values {observed} "
                    f"across {int(n_cells[group])} cells"
                )
            extra = len(offenders) - _MAX_REPORTED
            suffix = f" (+{extra} more specimen(s))" if extra > 0 else ""
            raise CellQuorumDataError(
                "Sample-level metadata must have exactly one value, fully observed, within "
                "each pseudobulk unit: " + "; ".join(lines) + suffix + "."
            )
        result[column] = grouped.first().reindex(groups)

    frame = pd.DataFrame(result, index=groups)
    frame.index.name = None
    return frame


def aggregate_pseudobulk(
    adata: ad.AnnData,
    *,
    layer: str,
    sample_col: str | None = None,
    metadata_cols: list[str] | None = None,
    optional_metadata_cols: list[str] | None = None,
    donor_col: str | None = None,
    condition_col: str | None = None,
    extra_obs: list[str] | None = None,
) -> PseudobulkResult:
    """
    Aggregate single cells to pseudobulk counts, one row per pseudobulk unit.

    Two grouping modes, selected by which arguments are given (exactly one):

    * **Specimen mode** (``sample_col``): one row per distinct ``sample_col`` value --
      one biological specimen. ``metadata_cols`` are the sample-level fields carried
      into ``sample_meta``.
    * **Legacy donor x condition mode** (``donor_col`` + ``condition_col``): one row
      per (donor, condition), keyed ``"<donor>__<condition>"``. ``extra_obs`` are the
      carried covariates. Kept for pairwise case/control workflows.

    In both modes every DECLARED metadata column must be constant and fully observed
    within the unit (see :func:`_group_metadata`); no ``first()`` is silently taken,
    and missing identifiers are rejected rather than coerced to ``"nan"``. Only
    declared columns are checked. The output is dense by design: a pseudobulk matrix
    has few rows (units), so densifying it is cheap, whereas the cell x gene input
    stays sparse throughout.

    Args:
        adata: Input AnnData with raw counts in ``layer``.
        layer: Layer name holding raw counts.
        sample_col: obs column identifying the biological specimen (specimen mode).
        metadata_cols: Sample-level obs columns carried into ``sample_meta`` (specimen
            mode); each must be constant and fully observed within a specimen.
        optional_metadata_cols: Columns (specimen mode) that may be entirely missing
            within a specimen; partial missingness is still an error. They are added
            to the carried columns if not already listed.
        donor_col: obs column identifying donors (legacy mode).
        condition_col: obs column holding the condition label (legacy mode).
        extra_obs: Extra covariate columns carried into ``sample_meta`` (legacy mode);
            each must be constant within a (donor, condition) group.

    Returns:
        PseudobulkResult with integer counts, aligned sample metadata and the number
        of cells summed into each unit.

    Raises:
        ValueError: If the grouping arguments do not select exactly one mode.
        KeyError: If the layer or a required obs column is absent.
        CellQuorumDataError: If an identifier is missing, or declared metadata is
            non-constant or missing within a unit.
    """

    specimen_mode = sample_col is not None
    legacy_mode = donor_col is not None or condition_col is not None
    legacy_incomplete = legacy_mode and (donor_col is None or condition_col is None)
    if specimen_mode == legacy_mode or legacy_incomplete:
        raise ValueError(
            "aggregate_pseudobulk needs either sample_col (specimen mode) or both "
            "donor_col and condition_col (legacy donor x condition mode), not a mix."
        )
    if legacy_mode and (metadata_cols or optional_metadata_cols):
        raise ValueError("metadata_cols apply to specimen mode; use extra_obs in legacy mode.")
    if specimen_mode and extra_obs:
        raise ValueError("extra_obs applies to legacy mode; use metadata_cols in specimen mode.")

    # Confirm the counts layer exists.
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found; available: {list(adata.layers)}.")

    optional = frozenset(optional_metadata_cols or [])
    if specimen_mode:
        key_cols = [str(sample_col)]
        carried = [*(metadata_cols or []), *(optional_metadata_cols or [])]
    else:
        key_cols = [str(donor_col), str(condition_col)]
        carried = [*key_cols, *(extra_obs or [])]
    carried = list(dict.fromkeys(carried))
    _require_columns(adata.obs, [*key_cols, *carried])
    _reject_null_keys(adata.obs, key_cols)

    # Build the pseudobulk-unit key per cell.
    if specimen_mode:
        group_key = adata.obs[key_cols[0]].astype(str).to_numpy()
    else:
        donor = adata.obs[key_cols[0]].astype(str)
        condition = adata.obs[key_cols[1]].astype(str)
        group_key = donor.to_numpy() + "__" + condition.to_numpy()

    # Stable ordered list of units (first-appearance order).
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

    # Declared sample-level metadata, validated constant + fully observed per unit.
    sample_meta = _group_metadata(adata.obs, group_key, groups, carried, optional=optional)
    n_cells = pd.Series(np.bincount(rows, minlength=len(groups)), index=groups, name="n_cells")

    return PseudobulkResult(counts=counts_df, sample_meta=sample_meta, n_cells=n_cells)


def build_specimen_universe(
    obs: pd.DataFrame,
    *,
    sample_col: str,
    metadata_cols: list[str],
    optional_metadata_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Record every specimen of the experiment, from the FULL (unsubsetted) cell table.

    Call this on the complete object before restricting to a cell type: once the
    object is subset, a specimen with zero cells of that population leaves no trace,
    and "specimen exists, population absent" becomes indistinguishable from
    "specimen was never part of the experiment". The result is the authoritative
    specimen list that :func:`evaluate_pseudobulk_support` audits against.

    Declared metadata is validated exactly as in :func:`aggregate_pseudobulk`, so a
    specimen mapped to two subjects or two factor levels fails here.

    Args:
        obs: Cell-level metadata of the complete experiment (``adata.obs``).
        sample_col: obs column identifying the specimen.
        metadata_cols: Sample-level columns (subject, factors, ...) to record.
        optional_metadata_cols: Columns allowed to be entirely missing per specimen.

    Returns:
        One row per specimen, indexed by sample id, with the metadata columns and
        ``n_cells_total`` (cells of the specimen across all populations).

    Raises:
        KeyError: If a requested column is absent.
        CellQuorumDataError: If an id is missing or declared metadata is inconsistent.
    """

    optional = frozenset(optional_metadata_cols or [])
    carried = list(dict.fromkeys([*metadata_cols, *(optional_metadata_cols or [])]))
    if _UNIVERSE_CELLS in carried:
        raise ValueError(f"'{_UNIVERSE_CELLS}' is reserved for the specimen universe.")
    _require_columns(obs, [sample_col, *carried])
    _reject_null_keys(obs, [sample_col])

    group_key = obs[sample_col].astype(str).to_numpy()
    groups = pd.Index(pd.unique(group_key))
    universe = _group_metadata(obs, group_key, groups, carried, optional=optional)
    sizes = pd.Series(group_key).value_counts()
    universe[_UNIVERSE_CELLS] = sizes.reindex(groups).to_numpy()
    return universe


def evaluate_pseudobulk_support(
    pb: PseudobulkResult,
    *,
    sample_metadata: pd.DataFrame,
    subject_col: str | None = None,
    factor_cols: list[str] | None = None,
    min_cells_per_sample: int | None = None,
    observed_n_cells: pd.Series | None = None,
) -> PseudobulkSupport:
    """
    Audit pseudobulk specimens against the complete specimen universe.

    Every specimen in ``sample_metadata`` gets one row, whether or not it has cells in
    ``pb`` -- so a specimen with zero cells of the analysed population is reported as
    ``population_absent`` instead of disappearing. Exclusion reasons, in precedence
    order:

    * ``population_absent``: no cells of this population in the specimen. This is
      abundance information, not an expression measurement.
    * ``population_present_but_no_de_eligible_cells`` (only with ``observed_n_cells``):
      the population exists in the specimen but QC leaves none of its cells eligible.
    * ``below_min_cells_after_qc`` (only with ``observed_n_cells``): fewer than
      ``min_cells_per_sample`` eligible cells, with some cells excluded by QC.
    * ``below_min_cells``: fewer than ``min_cells_per_sample`` cells.
    * ``zero_library_size``: cells exist but their counts sum to zero.

    ``pb`` holds the cells that may enter the model (QC-eligible). ``observed_n_cells``
    carries the population's cell count per specimen BEFORE eligibility, so a specimen
    whose population is present but QC-excluded is never confused with one where the
    population was never observed.

    No specimen is ever given a zero-filled expression profile. This function only
    decides; use :func:`restrict_to_included` to apply the decision to ``pb``.

    Args:
        pb: Aggregated pseudobulk of the analysed population (must carry ``n_cells``).
        sample_metadata: The specimen universe, one row per specimen indexed by sample
            id (see :func:`build_specimen_universe`).
        subject_col: Column of ``sample_metadata`` naming the subject; reported as
            ``subject_id``.
        factor_cols: Columns reported as factor values. Defaults to every metadata
            column other than the subject and the universe cell total.
        min_cells_per_sample: Minimum cells for a specimen to be usable; ``None`` means
            no cell floor beyond having any cells.
        observed_n_cells: Optional per-specimen cell count of this population before QC
            eligibility (indexed by specimen id; missing specimens count as 0). When
            given, the table also reports ``n_cells_observed``, ``n_cells_de_eligible``
            and ``n_cells_qc_excluded``.

    Returns:
        A :class:`PseudobulkSupport` whose table covers the whole universe.

    Raises:
        ValueError: If ``pb`` lacks ``n_cells``, ``min_cells_per_sample`` is below 1, or
            a reported column would shadow a support-table column.
        KeyError: If ``subject_col`` or a factor column is not in ``sample_metadata``.
        CellQuorumDataError: If ``pb`` holds a specimen outside the universe, or its
            carried metadata disagrees with the universe.
    """

    if pb.n_cells is None:
        raise ValueError("evaluate_pseudobulk_support needs a PseudobulkResult with n_cells.")
    if min_cells_per_sample is not None and min_cells_per_sample < 1:
        raise ValueError(f"min_cells_per_sample must be >= 1 or None; got {min_cells_per_sample}.")

    universe = sample_metadata
    if factor_cols is None:
        factor_cols = [c for c in universe.columns if c not in (subject_col, _UNIVERSE_CELLS)]
    for column in [*([subject_col] if subject_col else []), *factor_cols]:
        if column not in universe.columns:
            raise KeyError(f"column '{column}' not found in the specimen universe.")
    shadowed = _SUPPORT_RESERVED.intersection(factor_cols)
    if shadowed:
        raise ValueError(f"factor column(s) {sorted(shadowed)} collide with support-table columns.")

    # Every aggregated specimen must belong to the universe, with matching metadata.
    outside = [s for s in pb.counts.index if s not in universe.index]
    if outside:
        raise CellQuorumDataError(
            f"Pseudobulk specimen(s) {outside[:_MAX_REPORTED]} are not in the specimen "
            "universe; the universe must be built from the full experiment."
        )
    shared = [c for c in pb.sample_meta.columns if c in universe.columns]
    for column in shared:
        observed = pb.sample_meta[column].astype(object)
        expected = universe.loc[observed.index, column].astype(object)
        differs = ~((observed == expected) | (observed.isna() & expected.isna()))
        if differs.any():
            bad = list(observed.index[differs.to_numpy()])[:_MAX_REPORTED]
            raise CellQuorumDataError(
                f"Metadata column '{column}' of the aggregated pseudobulk disagrees with the "
                f"specimen universe for specimen(s) {bad}."
            )

    n_cells = pb.n_cells.reindex(universe.index).fillna(0).astype(np.int64)
    library = pb.library_size.reindex(universe.index).fillna(0).astype(np.int64)
    if observed_n_cells is None:
        n_observed = n_cells
    else:
        n_observed = observed_n_cells.reindex(universe.index).fillna(0).astype(np.int64)
        if bool((n_observed < n_cells).any()):
            bad = list(n_observed.index[(n_observed < n_cells).to_numpy()])[:_MAX_REPORTED]
            raise CellQuorumDataError(
                f"observed_n_cells is below the eligible cell count for specimen(s) {bad}."
            )
    observed_pop = n_observed > 0
    qc_excluded = n_observed - n_cells

    reason = pd.Series([None] * len(universe), index=universe.index, dtype=object)
    reason[library == 0] = REASON_ZERO_LIBRARY
    if min_cells_per_sample is not None:
        below = n_cells < min_cells_per_sample
        reason[below] = REASON_BELOW_MIN_CELLS
        if observed_n_cells is not None:
            reason[below & (qc_excluded > 0)] = REASON_BELOW_MIN_CELLS_AFTER_QC
    if observed_n_cells is not None:
        reason[observed_pop & (n_cells == 0)] = REASON_NO_ELIGIBLE_CELLS
    reason[~observed_pop] = REASON_POPULATION_ABSENT

    table = pd.DataFrame({"sample_id": [str(s) for s in universe.index]})
    if subject_col:
        table["subject_id"] = universe[subject_col].to_numpy()
    for column in factor_cols:
        table[column] = universe[column].to_numpy()
    table["n_cells"] = n_cells.to_numpy()
    if observed_n_cells is not None:
        table["n_cells_observed"] = n_observed.to_numpy()
        table["n_cells_de_eligible"] = n_cells.to_numpy()
        table["n_cells_qc_excluded"] = qc_excluded.to_numpy()
    table["library_size"] = library.to_numpy()
    table["observed_population"] = observed_pop.to_numpy()
    table["included_for_expression"] = reason.isna().to_numpy()
    table["exclusion_reason"] = reason.to_numpy()
    return PseudobulkSupport(table=table, min_cells_per_sample=min_cells_per_sample)


def restrict_to_included(pb: PseudobulkResult, support: PseudobulkSupport) -> PseudobulkResult:
    """
    Keep only the specimens the support audit marked usable for expression analysis.

    Args:
        pb: Aggregated pseudobulk.
        support: Result of :func:`evaluate_pseudobulk_support` for the same ``pb``.

    Returns:
        A new :class:`PseudobulkResult` with counts, metadata and ``n_cells`` restricted
        to the included specimens (in ``pb``'s original row order).
    """

    included = set(support.included_ids)
    keep = [s for s in pb.counts.index if s in included]
    return PseudobulkResult(
        counts=pb.counts.loc[keep],
        sample_meta=pb.sample_meta.loc[keep],
        n_cells=None if pb.n_cells is None else pb.n_cells.loc[keep],
    )


@dataclass(frozen=True)
class PairingDecision:
    """Resolved donor-pairing outcome for a pseudobulk DE fit.

    Args:
        paired: Final paired flag after any auto-promotion.
        counts: Pseudo-sample counts, restricted to complete donor pairs when
            the final fit is paired and some donors appear in only one arm.
        sample_meta: Sample metadata aligned to ``counts`` (same restriction).
        n_complete_pairs: Distinct donors contributing both the case and the
            control arm.
        notes: Human-readable design notes (auto-promotion and/or restriction).
    """

    # Final paired flag after auto-promotion.
    paired: bool

    # Counts restricted to complete pairs when the paired fit needs it.
    counts: pd.DataFrame

    # Sample metadata aligned to counts.
    sample_meta: pd.DataFrame

    # Donors contributing both arms.
    n_complete_pairs: int

    # Design notes describing any promotion/restriction.
    notes: list[str]


def resolve_donor_pairing(
    pb: PseudobulkResult,
    *,
    donor_col: str,
    condition_col: str,
    case: str,
    control: str,
    paired: bool,
) -> PairingDecision:
    """
    Decide the paired flag and complete-pair restriction for a pseudobulk fit.

    This is the safety net for the silent-wrong-DE class where a fully matched
    (every donor in both arms) design is analysed *unpaired*, leaving donor
    baseline variance in the residual and producing false nulls. It performs two
    corrections, in order:

    1. **Auto-promote**: when ``paired`` is ``False`` but every donor contributes
       both a ``case`` and a ``control`` pseudo-sample (and there are at least two
       donors), the fit is promoted to paired so donor is blocked.
    2. **Restrict**: when the (possibly promoted) fit is paired but some donors
       appear in only one arm, those donors are dropped so the donor-blocked
       design stays estimable. The drop is reported, never silent.

    The logic is pure — it reads ``pb`` and returns a decision — so the promotion
    and restriction branches are testable without invoking the R backend.

    Args:
        pb: Pseudobulk counts and aligned sample metadata.
        donor_col: Sample-metadata column identifying donors.
        condition_col: Sample-metadata column holding the condition label.
        case: Case condition label.
        control: Control condition label.
        paired: Whether the caller declared a paired design.

    Returns:
        A :class:`PairingDecision` with the final paired flag, the (possibly
        restricted) counts and metadata, the complete-pair count, and notes.
    """

    # Start from the aggregated pseudobulk; restriction may narrow these.
    sample_meta = pb.sample_meta
    counts = pb.counts

    # Compute donor support per arm.
    case_donors = set(sample_meta.loc[sample_meta[condition_col] == case, donor_col])
    control_donors = set(sample_meta.loc[sample_meta[condition_col] == control, donor_col])
    all_donors = case_donors | control_donors
    complete_pairs = case_donors & control_donors
    notes: list[str] = []

    # Auto-promote to paired when the design is fully matched but was not
    # declared paired — the corrected, higher-powered default for such data.
    if not paired and complete_pairs and complete_pairs == all_donors and len(all_donors) >= 2:
        paired = True
        notes.append(
            "Auto-promoted to PAIRED: every donor contributes both a "
            f"{case} and a {control} pseudobulk sample "
            f"({len(complete_pairs)} complete donor pairs). Blocking on donor."
        )

    # For a paired fit, keep only donors with a complete pair so the
    # donor-blocked design stays estimable; log what was dropped.
    if paired:
        incomplete = all_donors - complete_pairs
        if incomplete:
            keep = sample_meta.index[sample_meta[donor_col].isin(complete_pairs)]
            counts = counts.loc[keep]
            sample_meta = sample_meta.loc[keep]
            notes.append(
                f"Paired fit restricted to {len(complete_pairs)} complete donor "
                f"pairs; dropped {len(incomplete)} donor(s) present in only one "
                f"arm: {sorted(str(d) for d in incomplete)}."
            )

    return PairingDecision(
        paired=paired,
        counts=counts,
        sample_meta=sample_meta,
        n_complete_pairs=len(complete_pairs),
        notes=notes,
    )


__all__ = [
    "REASON_BELOW_MIN_CELLS",
    "REASON_BELOW_MIN_CELLS_AFTER_QC",
    "REASON_NO_ELIGIBLE_CELLS",
    "REASON_POPULATION_ABSENT",
    "REASON_ZERO_LIBRARY",
    "PairingDecision",
    "PseudobulkResult",
    "PseudobulkSupport",
    "aggregate_pseudobulk",
    "build_specimen_universe",
    "evaluate_pseudobulk_support",
    "resolve_donor_pairing",
    "restrict_to_included",
]
