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
    "PairingDecision",
    "PseudobulkResult",
    "aggregate_pseudobulk",
    "resolve_donor_pairing",
]
