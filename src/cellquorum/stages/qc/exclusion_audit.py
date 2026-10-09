"""Two different things QC does to a cell, reported as two different tables.

* **Physically removed**: the barcode is gone from the AnnData (hard floors, or doublet
  removal when ``doublets.remove`` is on). Accounting: ``input - removed = retained``.
* **Inference-excluded**: the barcode stays in the master object but may not fit, receive or
  inform one or more analyses (the eligibility masks). It is *retained*, never "removed".

Conflating them (one ``removed_cells.csv`` listing both) makes manuscript QC accounting
internally inconsistent: thousands of retained cells appear as deletions.
"""

from __future__ import annotations

import re

import anndata as ad
import pandas as pd

from cellquorum.stages.qc.evidence import AdjudicationReason, QCStateInitial

_MASK_COLUMN = re.compile(r"^qc_(fit|transform|inference)_[a-z]+$")

# Mutually exclusive categories, in precedence order of the checks below.
PROBABLE_MULTIPLET_ONLY = "probable_multiplet_only"
MULTIPLET_PLUS_DAMAGE = "multiplet_plus_damage"
QUARANTINE_CAPTURE = "quarantine_capture_complexity"
QUARANTINE_MULTI_FAMILY = "quarantine_multi_family_damage"
QUARANTINE_OTHER = "quarantine_other"
BORDERLINE_NONMULTIPLET = "borderline_nonmultiplet"
OTHER = "other"


def eligibility_mask_columns(obs: pd.DataFrame) -> list[str]:
    """The stored per-analysis eligibility mask columns present in ``obs``."""
    return [c for c in obs.columns if _MASK_COLUMN.match(str(c))]


def exclusion_category(obs: pd.DataFrame) -> pd.Series:
    """Assign every cell exactly one exclusion category from its QC verdict.

    Categories partition the cells that lose at least one permission, so counts add up
    without double counting; the overlap between damage and multiplet status is carried by
    ``multiplet_plus_damage`` rather than by counting a cell twice.

    Args:
        obs: Cell table carrying ``qc_state_initial`` (and, when present,
            ``qc_state_reason`` / ``qc_probable_multiplet``).

    Returns:
        A string Series aligned to ``obs.index``; ``"eligible"`` for cells keeping every
        stored permission.
    """

    state = obs["qc_state_initial"].astype(str)
    reason = (
        obs["qc_state_reason"].astype(str)
        if "qc_state_reason" in obs
        else pd.Series("", index=obs.index)
    )
    multiplet = (
        obs["qc_probable_multiplet"].fillna(False).astype(bool)
        if "qc_probable_multiplet" in obs
        else pd.Series(False, index=obs.index)
    )
    core = state == str(QCStateInitial.CORE)
    quarantine = state == str(QCStateInitial.QUARANTINE)
    borderline = state == str(QCStateInitial.BORDERLINE)

    category = pd.Series("eligible", index=obs.index, dtype=object)
    category[borderline & ~multiplet] = BORDERLINE_NONMULTIPLET
    category[quarantine & ~multiplet] = QUARANTINE_OTHER
    category[
        quarantine & ~multiplet & (reason == str(AdjudicationReason.UNINFORMATIVE_BARCODE))
    ] = QUARANTINE_CAPTURE
    category[
        quarantine & ~multiplet & (reason == str(AdjudicationReason.CONCORDANT_SEVERE_DAMAGE))
    ] = QUARANTINE_MULTI_FAMILY
    category[multiplet & core] = PROBABLE_MULTIPLET_ONLY
    category[multiplet & ~core] = MULTIPLET_PLUS_DAMAGE
    return category


def inference_excluded_table(adata: ad.AnnData) -> pd.DataFrame:
    """Per-barcode record of cells retained in ``adata`` but barred from some analysis.

    Args:
        adata: The post-floor object carrying the graded verdict and eligibility masks.

    Returns:
        One row per retained cell that loses at least one stored permission, with its
        mutually exclusive ``exclusion_category``, the verdict fields behind it, how many
        stored permissions it lacks, and its DE/manifold/composition masks. Empty (with the
        same columns) when QC wrote no verdict or excluded nobody.
    """

    obs = adata.obs
    columns = [
        "exclusion_category",
        "qc_state_initial",
        "qc_state_reason",
        "qc_primary_driver",
        "qc_concerning_families",
        "qc_probable_multiplet",
        "n_permissions_denied",
    ]
    masks = eligibility_mask_columns(obs)
    shown = [
        c
        for c in ("qc_fit_manifold", "qc_fit_de", "qc_inference_de", "qc_inference_composition")
        if c in obs
    ]
    if "qc_state_initial" not in obs.columns or not masks:
        return pd.DataFrame(columns=[*columns, *shown])

    denied = (~obs[masks].astype(bool)).sum(axis=1)
    out = pd.DataFrame(index=obs.index)
    out["exclusion_category"] = exclusion_category(obs)
    for c in (
        "qc_state_initial",
        "qc_state_reason",
        "qc_primary_driver",
        "qc_concerning_families",
        "qc_probable_multiplet",
    ):
        out[c] = obs[c] if c in obs else pd.NA
    out["n_permissions_denied"] = denied
    for c in shown:
        out[c] = obs[c]
    return out.loc[denied.to_numpy() > 0].copy()


def physically_removed_table(accounting: pd.DataFrame, adata: ad.AnnData) -> pd.DataFrame:
    """Restrict the non-analysable accounting table to barcodes actually absent from ``adata``.

    Args:
        accounting: Output of the stage's removal accounting (every non-analysable barcode).
        adata: The object QC returns.

    Returns:
        Only the rows whose barcode is no longer in ``adata`` (hard floors, doublet removal).
    """
    return accounting.loc[~accounting.index.isin(adata.obs_names)].copy()


__all__ = [
    "BORDERLINE_NONMULTIPLET",
    "MULTIPLET_PLUS_DAMAGE",
    "OTHER",
    "PROBABLE_MULTIPLET_ONLY",
    "QUARANTINE_CAPTURE",
    "QUARANTINE_MULTI_FAMILY",
    "QUARANTINE_OTHER",
    "eligibility_mask_columns",
    "exclusion_category",
    "inference_excluded_table",
    "physically_removed_table",
]
