"""Choosing the cells a latent-variable integration model may train on.

The integration stage declares ``fit_scope=CORE``, and its three methods can honour that to
different degrees:

    scvi, scanvi   a trained encoder is a function, so it can be applied to held-out cells
    harmony        returns corrected coordinates directly; no out-of-sample transform exists

So scVI and scANVI train on the QC fit population and encode everyone, and Harmony records
that it could not. This module holds the shared decision so the two VAE methods cannot drift
on it, and so the one condition that makes the split unsafe is checked in exactly one place.

## The condition

scVI conditions the decoder on batch, and scANVI additionally on labels. Both are categorical
covariates registered at ``setup_anndata`` time. If a category exists in the cohort but not in
the fit population, the model never learns an embedding for it and encoding those cells is
either an error or — worse — silently wrong. A rare batch that happens to be mostly borderline
cells is a realistic way to hit this, so the fit population is used only when it covers every
category of every conditioning variable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import anndata as ad
import pandas as pd

from cellquorum.stages.qc.eligibility import fitting_cells

logger = logging.getLogger(__name__)


def resolve_training_set(
    work: ad.AnnData,
    *,
    conditioning_keys: Sequence[str],
) -> tuple[ad.AnnData, str | None, str | None]:
    """Return the object to train on, and a note or warning describing the outcome.

    Args:
        work: The model's working copy, already carrying counts in ``X``.
        conditioning_keys: obs columns the model conditions on — batch for scVI, batch and
            labels for scANVI. Every category of each must appear in the fit population.

    Returns:
        ``(train, note, warning)``. ``train`` is ``work`` itself when training on every
        cell, or a copy restricted to the fit population. At most one of ``note``/
        ``warning`` is set: ``note`` describes a routine, working outcome (no QC masks
        exist, or the split was applied cleanly) and belongs in ``StageResult.notes``.
        ``warning`` means the fit_scope=CORE contract could not be honoured for a reason
        the algorithm could otherwise have avoided — a conditioning category absent from
        the core — so non-core cells influenced the latent space. This must reach
        ``StageResult.warnings``, not just notes, or it is invisible outside verbose runs.
    """
    fitting = fitting_cells(work.obs)
    if fitting is None:
        return work, None, None

    missing = _categories_absent_from(work.obs, fitting, conditioning_keys)
    if missing:
        warning = (
            f"Integration trained on all cells: the QC fit population is missing "
            f"{_describe(missing)}, which the model conditions on, so held-out cells could "
            f"not be encoded. Non-core cells therefore influenced the latent space."
        )
        logger.warning(warning)
        return work, None, warning

    train = work[fitting.to_numpy(dtype=bool)].copy()
    note = (
        f"Latent space trained on {train.n_obs} QC-permitted cells; "
        f"{work.n_obs - train.n_obs} further cells encoded by the trained model without "
        f"influencing it."
    )
    logger.info(note)
    return train, note, None


def _categories_absent_from(
    obs: pd.DataFrame,
    fitting: pd.Series,
    keys: Sequence[str],
) -> dict[str, list[str]]:
    """Categories present in the cohort but absent from the fit population, per key."""
    absent: dict[str, list[str]] = {}
    for key in keys:
        if key not in obs.columns:
            continue
        values = obs[key].astype(str)
        gap = sorted(set(values) - set(values[fitting.to_numpy(dtype=bool)]))
        if gap:
            absent[key] = gap
    return absent


def _describe(missing: dict[str, list[str]]) -> str:
    """Human-readable summary of the categories that blocked the split."""
    return "; ".join(
        f"{len(values)} {key} value(s) ({', '.join(values[:3])}"
        f"{', ...' if len(values) > 3 else ''})"
        for key, values in sorted(missing.items())
    )


__all__ = ["resolve_training_set"]
