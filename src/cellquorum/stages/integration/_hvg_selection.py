"""Shared HVG-restriction logic for the VAE-based integration methods (scVI, scANVI).

Both methods train a neural encoder/decoder whose parameter count scales with the gene
count. Left unrestricted, they train on every gene in the object -- for a typical cohort
that is an order of magnitude more genes than the usual 2,000-5,000-gene setup, and the
uninformative genes contribute reconstruction loss without contributing structure. The
latent space that comes out is worse, and that latent space is what clustering uses and
what scArches builds its query model on: a degraded embedding propagates all the way into
the cell-type calls.

This honours `feature_selection` rather than selecting genes itself: that stage flags
`var['highly_variable']` and deliberately does not subset, so consumption happens here.
An explicit `use_highly_variable: true` with no flag present is an error rather than a
note: a warning in a long training run scrolls past, and the resulting latent space is
perfectly usable-looking while being built from the wrong genes.

Extracted so scVI and scANVI cannot drift on the threshold, the messages, or whether the
check happens at all -- which is exactly how scANVI ended up with no restriction at all
while scVI had a carefully documented one.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.core.exceptions import CellQuorumStageError

#: Below this many highly variable genes a latent space is not worth fitting on them;
#: fall back to every gene and say so, rather than training on a handful.
_MIN_HVG_FOR_TRAINING = 500


def _add_decodable_genes(
    work: ad.AnnData,
    mask: np.ndarray,
    extra_genes: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Extend an HVG training mask to cover genes a caller needs kept regardless.

    scVI's decoder can only reconstruct genes the model saw during training, so a marker that
    missed the HVG cut is not merely less accurate -- it is unavailable. The markers that miss
    are systematically the broadly expressed lineage genes (high mean, low relative variance).

    Args:
        work: The object about to be subset; supplies ``var_names``.
        mask: Boolean HVG mask over ``work.var_names``. Not mutated.
        extra_genes: Genes the caller wants kept in the training set. Absent names are
            ignored here; the caller reports them.

    Returns:
        ``(mask, added)`` -- the extended mask, and the genes it gained in request order.
    """
    extended = mask.copy()
    if not extra_genes:
        return extended, []

    added: list[str] = []
    for gene in dict.fromkeys(extra_genes):
        if gene not in work.var_names:
            continue
        index = work.var_names.get_loc(gene)
        if not extended[index]:
            extended[index] = True
            added.append(gene)
    return extended, added


def restrict_to_highly_variable(
    work: ad.AnnData,
    config: dict,
    *,
    method_name: str,
    min_hvg: int = _MIN_HVG_FOR_TRAINING,
    extra_required_genes: list[str] | None = None,
) -> tuple[ad.AnnData, str]:
    """Subset ``work`` to highly variable genes per `feature_selection` and config.

    ``None`` (the default for ``config['use_highly_variable']``) follows the
    feature-selection stage: use the HVGs when they were flagged, all genes when they
    were not. Restating the decision at the call site is what let a stage-enabled config
    train on every gene anyway.

    Args:
        work: The training-candidate object (already counts-only; not mutated in place --
            a new, possibly gene-subset object is returned).
        config: The method's config dict; reads ``use_highly_variable``.
        method_name: "scVI" or "scANVI", for messages.
        min_hvg: Below this many flagged genes, fall back to all genes.
        extra_required_genes: Genes to keep in the training set even if not flagged
            highly variable (e.g. scVI's ``denoised_genes``). Ignored when empty/None.

    Returns:
        ``(work, note)`` -- the (possibly subset) object, and a note describing what
        happened, for the caller's ``StageResult.notes``.

    Raises:
        CellQuorumStageError: If ``use_highly_variable`` is explicitly requested but
            ``var['highly_variable']`` is absent.
    """
    hvg_setting = config.get("use_highly_variable")
    has_hvg_flag = "highly_variable" in work.var.columns

    if hvg_setting is None:
        use_hvg = has_hvg_flag
    else:
        use_hvg = bool(hvg_setting)
        if use_hvg and not has_hvg_flag:
            raise CellQuorumStageError(
                "integration",
                f"{method_name}: use_highly_variable is set but var['highly_variable'] is "
                f"absent, so training would silently use every gene. Enable the "
                f"feature-selection stage (`stages.feature_selection: true`), or set "
                f"`integration.use_highly_variable: false` to use all genes on purpose.",
            )

    if not use_hvg:
        note = (
            f"{method_name} trained on all {work.n_vars:,} genes: the feature-selection "
            f"stage flagged no highly variable genes. Standard practice is 2,000-5,000."
            if hvg_setting is None
            else f"{method_name} trained on all {work.n_vars:,} genes (use_highly_variable=false)."
        )
        return work, note

    mask = work.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
    if int(mask.sum()) < min_hvg:
        return work, (
            f"{method_name}: only {int(mask.sum())} highly variable genes flagged, below "
            f"the {min_hvg} needed for a usable latent space -- using all {work.n_vars:,} "
            f"genes instead."
        )

    n_hvg = int(mask.sum())
    mask, added = _add_decodable_genes(work, mask, extra_required_genes or [])
    work = work[:, mask].copy()
    note = f"{method_name} trained on {n_hvg:,} highly variable genes."
    if added:
        note += (
            f" Plus {len(added)} non-variable gene(s) required by the caller so they stay "
            f"decodable: {', '.join(added)}."
        )
    return work, note


__all__ = ["restrict_to_highly_variable"]
