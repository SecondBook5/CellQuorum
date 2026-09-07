"""scVI integration method (GPU-only).

scVI trains a variational model on raw counts and writes a latent embedding.
It requires a GPU in practice; this method self-gates by raising a clear
CellQuorumStageError when no GPU backend is available. Harmony is the
CPU-capable default, so scVI is strictly opt-in via config.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellquorum.core.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumDataError, CellQuorumStageError
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.stages.integration._fit_population import resolve_training_set

#: Below this many highly variable genes a latent space is not worth fitting on them;
#: fall back to every gene and say so, rather than training on a handful.
_MIN_HVG_FOR_SCVI = 500


def _add_decodable_genes(
    work: ad.AnnData,
    mask: np.ndarray,
    denoised_genes: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Extend an HVG training mask to cover the genes the denoised panel asks for.

    scVI's decoder can only reconstruct genes the model saw during training, so a marker that
    missed the HVG cut is not merely less accurate — it is unavailable. The markers that miss
    are systematically the broadly expressed lineage genes (high mean, low relative variance),
    which is exactly what a panel like ``PECAM1``/``COL1A1`` is made of.

    Args:
        work: The object about to be subset; supplies ``var_names``.
        mask: Boolean HVG mask over ``work.var_names``. Not mutated.
        denoised_genes: Genes the caller wants decoded later. Absent names are ignored here;
            :func:`_write_denoised_layer` reports them.

    Returns:
        ``(mask, added)`` — the extended mask, and the genes it gained in request order.
    """
    extended = mask.copy()
    if not denoised_genes:
        return extended, []

    added: list[str] = []
    for gene in dict.fromkeys(denoised_genes):
        if gene not in work.var_names:
            continue
        index = work.var_names.get_loc(gene)
        if not extended[index]:
            extended[index] = True
            added.append(gene)
    return extended, added


def _write_denoised_layer(
    adata: ad.AnnData,
    work: ad.AnnData,
    *,
    model: Any,
    layer: str,
    genes: list[str],
    library_size: float | str,
) -> list[str]:
    """Write scVI's decoded expression as an ``imputed``-tagged layer.

    scVI models dropout explicitly through its negative-binomial likelihood, so the decoder's
    expected expression is a denoised estimate: a marker that reads as scattered noise on a
    UMAP because 80% of its counts are zero reads as a coherent domain once decoded.

    **Tagged ``imputed``, deliberately.** These are model output, not measurements. The model
    has already borrowed information across cells, so the effective sample size is not the cell
    count and any test computed on them is anticonservative — a p-value from denoised expression
    is not a p-value. Tagging routes them through
    :func:`cellquorum.core.contracts.magic_guard.assert_not_imputed`, which is what stops them
    reaching differential expression or abundance testing. Figures and scoring opt in knowingly.

    **A gene list is required.** The decoded matrix is dense, so asking for every gene on this
    cohort would be 202,000 x 33,000 float32 ~ 26 GB. Naming the genes matches the real use —
    a handful of markers for a panel — and refuses the request that would exhaust memory rather
    than discovering it after the model has trained.

    The layer is filled only for the named genes; every other gene stays zero, which the tag and
    the recorded gene list make legible.

    Args:
        adata: The full object the layer is written onto.
        work: The counts-backed object the model was set up against — possibly gene-subset,
            which bounds what the decoder can reconstruct.
        model: A trained ``scvi.model.SCVI``.
        layer: Destination layer name.
        genes: Genes to decode. Empty is refused.
        library_size: Scaling for the decoded values; ``1e4`` gives CP10K-like units and
            ``"latent"`` uses each cell's inferred size factor.

    Returns:
        Notes for the stage result.

    Raises:
        CellQuorumDataError: If no genes were named, or none of them can be decoded — absent
            from the object, or outside the gene set the model trained on.
    """
    from cellquorum.core.contracts.layer_tags import set_layer_tag

    if not genes:
        raise CellQuorumDataError(
            "integration.denoised_layer was requested without denoised_genes. Decoded "
            "expression is dense, so every gene on a cohort this size would be tens of "
            "gigabytes; name the genes the figures need."
        )

    # Decodable means present in the object AND in the gene set the MODEL was trained on.
    # Testing only `adata.var_names` is what broke this: with scVI restricted to 2,000 HVGs,
    # all 20 requested markers looked present, the decoder returned the 17 it actually knew,
    # and the write failed with "value array of shape (201871,17) could not be broadcast to
    # indexing result of shape (201871,20)" — after six minutes of GPU training.
    requested = list(dict.fromkeys(genes))
    present = [gene for gene in requested if gene in adata.var_names and gene in work.var_names]
    absent = [gene for gene in requested if gene not in adata.var_names]
    unmodelled = [
        gene for gene in requested if gene in adata.var_names and gene not in work.var_names
    ]
    if not present:
        raise CellQuorumDataError(
            f"None of the {len(genes)} requested denoised_genes can be decoded: "
            f"{len(absent)} absent from the object, {len(unmodelled)} present but outside "
            f"the {work.n_vars:,} genes scVI was trained on. First few asked for: {genes[:5]}"
        )

    decoded = model.get_normalized_expression(
        work, gene_list=present, library_size=library_size, return_mean=True
    )

    # SPARSE, because the layer spans the full gene space while only the named genes carry
    # values. Densely, this cohort's 201,871 x 33,417 float32 is 27 GB to store 20 genes of
    # information — which inflated the post-annotation checkpoint to 41.6 GB and left the
    # process holding ~35 GB before reference_mapping tried to load a 24.8 GB atlas on top of
    # it, on a 54 GB machine. As CSC the same content is about 16 MB, and column slicing (how
    # every figure reads a marker) stays cheap.
    positions = [adata.var_names.get_loc(gene) for gene in present]
    dense_block = np.asarray(decoded, dtype="float32")
    if dense_block.ndim == 1:
        dense_block = dense_block.reshape(-1, 1)

    # Built straight into COO then CSC: one pass, no per-column assignment into a LIL, which
    # at 201,871 rows is slow enough to look like a hang.
    n_genes = dense_block.shape[1]
    rows = np.tile(np.arange(adata.n_obs, dtype=np.int32), n_genes)
    cols = np.repeat(np.asarray(positions, dtype=np.int32), adata.n_obs)
    adata.layers[layer] = sp.coo_matrix(
        (dense_block.ravel(order="F"), (rows, cols)),
        shape=(adata.n_obs, adata.n_vars),
        dtype="float32",
    ).tocsc()
    set_layer_tag(adata, layer, kind="imputed", recipe="scvi_decoder")

    adata.uns.setdefault("cellquorum", {}).setdefault("denoised_layers", {})[layer] = {
        "recipe": "scvi_decoder",
        "genes": present,
        "library_size": library_size,
    }

    notes = [
        f"{layer}: scVI-decoded expression for {len(present)} gene(s), tagged imputed — "
        f"blocked from inference by contract, available to figures."
    ]
    if absent:
        notes.append(f"{layer}: {len(absent)} requested gene(s) absent from the object: {absent}")
    if unmodelled:
        # Should not happen once the training set unions in `denoised_genes`, so if it does
        # the panel is incomplete for a reason worth naming rather than a silent blank row.
        notes.append(
            f"{layer}: {len(unmodelled)} requested gene(s) are in the object but were not "
            f"among the genes scVI trained on, so the decoder cannot reconstruct them and "
            f"they stay zero: {unmodelled}"
        )
    return notes


class ScVIMethod(AnalysisMethod):
    """scVI latent-space integration strategy (GPU-only, opt-in)."""

    # Registry identity.
    name = "scvi"
    stage_category = "integration"
    backend = "gpu"

    def requires_layers(self) -> list[str]:
        """scVI trains on raw counts."""

        # Require a counts layer.
        return ["counts"]

    def requires_obs(self, config: dict) -> list[str]:
        """Return the batch key that must exist for integration to run."""

        # Read the batch column from config.
        batch_key = config.get("batch_key", "patient_id")

        # Require the batch column to exist.
        return [batch_key]

    def input_contract(self, config: dict) -> DataContract:
        """Require the counts layer and the batch obs column."""

        # Read the batch column from config.
        batch_key = config.get("batch_key", "patient_id")
        return DataContract(required_layers=["counts"], required_obs=[batch_key])

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """
        Train scVI and write the latent embedding.

        Raises:
            CellQuorumStageError: If no GPU backend is available.
        """

        # Self-gate on GPU availability via the backend registry when present.
        registry = getattr(context, "backend_registry", None)
        gpu_ok = False
        if registry is not None and hasattr(registry, "available"):
            try:
                gpu_ok = bool(registry.available("gpu"))
            except Exception:
                gpu_ok = False
        if not gpu_ok:
            raise CellQuorumStageError(
                "integration",
                "scVI integration requires a GPU backend, which is unavailable. "
                "Use method='harmony' for CPU integration.",
            )

        # Import scvi lazily (heavy) and train.
        import scvi

        batch_key = config.get("batch_key", "patient_id")
        n_latent = int(config.get("n_latent", 30))
        # Default to a scVI-specific key: scVI writes a latent space, not a
        # Harmony-corrected PCA, so it must not masquerade under X_pca_harmony.
        output_rep = config.get("output_rep", "X_scvi")
        max_epochs = config.get("max_epochs", None)
        random_state = int(config.get("random_state", 0))
        # Optional denoised-expression output. See `_write_denoised_layer`.
        denoised_layer = config.get("denoised_layer") or None
        denoised_genes = list(config.get("denoised_genes") or [])
        denoised_library_size = config.get("denoised_library_size", 1e4)
        # `None` (the default) follows the feature-selection stage: use the HVGs when they
        # were flagged, all genes when they were not. Restating the decision here is what
        # let `stages.feature_selection: true` train scVI on all ~33,000 genes anyway.
        hvg_setting = config.get("use_highly_variable")

        scvi.settings.seed = random_state
        work = adata.copy()
        work.X = work.layers["counts"]

        # Restrict to highly variable genes before training.
        #
        # This was absent, so scVI trained on every gene — on this cohort ~33,000 of them for
        # 202,000 cells, roughly sixteen times the decoder parameters of the usual 2,000-gene
        # setup. That is not merely slow: uninformative genes contribute reconstruction loss
        # without contributing structure, so the latent space is worse, and this latent space is
        # what clustering uses and what scArches builds its query model on. A degraded embedding
        # therefore propagates all the way into the cell-type calls.
        #
        # Honours `feature_selection` rather than selecting genes itself: that stage flags
        # `var['highly_variable']` and deliberately does not subset, so consumption happens
        # here. An explicit `true` with no flag present is an error rather than a note: a
        # warning in a 40-minute run scrolls past, and the resulting latent space is
        # perfectly usable-looking while being built from the wrong genes.
        has_hvg_flag = "highly_variable" in work.var.columns
        if hvg_setting is None:
            use_hvg = has_hvg_flag
        else:
            use_hvg = bool(hvg_setting)
            if use_hvg and not has_hvg_flag:
                raise CellQuorumStageError(
                    "integration",
                    "scVI: use_highly_variable is set but var['highly_variable'] is absent, "
                    "so training would silently use every gene. Enable the feature-selection "
                    "stage (`stages.feature_selection: true`), or set "
                    "`integration.use_highly_variable: false` to use all genes on purpose.",
                )

        if use_hvg:
            mask = work.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
            if int(mask.sum()) >= _MIN_HVG_FOR_SCVI:
                # Genes requested for the denoised panel are added to the training set even
                # when they are not highly variable. The decoder can only reconstruct genes
                # the model saw, so subsetting to HVGs alone silently drops any marker that
                # missed the cut — and the ones that miss are the broadly expressed lineage
                # markers (high mean, low relative variance) that a panel most needs. Left
                # unhandled it is not even a quiet degradation: the decoder returned 17
                # columns for a 20-gene request and the write failed on a shape mismatch
                # after the model had trained.
                n_hvg = int(mask.sum())
                mask, added = _add_decodable_genes(
                    work, mask, denoised_genes if denoised_layer else []
                )
                work = work[:, mask].copy()
                hvg_note = f"scVI trained on {n_hvg:,} highly variable genes."
                if added:
                    hvg_note += (
                        f" Plus {len(added)} non-variable gene(s) required by "
                        f"`denoised_genes` so the decoder can reconstruct them: "
                        f"{', '.join(added)}."
                    )
            else:
                hvg_note = (
                    f"scVI: only {int(mask.sum())} highly variable genes flagged, below the "
                    f"{_MIN_HVG_FOR_SCVI} needed for a usable latent space — using all "
                    f"{work.n_vars:,} genes instead."
                )
        else:
            hvg_note = (
                f"scVI trained on all {work.n_vars:,} genes: the feature-selection stage "
                f"flagged no highly variable genes. Standard practice is 2,000-5,000."
            )

        # A trained encoder is a function, so scVI can honour fit_scope=CORE where Harmony
        # cannot: train on the cells QC permits, then encode every cell through the trained
        # model. The excluded cells get a real latent coordinate without having shaped the
        # latent space.
        train, scope_note = resolve_training_set(work, conditioning_keys=[batch_key])

        scvi.model.SCVI.setup_anndata(train, batch_key=batch_key)
        model = scvi.model.SCVI(train, n_latent=n_latent)
        model.train(max_epochs=max_epochs)

        # Passing `work` explicitly is the out-of-sample step. When training used every cell
        # the default argument is equivalent, and left alone so the common path is untouched.
        adata.obsm[output_rep] = (
            model.get_latent_representation()
            if train is work
            else model.get_latent_representation(work)
        )

        # Optional: the decoder's expected expression, for the genes the caller named.
        # Collected here and appended below, where the stage's note list is built.
        denoised_notes: list[str] = []
        if denoised_layer is not None:
            denoised_notes.extend(
                _write_denoised_layer(
                    adata,
                    work,
                    model=model,
                    layer=denoised_layer,
                    genes=denoised_genes,
                    library_size=denoised_library_size,
                )
            )

        cq = adata.uns.setdefault("cellquorum", {})
        # Single-method provenance (backward-compatible path, last-wins).
        cq["integration"] = {
            "method": "scvi",
            "batch_key": batch_key,
            "output_rep": output_rep,
            "n_latent": n_latent,
        }
        # Per-method provenance (multi-method path, namespaced by output_rep).
        cq.setdefault("integration_methods", {})[output_rep] = {
            "method": "scvi",
            "batch_key": batch_key,
            "output_rep": output_rep,
            "n_latent": n_latent,
        }
        notes = [f"scVI latent ({n_latent}d) over '{batch_key}' -> {output_rep}.", hvg_note]
        notes.extend(denoised_notes)
        if scope_note:
            notes.append(scope_note)
        return StageResult(
            adata=adata,
            metrics={"method": "scvi", "n_latent": n_latent, "output_rep": output_rep},
            notes=notes,
        )


__all__ = ["ScVIMethod"]
