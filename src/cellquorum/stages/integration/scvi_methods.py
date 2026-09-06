"""scVI integration method (GPU-only).

scVI trains a variational model on raw counts and writes a latent embedding.
It requires a GPU in practice; this method self-gates by raising a clear
CellQuorumStageError when no GPU backend is available. Harmony is the
CPU-capable default, so scVI is strictly opt-in via config.
"""

from __future__ import annotations

from typing import Any

import anndata as ad

from cellquorum.core.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumDataError, CellQuorumStageError
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.stages.integration._fit_population import resolve_training_set

#: Below this many highly variable genes a latent space is not worth fitting on them;
#: fall back to every gene and say so, rather than training on a handful.
_MIN_HVG_FOR_SCVI = 500


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
        work: The counts-backed object the model was set up against.
        model: A trained ``scvi.model.SCVI``.
        layer: Destination layer name.
        genes: Genes to decode. Empty is refused.
        library_size: Scaling for the decoded values; ``1e4`` gives CP10K-like units and
            ``"latent"`` uses each cell's inferred size factor.

    Returns:
        Notes for the stage result.

    Raises:
        CellQuorumDataError: If no genes were named, or none of them are in the object.
    """
    import numpy as np

    from cellquorum.core.contracts.layer_tags import set_layer_tag

    if not genes:
        raise CellQuorumDataError(
            "integration.denoised_layer was requested without denoised_genes. Decoded "
            "expression is dense, so every gene on a cohort this size would be tens of "
            "gigabytes; name the genes the figures need."
        )

    present = [gene for gene in dict.fromkeys(genes) if gene in adata.var_names]
    missing = [gene for gene in dict.fromkeys(genes) if gene not in adata.var_names]
    if not present:
        raise CellQuorumDataError(
            f"None of the {len(genes)} requested denoised_genes are in this object. "
            f"First few asked for: {genes[:5]}"
        )

    decoded = model.get_normalized_expression(
        work, gene_list=present, library_size=library_size, return_mean=True
    )

    values = np.zeros((adata.n_obs, adata.n_vars), dtype="float32")
    positions = [adata.var_names.get_loc(gene) for gene in present]
    values[:, positions] = np.asarray(decoded, dtype="float32")
    adata.layers[layer] = values
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
    if missing:
        notes.append(f"{layer}: {len(missing)} requested gene(s) absent: {missing[:5]}")
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
        # Default True: training on all genes is the wrong default for scVI, and a caller who
        # genuinely wants every gene can say so.
        use_hvg = bool(config.get("use_highly_variable", True))

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
        # `var['highly_variable']` and deliberately does not subset, so each consumer opts in.
        # Absent the flag, every gene is used and a note says so — silently training on all
        # genes is the behaviour being fixed, so it must not be the silent default.
        if use_hvg:
            if "highly_variable" in work.var.columns:
                mask = work.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
                if int(mask.sum()) >= _MIN_HVG_FOR_SCVI:
                    work = work[:, mask].copy()
                    hvg_note = f"scVI trained on {int(mask.sum()):,} highly variable genes."
                else:
                    hvg_note = (
                        f"scVI: only {int(mask.sum())} highly variable genes flagged, below the "
                        f"{_MIN_HVG_FOR_SCVI} needed for a usable latent space — using all "
                        f"{work.n_vars:,} genes instead."
                    )
            else:
                hvg_note = (
                    "scVI: use_highly_variable is set but var['highly_variable'] is absent — "
                    "enable the feature_selection stage. Using all "
                    f"{work.n_vars:,} genes."
                )
        else:
            hvg_note = (
                f"scVI trained on all {work.n_vars:,} genes (use_highly_variable not set). "
                f"Standard practice is 2,000-5,000 highly variable genes."
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
