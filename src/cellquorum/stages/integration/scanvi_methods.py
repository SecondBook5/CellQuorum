"""scANVI integration method (GPU-only, semi-supervised).

scANVI extends scVI with cell-type labels: it trains scVI on raw counts, then a
scANVI model that uses partial labels (``label_key``, with unlabeled cells marked
``unlabeled_category``) to produce a batch-corrected latent that better preserves
biological identity. Like scVI it is GPU-oriented and self-gates when no GPU
backend is available. Harmony remains the CPU-capable default, so scANVI is
strictly opt-in via config.
"""

from __future__ import annotations

import anndata as ad

from cellquorum.core.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumStageError
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.stages.integration._embedding_collapse import check_embedding_collapse
from cellquorum.stages.integration._fit_population import resolve_training_set


class ScANVIMethod(AnalysisMethod):
    """scANVI semi-supervised latent-space integration strategy (GPU-only, opt-in)."""

    # Registry identity.
    name = "scanvi"
    stage_category = "integration"
    backend = "gpu"

    def requires_layers(self) -> list[str]:
        """scANVI trains on raw counts."""

        return ["counts"]

    def requires_obs(self, config: dict) -> list[str]:
        """Require the batch column and the label column for semi-supervision."""

        batch_key = config.get("batch_key", "patient_id")
        label_key = config.get("label_key")
        required = [batch_key]
        if label_key:
            required.append(label_key)
        return required

    def input_contract(self, config: dict) -> DataContract:
        """Require the counts layer, the batch column, and the label column."""

        batch_key = config.get("batch_key", "patient_id")
        label_key = config.get("label_key")
        required_obs = [batch_key] + ([label_key] if label_key else [])
        return DataContract(required_layers=["counts"], required_obs=required_obs)

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """
        Train scVI then scANVI and write the batch-corrected latent embedding.

        Raises:
            CellQuorumStageError: If no GPU backend is available or no label
                column is configured (scANVI is semi-supervised and needs labels).
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
                "scANVI integration requires a GPU backend, which is unavailable. "
                "Use method='harmony' for CPU integration.",
            )

        # scANVI is semi-supervised: it needs a label column to condition on.
        label_key = config.get("label_key")
        if not label_key:
            raise CellQuorumStageError(
                "integration",
                "scANVI integration requires 'label_key' (a cell-type column). "
                "Set integration.label_key, or use method='scvi' for unsupervised "
                "latent integration.",
            )
        if label_key not in adata.obs.columns:
            raise CellQuorumStageError(
                "integration",
                f"scANVI label_key '{label_key}' is not present in adata.obs.",
            )

        # Import scvi lazily (heavy) and train.
        import scvi

        batch_key = config.get("batch_key", "patient_id")
        n_latent = int(config.get("n_latent", 30))
        # scANVI writes a latent space, not a Harmony-corrected PCA.
        output_rep = config.get("output_rep", "X_scanvi")
        max_epochs = config.get("max_epochs", None)
        unlabeled_category = config.get("unlabeled_category", "Unknown")
        random_state = int(config.get("random_state", 0))

        scvi.settings.seed = random_state
        # A MINIMAL object: one matrix, plus obs and var.
        #
        # `adata.copy()` duplicated every layer and every obsm to change which matrix is .X --
        # and nothing here reads `work.layers` afterwards. On this cohort that is counts AND
        # cellquorum_normalized AND the denoised layer AND four embeddings copied so that one of
        # them could be assigned to X. obs and var are carried whole because the training-set
        # split, the batch key and the HVG mask all live there, and DataFrames are cheap beside
        # the matrices. obs is copied because a label column is written into it below.
        work = ad.AnnData(
            X=adata.layers["counts"],
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )
        work.obs["_scanvi_labels"] = work.obs[label_key].astype(str)

        # As with scVI, the encoder is a function, so fit_scope=CORE is honourable: train on
        # the cells QC permits, encode everyone. scANVI conditions on labels as well as batch,
        # so both are checked for coverage before the split is taken.
        train, scope_note = resolve_training_set(
            work, conditioning_keys=[batch_key, "_scanvi_labels"]
        )

        # Train the unsupervised scVI base model.
        scvi.model.SCVI.setup_anndata(train, batch_key=batch_key)
        vae = scvi.model.SCVI(train, n_latent=n_latent)
        vae.train(max_epochs=max_epochs)

        # Train scANVI from the scVI model using the partial labels.
        scanvi = scvi.model.SCANVI.from_scvi_model(
            vae,
            unlabeled_category=unlabeled_category,
            labels_key="_scanvi_labels",
        )
        scanvi.train(max_epochs=max_epochs)

        latent = (
            scanvi.get_latent_representation()
            if train is work
            else scanvi.get_latent_representation(work)
        )

        # Same degenerate-embedding check as scVI, shared so the two cannot drift.
        collapse_warning = check_embedding_collapse(latent, n_latent, method_name="scANVI")

        adata.obsm[output_rep] = latent

        cq = adata.uns.setdefault("cellquorum", {})
        # Single-method provenance (backward-compatible path, last-wins).
        cq["integration"] = {
            "method": "scanvi",
            "batch_key": batch_key,
            "label_key": label_key,
            "output_rep": output_rep,
            "n_latent": n_latent,
        }
        # Per-method provenance (multi-method path, namespaced by output_rep).
        cq.setdefault("integration_methods", {})[output_rep] = {
            "method": "scanvi",
            "batch_key": batch_key,
            "label_key": label_key,
            "output_rep": output_rep,
            "n_latent": n_latent,
        }
        return StageResult(
            adata=adata,
            metrics={
                "method": "scanvi",
                "n_latent": n_latent,
                "output_rep": output_rep,
                "label_key": label_key,
            },
            notes=[
                f"scANVI latent ({n_latent}d) over '{batch_key}' "
                f"conditioned on '{label_key}' -> {output_rep}.",
                *([scope_note] if scope_note else []),
            ],
            warnings=[collapse_warning] if collapse_warning else [],
        )


__all__ = ["ScANVIMethod"]
