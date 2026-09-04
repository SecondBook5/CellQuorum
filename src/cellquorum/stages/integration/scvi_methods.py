"""scVI integration method (GPU-only).

scVI trains a variational model on raw counts and writes a latent embedding.
It requires a GPU in practice; this method self-gates by raising a clear
CellQuorumStageError when no GPU backend is available. Harmony is the
CPU-capable default, so scVI is strictly opt-in via config.
"""

from __future__ import annotations

import anndata as ad

from cellquorum.core.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumStageError
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.stages.integration._fit_population import resolve_training_set


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

        scvi.settings.seed = random_state
        work = adata.copy()
        work.X = work.layers["counts"]

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
        notes = [f"scVI latent ({n_latent}d) over '{batch_key}' -> {output_rep}."]
        if scope_note:
            notes.append(scope_note)
        return StageResult(
            adata=adata,
            metrics={"method": "scvi", "n_latent": n_latent, "output_rep": output_rep},
            notes=notes,
        )


__all__ = ["ScVIMethod"]
