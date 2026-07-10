"""scVI integration method (GPU-only).

scVI trains a variational model on raw counts and writes a latent embedding.
It requires a GPU in practice; this method self-gates by raising a clear
CellQuorumStageError when no GPU backend is available. Harmony is the
CPU-capable default, so scVI is strictly opt-in via config.
"""

from __future__ import annotations

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumStageError
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod


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
        scvi.model.SCVI.setup_anndata(work, batch_key=batch_key)
        model = scvi.model.SCVI(work, n_latent=n_latent)
        model.train(max_epochs=max_epochs)
        adata.obsm[output_rep] = model.get_latent_representation()

        cq = adata.uns.setdefault("cellquorum", {})
        cq["integration"] = {
            "method": "scvi",
            "batch_key": batch_key,
            "output_rep": output_rep,
            "n_latent": n_latent,
        }
        return StageResult(
            adata=adata,
            metrics={"method": "scvi", "n_latent": n_latent, "output_rep": output_rep},
            notes=[f"scVI latent ({n_latent}d) over '{batch_key}' -> {output_rep}."],
        )


__all__ = ["ScVIMethod"]
