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

from cellquorum.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumStageError
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod


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
        work = adata.copy()
        work.X = work.layers["counts"]
        work.obs["_scanvi_labels"] = work.obs[label_key].astype(str)

        # Train the unsupervised scVI base model.
        scvi.model.SCVI.setup_anndata(work, batch_key=batch_key)
        vae = scvi.model.SCVI(work, n_latent=n_latent)
        vae.train(max_epochs=max_epochs)

        # Train scANVI from the scVI model using the partial labels.
        scanvi = scvi.model.SCANVI.from_scvi_model(
            vae,
            unlabeled_category=unlabeled_category,
            labels_key="_scanvi_labels",
        )
        scanvi.train(max_epochs=max_epochs)

        adata.obsm[output_rep] = scanvi.get_latent_representation()

        cq = adata.uns.setdefault("cellquorum", {})
        cq["integration"] = {
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
                f"conditioned on '{label_key}' -> {output_rep}."
            ],
        )


__all__ = ["ScANVIMethod"]
