"""CellTypist reference-model annotation.

CellTypist's built-in models are trained on log1p(CP10k) at target_sum 1e4. Our
cellquorum_normalized layer is PFlog1pPF (centered), which would give wrong labels,
so this method builds CP10k-log FROM THE COUNTS LAYER on a working copy and runs
CellTypist on that. The input contract asserts the counts layer really is counts.
"""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class CellTypistMethod(AnalysisMethod):
    """CellTypist label-transfer annotation strategy."""

    name = "celltypist"
    stage_category = "annotation"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """Require a real counts layer (guards against PFlog1pPF being passed)."""

        counts_layer = config.get("counts_layer", "counts")
        return DataContract(
            required_layers=[counts_layer],
            expression_layer=counts_layer,
            expected_kind="counts",
        )

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        """Normalize counts to CP10k-log, run CellTypist, write labels."""

        counts_layer = config.get("counts_layer", "counts")
        key_added = config.get("key_added", "cell_type")
        model = config.get("model", None)
        majority_voting = bool(config.get("majority_voting", True))

        # No model configured -> skip (a model is a required asset, not a default).
        if not model:
            return MethodSkip(
                reason="celltypist skipped: no model configured (annotation.model)",
                details={"method": self.name},
            )

        # Import + load the model lazily; a missing package/model is a graceful skip.
        try:
            import celltypist
            from celltypist import annotate
        except Exception as exc:  # noqa: BLE001
            return MethodSkip(
                reason=f"celltypist skipped: import failed ({type(exc).__name__})",
                details={"method": self.name, "error": str(exc)[:120]},
            )
        try:
            # Resolve the model (name -> downloaded/cached, or a filesystem path).
            loaded_model = celltypist.models.Model.load(model)
        except Exception as exc:  # noqa: BLE001
            return MethodSkip(
                reason=f"celltypist skipped: model '{model}' unavailable",
                details={"method": self.name, "error": str(exc)[:120]},
            )

        # Build the CP10k-log space CellTypist expects, FROM COUNTS, on a copy.
        work = adata.copy()
        work.X = work.layers[counts_layer].copy()
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)

        # Annotate; majority-voting refines labels over CellTypist's over-clustering.
        predictions = annotate(work, model=loaded_model, majority_voting=majority_voting)
        labels_df = predictions.predicted_labels
        label_col = (
            "majority_voting"
            if (majority_voting and "majority_voting" in labels_df)
            else "predicted_labels"
        )

        # Write labels + a confidence column onto the REAL object.
        adata.obs[key_added] = labels_df[label_col].to_numpy()
        adata.obs[key_added] = adata.obs[key_added].astype("category")
        try:
            conf = predictions.probability_matrix.max(axis=1).to_numpy()
            adata.obs[f"{key_added}_conf"] = conf
        except Exception:  # noqa: BLE001
            pass

        n_types = int(adata.obs[key_added].nunique())
        return StageResult(
            adata=adata,
            metrics={
                "method": "celltypist",
                "model": str(model),
                "n_types": n_types,
                "key_added": key_added,
                "majority_voting": majority_voting,
            },
            notes=[f"celltypist ({model}) assigned {n_types} cell types -> {key_added}."],
        )


__all__ = ["CellTypistMethod"]
