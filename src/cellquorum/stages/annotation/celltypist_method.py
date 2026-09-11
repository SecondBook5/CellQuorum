"""CellTypist reference-model annotation.

CellTypist's built-in models are trained on log1p(CP10k) at target_sum 1e4. Our
cellquorum_normalized layer is PFlog1pPF (centered), which would give wrong labels,
so this method builds CP10k-log FROM THE COUNTS LAYER on a working copy and runs
CellTypist on that. The input contract asserts the counts layer really is counts.
"""

from __future__ import annotations

import anndata as ad
import pandas as pd
import scanpy as sc

from cellquorum.core.contracts import DataContract
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
            return self._skip("no model configured (annotation.model)")

        # Import + load the model lazily; a missing package/model is a graceful skip.
        try:
            import celltypist
            from celltypist import annotate
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"import failed ({type(exc).__name__})", error=str(exc)[:120])
        try:
            # Resolve the model (name -> downloaded/cached, or a filesystem path).
            loaded_model = celltypist.models.Model.load(model)
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"model '{model}' unavailable", error=str(exc)[:120])

        # Build the CP10k-log space CellTypist expects, FROM COUNTS, on a MINIMAL object.
        #
        # The counts matrix genuinely has to be copied -- normalize_total and log1p write into it
        # -- but the rest of the cohort does not. `adata.copy()` duplicated both layers, every
        # obsm and ~90 obs columns as well, on a 201,871 x 33,417 object.
        #
        # The NEIGHBOUR GRAPH is carried deliberately. With `majority_voting`, CellTypist reuses
        # an existing graph if it finds one ("Detected a neighborhood graph in the input object,
        # will run over-clustering on the basis of it") and otherwise computes its own. Dropping
        # it would not just be slower, it would change the over-clustering and therefore the
        # labels -- so the memory saving must not extend to it. It is sparse: for 201,871 cells at
        # 15 neighbours, a few hundred MB against the several GB the full copy cost.
        work = ad.AnnData(
            X=adata.layers[counts_layer].copy(),
            obs=pd.DataFrame(index=adata.obs_names),
            var=pd.DataFrame(index=adata.var_names),
            obsp={key: adata.obsp[key] for key in adata.obsp},
            uns={"neighbors": adata.uns["neighbors"]} if "neighbors" in adata.uns else {},
        )
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

        # Write labels + a confidence column onto the REAL object, realigned by cell
        # name rather than trusted-by-row-position: celltypist currently preserves
        # input order (verified empirically, including under majority_voting), but
        # nothing here should silently depend on that never changing -- a reordered
        # result would otherwise scramble every cell's label with no error at all.
        adata.obs[key_added] = labels_df[label_col].reindex(adata.obs_names).to_numpy()
        adata.obs[key_added] = adata.obs[key_added].astype("category")
        try:
            conf = predictions.probability_matrix.max(axis=1).reindex(adata.obs_names).to_numpy()
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
