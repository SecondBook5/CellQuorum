"""decoupler TF/pathway activity method (ulm over per-cell lognorm data)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.comparative.enrichment.priors import PriorFetchError, get_net
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class ActivityMethod(AnalysisMethod):
    """Per-cell decoupler activity (ulm), aggregated to per-cell-type means."""

    name = "activity"
    stage_category = "enrichment"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        cell_type_col = config.get("cell_type_col", "cell_type")
        layer = config.get("layer", "cellquorum_normalized")
        return DataContract(
            required_obs=[cell_type_col],
            required_layers=[layer] if layer != "X" else [],
            expression_layer=layer,
            expected_kind="lognorm",
        )

    def requires_obs(self, config: dict) -> list[str]:
        return [config.get("cell_type_col", "cell_type")]

    def requires_layers(self) -> list[str]:
        # Base hook is config-less (the scvi pattern); guard on the default
        # lognorm layer name. The contract re-checks the configured layer
        # (and its lognorm tag) before _run.
        return ["cellquorum_normalized"]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        resources = config.get("activity_resources", ["collectri", "progeny"])
        cell_type_col = config.get("cell_type_col", "cell_type")
        layer = config.get("layer", "cellquorum_normalized")
        gmt_path = config.get("gmt_path")
        organism = config.get("organism", "human")
        license = config.get("license", "academic")
        min_size = int(config.get("min_size", 5))

        matrix = adata.layers[layer] if layer != "X" and layer in adata.layers else adata.X
        dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
        data = pd.DataFrame(dense, index=adata.obs_names, columns=adata.var_names)
        # Keep labels as a Series indexed by obs name so we can realign to
        # decoupler's returned (possibly shortened) index — decoupler drops any
        # cell that is all-zero over the net's targets, so a positional label
        # array would mismatch the returned frame's length.
        labels = adata.obs[cell_type_col].astype(str)

        try:
            import decoupler as dc
        except Exception as exc:
            return self._skip("decoupler unavailable", error=str(exc)[:300])
        if dc is None:
            return self._skip("decoupler unavailable")

        writer = StageArtifactWriter.from_context(context)
        artifacts, done, skipped = [], [], []
        for resource in resources:
            try:
                net = get_net(resource, organism=organism, gmt_path=gmt_path, license=license)
            except PriorFetchError as exc:
                skipped.append({"resource": resource, "reason": str(exc)[:300]})
                continue
            # Everything from the ulm call through label-assembly + aggregation is
            # guarded: decoupler may drop all-zero cells (returning a shorter
            # frame), so any residual length/index mismatch degrades to a recorded
            # skip rather than aborting the stage and every sibling method.
            try:
                es, _ = dc.mt.ulm(data, net, tmin=min_size)
                es = es.copy()
                # decoupler preserves obs names on the returned index, so realign
                # the cell-type labels to that (possibly shortened) index instead
                # of assigning the full-length original label array.
                aligned = labels.reindex(es.index)
                if es.shape[0] == 0 or aligned.isna().any():
                    skipped.append(
                        {"resource": resource, "reason": "no cells survived decoupler filtering"}
                    )
                    continue
                es[cell_type_col] = aligned.values
                per_ct = es.groupby(cell_type_col).mean()
            except Exception as exc:
                skipped.append({"resource": resource, "reason": str(exc)[:300]})
                continue

            long = per_ct.reset_index().melt(
                id_vars=cell_type_col, var_name="source", value_name="mean_score"
            )
            long = long.rename(columns={cell_type_col: "cell_type"})
            long = long[["cell_type", "source", "mean_score"]]
            artifacts.append(
                writer.table(
                    long,
                    f"enrichment_activity_{resource}.csv",
                    name="enrichment_results",
                    description=f"decoupler ulm activity ({resource}), per cell type.",
                    index=False,
                )
            )
            done.append(resource)

        if not done:
            return self._skip("no resource produced results", skipped=skipped)

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"Activity (ulm) over {done}."],
            metrics={
                "method": self.name,
                "n_resources": len(done),
                "resources": done,
                "skipped": skipped,
            },
            backend="python",
        )


__all__ = ["ActivityMethod"]
