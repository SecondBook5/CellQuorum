"""decoupler TF/pathway activity method (ulm over per-cell lognorm data)."""

from __future__ import annotations

import anndata as ad

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.comparative.enrichment.priors import PriorFetchError, get_net


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

        # Keep labels as a Series indexed by obs name so we can realign to
        # decoupler's returned (possibly shortened) index — decoupler drops any
        # cell that is all-zero over the net's targets, so a positional label
        # array would mismatch the returned frame's length.
        labels = adata.obs[cell_type_col].astype(str)
        ulm_layer = layer if layer != "X" and layer in adata.layers else None

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
            #
            # Passed as AnnData + layer name, not a dense pandas DataFrame built from
            # the FULL expression matrix: decoupler's AnnData-input path handles a
            # sparse layer natively (internal batching via bsize), so manually
            # densifying here would be the same "12 GB in a minute, VM died" failure
            # mode already fixed once in state_scoring/aucell_method.py -- a real
            # cohort's full matrix (201,871 x 33,417) densified is ~54 GB. decoupler
            # mutates adata.obsm["score_ulm"] in place and returns None when no
            # observations were dropped; when it drops all-zero observations it
            # instead returns a NEW, smaller AnnData and leaves the original
            # untouched (verified empirically, same as aucell's behavior).
            try:
                result = dc.mt.ulm(adata, net, tmin=min_size, layer=ulm_layer)
                source = adata if result is None else result
                es = source.obsm.pop("score_ulm")
                source.obsm.pop("padj_ulm", None)
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
