"""Activity-along-pseudotime cascade viz method.

Renders the "working correctly" pseudotime heatmap: per-cell pathway / TF
activity (decoupler) binned along pseudotime into a center-of-mass-ordered
``RdBu_r`` cascade. The enrichment stage only persists *averaged* activity (per
cell type / per sample), so this method scores per-cell activity itself on the
normalized layer within the viz context, then bins it along the trajectory.

Study-agnostic: the nets (PROGENy / CollecTRI / Hallmark / DoRothEA / ...), the
decoupler method per net, the label prettifier, and the normalized layer all
come from config — no biology is hardcoded. Skip-not-crash throughout: a
missing pseudotime, absent decoupler, an un-fetchable net, or a plotting
failure records a clean skip instead of aborting sibling methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.comparative.enrichment.priors import PriorFetchError, get_net
from cellquorum.stages.trajectory.viz import _helpers
from cellquorum.visualization.figstyle import apply_theme

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

# Weighted nets (signed regulons / pathway weights) score better with the
# multivariate linear model; unweighted gene sets with the univariate one.
# Config `activity_methods` overrides per resource.
_DEFAULT_METHOD_BY_RESOURCE = {"progeny": "mlm", "dorothea": "ulm", "collectri": "ulm"}
# How many top-|rho| sources to show per net. Small collections (PROGENy) show
# all (None). Config `cascade_top` overrides per resource.
_DEFAULT_TOP_BY_RESOURCE: dict[str, int | None] = {
    "progeny": None,
    "hallmark": 10,
    "collectri": 20,
    "dorothea": 20,
}
# Proper-name display labels for the well-known decoupler collections (a naming
# convenience only, mirroring priors.py — no study biology). Unknown resources
# fall back to their verbatim key.
_RESOURCE_DISPLAY = {
    "progeny": "PROGENy pathway",
    "collectri": "CollecTRI TF",
    "dorothea": "DoRothEA TF",
    "hallmark": "Hallmark",
    "reactome": "Reactome",
    "kegg": "KEGG",
}


def _cascade_title(resource: str, pt_key: str) -> str:
    """Publication title: display resource name + humanized pseudotime axis."""
    disp = _RESOURCE_DISPLAY.get(str(resource).lower(), str(resource))
    pt = pt_key.replace("_pseudotime", "").replace("_", " ").strip() or "pseudotime"
    return f"{disp} activity along {pt} pseudotime"


class ActivityCascadeVizMethod(AnalysisMethod):
    """Per-cell activity-along-pseudotime cascade heatmap (one figure per net)."""

    name = "activity_cascade"
    stage_category = "trajectory_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        pts = _helpers.available_pseudotimes(adata, config.get("pseudotime_keys"))
        if not pts:
            return self._skip("no *_pseudotime obs column")
        pt_key = pts[0]

        try:
            import decoupler as dc
        except Exception as exc:  # noqa: BLE001
            return self._skip("decoupler unavailable", error=str(exc)[:300])
        if dc is None:
            return self._skip("decoupler unavailable")

        # `.get(key, default)` only defaults an *absent* key; the stage augments
        # config with these keys set to None (their TrajectoryVizConfig default),
        # so fall back with `or` to treat an explicit None as "use the default".
        resources = config.get("activity_resources") or ["progeny", "collectri"]
        method_by_res = {**_DEFAULT_METHOD_BY_RESOURCE, **(config.get("activity_methods") or {})}
        top_by_res = {**_DEFAULT_TOP_BY_RESOURCE, **(config.get("cascade_top") or {})}
        layer = config.get("layer") or "cellquorum_normalized"
        organism = config.get("organism") or "human"
        license = config.get("license") or "academic"
        gmt_path = config.get("gmt_path")
        min_size = int(config.get("min_size") or 5)
        n_bins = int(config.get("cascade_n_bins") or 20)
        xlab = config.get("cascade_xlab") or ""

        # Passed to decoupler as AnnData + layer name, not a dense pandas DataFrame
        # built from the FULL expression matrix: decoupler's AnnData-input path
        # handles a sparse layer natively (internal batching via bsize), so manually
        # densifying here would be the same "12 GB in a minute, VM died" failure
        # mode already fixed twice elsewhere (state_scoring/aucell_method.py,
        # comparative/enrichment/activity_method.py) -- a real cohort's full matrix
        # (201,871 x 33,417) densified is ~54 GB.
        activity_layer = layer if layer != "X" and layer in adata.layers else None
        pt_full = pd.to_numeric(adata.obs[pt_key], errors="coerce")

        from cellquorum.stages.trajectory.viz import activity_cascade as cascade

        apply_theme("notebook")
        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))

        artifacts, warnings, skipped, done = [], [], [], []
        for resource in resources:
            try:
                net = get_net(resource, organism=organism, gmt_path=gmt_path, license=license)
            except PriorFetchError as exc:
                skipped.append({"resource": resource, "reason": str(exc)[:300]})
                continue

            method = method_by_res.get(str(resource).lower(), "ulm")
            top = top_by_res.get(str(resource).lower(), 20)
            try:
                # decoupler mutates adata.obsm[f"score_{method}"] in place and
                # returns None when no observations were dropped; when it drops
                # all-zero observations it instead returns a NEW, smaller AnnData
                # carrying the scores and leaves the original untouched (verified
                # empirically, same behavior across aucell/ulm/mlm).
                result = getattr(dc.mt, method)(adata, net, tmin=min_size, layer=activity_layer)
                source = adata if result is None else result
                es = source.obsm.pop(f"score_{method}")
                source.obsm.pop(f"padj_{method}", None)
                # decoupler may drop all-zero cells — realign pseudotime to the
                # returned (possibly shortened) index before binning.
                pt = pt_full.reindex(es.index).to_numpy(dtype=float)
                fig = cascade.cascade_heatmap(
                    es,
                    pt,
                    top=top,
                    n_bins=n_bins,
                    title=_cascade_title(resource, pt_key),
                    xlab=xlab,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"activity_cascade: {resource} failed: {str(exc)[:200]}")
                continue
            if fig is None:
                skipped.append(
                    {"resource": resource, "reason": "no source associated with pseudotime"}
                )
                continue

            paths = _helpers.save_figure(
                fig, figures_dir, f"activity_cascade_{resource}", formats=formats, dpi=dpi
            )
            artifacts += _helpers.figure_artifacts(
                paths,
                name="trajectory_figure",
                description=f"{resource} activity cascade along {pt_key} ({method}).",
            )
            done.append(resource)

        if not done:
            return self._skip(
                "no activity net produced a cascade", skipped=skipped, warnings=warnings
            )
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"activity_cascade rendered {done} along {pt_key}."],
            warnings=warnings,
            metrics={
                "method": self.name,
                "pseudotime_key": pt_key,
                "resources": done,
                "skipped": skipped,
            },
            backend="python",
        )


__all__ = ["ActivityCascadeVizMethod"]
