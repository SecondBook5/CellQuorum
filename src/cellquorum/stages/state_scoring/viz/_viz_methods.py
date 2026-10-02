"""State-scoring visualization methods: quantitative companions to state_score_umap."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.state_scoring.viz.plots import state_score_violin
from cellquorum.visualization.figio import figure_artifacts
from cellquorum.visualization.figstyle import LE_RED, NORMAL_BLUE, apply_theme, save_figure

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


class StateScoreViolinMethod(AnalysisMethod):
    """Render one violin+p-value figure per scored program: score by
    cluster, split control vs case, with a Mann-Whitney U p-value per
    cluster -- the number the state_score_umap spatial view doesn't give.
    """

    name = "state_score_violin"
    stage_category = "state_scoring_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        score_source = config.get("score_source", "aucell")
        obsm_key = f"X_state_{score_source}"
        uns_key = f"state_{score_source}"
        if obsm_key not in adata.obsm or uns_key not in adata.uns:
            return self._skip(f"{obsm_key} not present -- run {score_source} first")

        available = list(adata.uns[uns_key].get("programs", []))
        requested = list(config.get("programs") or available)
        requested = [p for p in requested if p in available]
        if not requested:
            return self._skip("no requested programs available in scored output")

        cluster_key = config.get("cluster_key") or config.get("cell_type_col", "cell_type")
        if cluster_key not in adata.obs.columns:
            return self._skip(f"obs column '{cluster_key}' not found")

        design = getattr(getattr(context, "config", None), "design", None)
        condition_col = (
            config.get("condition_col") or getattr(design, "condition_col", None) or "condition"
        )
        case = config.get("case") or getattr(design, "case", None)
        control = config.get("control") or getattr(design, "control", None)
        if condition_col not in adata.obs.columns or not case or not control:
            return self._skip(
                "condition_col/case/control not resolvable (no design block and none "
                "given directly)"
            )

        cluster_labels = config.get("cluster_labels") or {}
        figures_dir = Path(context.paths.figures) / "state_scoring"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))

        programs_arr = list(adata.uns[uns_key]["programs"])
        apply_theme("notebook")
        artifacts = []
        for program in requested:
            idx = programs_arr.index(program)
            df = pd.DataFrame(
                {
                    "cluster": adata.obs[cluster_key].astype(str).to_numpy(),
                    "condition": adata.obs[condition_col].astype(str).to_numpy(),
                    "score": adata.obsm[obsm_key][:, idx],
                }
            )
            try:
                fig = state_score_violin(
                    df,
                    cluster_col="cluster",
                    condition_col="condition",
                    score_col="score",
                    case=case,
                    control=control,
                    case_color=LE_RED,
                    control_color=NORMAL_BLUE,
                    score_label=f"{program} ({score_source})",
                    cluster_labels=cluster_labels,
                )
            except Exception:  # noqa: BLE001 -- one bad program must not sink the rest
                continue
            paths = save_figure(
                fig,
                figures_dir,
                f"state_score_violin_{score_source}_{program}",
                formats=formats,
                dpi=dpi,
            )
            artifacts.extend(
                figure_artifacts(
                    paths,
                    name="state_score_violin",
                    description=(
                        f"{program} ({score_source}) by cluster, {control} vs {case}, "
                        "with Mann-Whitney p-value per cluster."
                    ),
                )
            )

        if not artifacts:
            return self._skip("no figures produced")

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"state_score_violin rendered {len(requested)} program(s)."],
            warnings=[],
            metrics={"method": self.name, "programs": requested, "score_source": score_source},
            backend="python",
        )


__all__ = ["StateScoreViolinMethod"]
