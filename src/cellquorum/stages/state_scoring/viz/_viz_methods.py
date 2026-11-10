"""State-scoring visualization methods: quantitative companions to state_score_umap."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.state_scoring.viz.plots import population_score_violin, state_score_violin
from cellquorum.visualization.figio import figure_artifacts
from cellquorum.visualization.figstyle import LE_RED, NORMAL_BLUE, apply_theme, save_figure

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


class StateScoreViolinMethod(AnalysisMethod):
    """Render one violin figure per scored program: score by cluster, split control vs case, with
    a donor-level significance mark per cluster (donors from ``design``, BH across clusters) -- the
    number the state_score_umap spatial view doesn't give. Cells are never the unit of replication.
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
        descriptive = config.get("descriptive", False)
        if not descriptive and (condition_col not in adata.obs.columns or not case or not control):
            return self._skip(
                "condition_col/case/control not resolvable (no design block and none "
                "given directly)"
            )

        donor_col = getattr(design, "donor_col", None)
        if donor_col is not None and donor_col not in adata.obs.columns:
            donor_col = None
        paired = bool(getattr(design, "paired", True))

        cluster_labels = config.get("cluster_labels") or {}
        figures_dir = Path(context.paths.figures) / "state_scoring"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))

        programs_arr = list(adata.uns[uns_key]["programs"])
        apply_theme("notebook")
        artifacts = []
        failures = []
        for program in requested:
            idx = programs_arr.index(program)
            df = pd.DataFrame(
                {
                    "cluster": adata.obs[cluster_key].astype(str).to_numpy(),
                    "score": adata.obsm[obsm_key][:, idx],
                }
            )
            if not descriptive:
                df["condition"] = adata.obs[condition_col].astype(str).to_numpy()
            if donor_col is not None:
                df["donor"] = adata.obs[donor_col].astype(str).to_numpy()
            try:
                if descriptive:
                    fig = population_score_violin(
                        df,
                        score_label=f"{program} ({score_source})",
                        cluster_labels=cluster_labels,
                        cluster_key=cluster_key,
                        color=NORMAL_BLUE,
                    )
                else:
                    fig = state_score_violin(
                        df,
                        cluster_col="cluster",
                        condition_col="condition",
                        score_col="score",
                        case=str(case),
                        control=str(control),
                        case_color=LE_RED,
                        control_color=NORMAL_BLUE,
                        score_label=f"{program} ({score_source})",
                        cluster_labels=cluster_labels,
                        donor_col="donor" if donor_col is not None else None,
                        paired=paired,
                    )
            except Exception as exc:
                failures.append(f"{program}: {exc}")
                continue
            paths = save_figure(
                fig,
                figures_dir,
                f"state_score_violin_{score_source}_{program}",
                formats=formats,
                dpi=dpi,
            )
            if descriptive:
                table = (
                    Path(context.paths.results)
                    / "state_scoring"
                    / f"state_score_violin_{score_source}_{program}.csv"
                )
                table.parent.mkdir(parents=True, exist_ok=True)
                df.set_axis(adata.obs_names).to_csv(table, index_label="cell")
                artifacts.append(
                    StageArtifact(table.stem, table, "csv", "Observed cell scores by population.")
                )
            artifacts.extend(
                figure_artifacts(
                    paths,
                    name="state_score_violin",
                    description=(
                        f"{program} ({score_source}) observed across populations."
                        if descriptive
                        else f"{program} ({score_source}) by cluster, {control} vs {case}, "
                        "with a donor-level test per cluster (BH-adjusted)."
                    ),
                )
            )

        if not artifacts:
            return self._skip("no figures produced", warnings=failures)

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"state_score_violin rendered {len(requested) - len(failures)} program(s)."],
            warnings=failures,
            metrics={
                "method": self.name,
                "programs": requested,
                "score_source": score_source,
                "incomplete_reasons": failures,
            },
            backend="python",
        )


__all__ = ["StateScoreViolinMethod"]
