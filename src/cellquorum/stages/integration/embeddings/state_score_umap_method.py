"""Cluster-state + per-condition score UMAP for a state_scoring program.

Nothing visualized state_scoring's obsm output (AUCell/UCell per-cell
program scores) on an embedding anywhere in the pipeline before this --
reuses embeddings.plots.plot_score_by_condition (cluster-state panel + two
condition panels on one shared color scale) rather than duplicating that
figure's centroid/palette/label machinery.

Lives in the ``embeddings`` package/stage category (order=200), NOT
``state_scoring`` (order=170), even though it reads state_scoring's
output: state_scoring runs BEFORE embeddings computes UMAP/PHATE, so a
method needing both a scored program AND an embedding can only run at or
after the later of the two.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.integration.embeddings.plots import (
    EMBEDDING_REGISTRY,
    plot_score_by_condition,
)
from cellquorum.visualization.figio import figure_artifacts
from cellquorum.visualization.figstyle import save_figure

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


class StateScoreUmapMethod(AnalysisMethod):
    """Render one cluster-state + per-condition score figure per program."""

    name = "state_score_umap"
    stage_category = "embeddings"
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

        basis_tag = config.get("basis", "umap")
        spec = EMBEDDING_REGISTRY.get(basis_tag)
        if spec is None or spec["obsm"] not in adata.obsm:
            return self._skip(f"embedding '{basis_tag}' not present")

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
        figures_dir = Path(context.paths.figures) / "embeddings"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))

        programs_arr = list(adata.uns[uns_key]["programs"])
        artifacts = []
        for program in requested:
            idx = programs_arr.index(program)
            score = adata.obsm[obsm_key][:, idx]
            try:
                fig = plot_score_by_condition(
                    adata,
                    score=score,
                    score_label=f"{program} ({score_source})",
                    cluster_key=cluster_key,
                    basis=spec["obsm"],
                    axis_labels=spec["axis"],
                    condition_col=condition_col,
                    case=case,
                    control=control,
                    cluster_labels=cluster_labels,
                )
            except Exception:  # noqa: BLE001 -- one bad program must not sink the rest
                continue
            paths = save_figure(
                fig, figures_dir, f"state_score_{score_source}_{program}", formats=formats, dpi=dpi
            )
            artifacts.extend(
                figure_artifacts(
                    paths,
                    name="state_score_umap",
                    description=(
                        f"Cluster-state map + {program} ({score_source}) score by "
                        f"condition ({control} vs {case})."
                    ),
                )
            )

        if not artifacts:
            return self._skip("no figures produced")

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"state_score_umap rendered {len(requested)} program(s)."],
            warnings=[],
            metrics={"method": self.name, "programs": requested, "score_source": score_source},
            backend="python",
        )


__all__ = ["StateScoreUmapMethod"]
