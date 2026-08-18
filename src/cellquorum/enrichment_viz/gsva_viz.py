"""GSVA visualization method: per-sample heatmap + contrast diverging bar."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.enrichment_viz import plots
from cellquorum.enrichment_viz.discovery import collections_from_glob
from cellquorum.enrichment_viz.save import apply_theme, figure_artifacts, save_figure
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.visualization.figstyle import get_group_palette

# Keep the heatmap readable when a collection has many sources.
_HEATMAP_TOP_N = 50


class GsvaVizMethod(AnalysisMethod):
    """Render GSVA figures from the enrichment stage's GSVA CSVs."""

    name = "gsva_viz"
    stage_category = "enrichment_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        figures_dir = Path(context.paths.figures) / "enrichment"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        top_k = int(config.get("top_k", 12))
        wanted = config.get("collections")
        sample_condition = config.get("sample_condition")  # optional {sample: condition}

        # Collections come from the scores files (contrast may be absent for some).
        scores_cols = collections_from_glob(results_dir, "enrichment_gsva_scores_")
        contrast_cols = collections_from_glob(results_dir, "enrichment_gsva_contrast_")
        collections = sorted(set(scores_cols) | set(contrast_cols))
        if wanted:
            collections = [c for c in collections if c in set(wanted)]
        if not collections:
            return MethodSkip(
                reason="gsva_viz skipped: no enrichment_gsva_*.csv in results",
                details={"method": self.name},
            )

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for collection in collections:
            # Figure 6: per-sample heatmap from the scores matrix.
            scores_path = results_dir / f"enrichment_gsva_scores_{collection}.csv"
            if scores_path.exists():
                try:
                    matrix = pd.read_csv(scores_path, index_col=0)
                    col_colors = None
                    if sample_condition:
                        cond = pd.Series(
                            {s: sample_condition.get(s) for s in matrix.columns}
                        ).dropna()
                        if len(cond) == matrix.shape[1]:
                            palette = get_group_palette(list(cond.unique()))
                            col_colors = cond.map(palette)
                        else:
                            warnings.append(
                                f"gsva_viz: partial sample_condition for {collection}; "
                                "strip skipped"
                            )
                    grid = plots.annotated_clustermap(
                        matrix, col_colors=col_colors, top_n=_HEATMAP_TOP_N
                    )
                    paths = save_figure(
                        grid.fig,
                        figures_dir,
                        f"gsva_heatmap_{collection}",
                        formats=formats,
                        dpi=dpi,
                    )
                    artifacts += figure_artifacts(
                        paths,
                        name="enrichment_figure",
                        description=f"GSVA per-sample heatmap ({collection}).",
                    )
                    n_figures += 1
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"gsva_viz: heatmap failed ({collection}): {str(exc)[:200]}")

            # Figure 7: contrast diverging bar (case_mean - control_mean, fallback statistic).
            contrast_path = results_dir / f"enrichment_gsva_contrast_{collection}.csv"
            if contrast_path.exists():
                try:
                    contrast = pd.read_csv(contrast_path)
                    if {"case_mean", "control_mean"}.issubset(contrast.columns):
                        contrast = contrast.assign(
                            delta=contrast["case_mean"] - contrast["control_mean"]
                        )
                        value_col = "delta"
                    else:
                        value_col = "statistic"
                    ax = plots.diverging_bar(
                        contrast,
                        value_col=value_col,
                        label_col="source",
                        pvalue_col="padj",
                        top_k=top_k,
                    )
                    paths = save_figure(
                        ax.figure,
                        figures_dir,
                        f"gsva_contrast_{collection}",
                        formats=formats,
                        dpi=dpi,
                    )
                    artifacts += figure_artifacts(
                        paths,
                        name="enrichment_figure",
                        description=f"GSVA contrast bar ({collection}).",
                    )
                    n_figures += 1
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        f"gsva_viz: contrast bar failed ({collection}): {str(exc)[:200]}"
                    )

        if n_figures == 0:
            return MethodSkip(
                reason="gsva_viz skipped: no plottable GSVA data",
                details={"method": self.name, "warnings": warnings},
            )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"gsva_viz rendered {n_figures} figures over {collections}."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures, "collections": collections},
            backend="python",
        )


__all__ = ["GsvaVizMethod"]
