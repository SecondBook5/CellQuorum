"""Activity visualization method: cross-cell-type activity dotplot."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.enrichment.viz import plots
from cellquorum.enrichment.viz.discovery import collections_from_glob
from cellquorum.enrichment.viz.save import apply_theme, figure_artifacts, save_figure
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class ActivityVizMethod(AnalysisMethod):
    """Render the per-cell-type activity dotplot from activity CSVs."""

    name = "activity_viz"
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
        wanted = config.get("resources")

        resources = collections_from_glob(results_dir, "enrichment_activity_")
        if wanted:
            resources = [r for r in resources if r in set(wanted)]
        if not resources:
            return self._skip("no enrichment_activity_*.csv in results")

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for resource in resources:
            try:
                df = pd.read_csv(results_dir / f"enrichment_activity_{resource}.csv")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"activity_viz: failed to read {resource} CSV: {str(exc)[:200]}")
                continue
            if df.empty:
                warnings.append(f"activity_viz: {resource} CSV is empty")
                continue
            try:
                fig = plots.cross_group_dotplot(
                    df, row_col="source", col_col="cell_type", value_col="mean_score", top_k=top_k
                )
                paths = save_figure(
                    fig, figures_dir, f"activity_dotplot_{resource}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths, name="enrichment_figure", description=f"Activity dotplot ({resource})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"activity_viz: dotplot failed ({resource}): {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("no plottable rows in any activity CSV", warnings=warnings)

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"activity_viz rendered {n_figures} figures over {resources}."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures, "resources": resources},
            backend="python",
        )


__all__ = ["ActivityVizMethod"]
