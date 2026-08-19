"""Summary viz method: CCI heatmap (per source) + topology role facets (per level)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.ccc_viz import _plots
from cellquorum.ccc_viz.discovery import load_canonical_lr_sources, load_topology
from cellquorum.ccc_viz.save import apply_theme, figure_artifacts, save_figure
from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class SummaryVizMethod(AnalysisMethod):
    """Render CCI heatmaps and topology role facets."""

    name = "summary_viz"
    stage_category = "ccc_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        figures_dir = Path(context.paths.figures) / "ccc"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        top_k = int(config.get("top_k", 15))

        sources = load_canonical_lr_sources(results_dir, getattr(adata, "uns", None))
        wanted = config.get("sources")
        if wanted:
            sources = [(lab, df) for lab, df in sources if lab in set(wanted)]
        topo = load_topology(results_dir)
        if not sources and not topo:
            return self._skip("no LR sources or topology found")

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for label, df in sources:
            try:
                fig = _plots.cci_heatmap(df, diverging=False)
                paths = save_figure(fig, figures_dir, f"heatmap_{label}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"CCI heatmap ({label})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"summary_viz: heatmap {label} failed: {str(exc)[:200]}")
        for level, df in topo.items():
            try:
                fig = _plots.topology_facets(df, top_k=top_k)
                paths = save_figure(fig, figures_dir, f"topology_{level}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"Topology roles ({level})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"summary_viz: topology {level} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("nothing plottable", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"summary_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


__all__ = ["SummaryVizMethod"]
