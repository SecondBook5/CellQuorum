"""Curvature-colored network viz method from ccc_network curvature CSVs."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.ccc_viz import _plots
from cellquorum.ccc_viz.discovery import load_curvature
from cellquorum.ccc_viz.save import apply_theme, figure_artifacts, save_figure
from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class NetworkVizMethod(AnalysisMethod):
    """Render curvature-colored networks from ccc_network curvature outputs."""

    name = "network_viz"
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
        wanted_levels = config.get("levels")
        levels = ("cci", "gci")
        if wanted_levels:
            levels = tuple(lv for lv in levels if lv in set(wanted_levels))

        curv = load_curvature(results_dir)
        if not curv:
            return self._skip("no curvature CSVs found")

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        # Whole-cohort levels: cci/gci (edges + nodes).
        for level in levels:
            edges = curv.get(f"{level}_edges")
            nodes = curv.get(f"{level}_nodes")
            if edges is None:
                continue
            try:
                fig = _plots.curvature_network(edges, nodes, top_k=top_k)
                paths = save_figure(fig, figures_dir, f"network_{level}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"Curvature network ({level})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"network_viz: {level} failed: {str(exc)[:200]}")
        # Differential levels: color by delta_curvature.
        for level in levels:
            diff = curv.get(f"differential_{level}")
            if diff is None or diff.empty:
                continue
            try:
                fig = _plots.curvature_network(
                    diff,
                    None,
                    curvature_col="delta_curvature",
                    node_curv_col="delta_curvature",
                    top_k=top_k,
                )
                paths = save_figure(
                    fig, figures_dir, f"network_diff_{level}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths,
                    name="ccc_figure",
                    description=f"Differential curvature network ({level}).",
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"network_viz: differential {level} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("no plottable curvature levels", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"network_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


__all__ = ["NetworkVizMethod"]
