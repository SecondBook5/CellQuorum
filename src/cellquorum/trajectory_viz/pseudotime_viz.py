"""Pseudotime-on-embedding scatter figures."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory_viz import inputs, plots
from cellquorum.trajectory_viz.save import apply_theme, figure_artifacts, save_figure


class PseudotimeVizMethod(AnalysisMethod):
    name = "pseudotime_viz"
    stage_category = "trajectory_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        basis = inputs.resolve_basis(adata, config.get("embedding_basis"))
        if basis is None:
            return self._skip("no embedding basis in obsm")
        pts = inputs.available_pseudotimes(adata, config.get("pseudotime_keys"))
        if not pts:
            return self._skip("no *_pseudotime obs column")

        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        coords = adata.obsm[basis]

        apply_theme()
        artifacts, warnings, n = [], [], 0
        for key in pts:
            try:
                values = inputs.numeric_obs(adata, key)
                fig = plots.embedding_scatter(coords, values, title=key, cbar_label=key)
                paths = save_figure(fig, figures_dir, f"pseudotime_{key}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="trajectory_figure", description=f"{key} on {basis}."
                )
                n += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"pseudotime_viz: {key} failed: {str(exc)[:200]}")

        if n == 0:
            return self._skip("no plottable pseudotime", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"pseudotime_viz rendered {n} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n, "basis": basis},
            backend="python",
        )


__all__ = ["PseudotimeVizMethod"]
