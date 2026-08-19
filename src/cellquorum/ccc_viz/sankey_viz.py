"""Sankey viz method: source->ligand->receptor->target flow from canonical LR frames."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.ccc_viz import _plots
from cellquorum.ccc_viz.discovery import load_canonical_lr_sources
from cellquorum.ccc_viz.save import apply_theme, figure_artifacts, save_figure
from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class SankeyVizMethod(AnalysisMethod):
    """Render Sankey flow diagrams from canonical LR frames."""

    name = "sankey_viz"
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
        wanted = config.get("sources")

        sources = load_canonical_lr_sources(results_dir, getattr(adata, "uns", None))
        if wanted:
            sources = [(lab, df) for lab, df in sources if lab in set(wanted)]
        if not sources:
            return self._skip("no canonical LR sources found")

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for label, df in sources:
            try:
                # Build palette from union of source/ligand/receptor/target values
                vals = (
                    set(df["source"].astype(str))
                    | set(df["target"].astype(str))
                    | set(df["ligand"].astype(str))
                    | set(df["receptor"].astype(str))
                )
                palette = _plots.celltype_palette(list(vals))
                fig = _plots.sankey_flow(df, palette=palette, top_k=top_k)
                paths = save_figure(fig, figures_dir, f"sankey_{label}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"Sankey diagram ({label})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"sankey_viz: {label} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("no plottable rows in any source", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"sankey_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


__all__ = ["SankeyVizMethod"]
