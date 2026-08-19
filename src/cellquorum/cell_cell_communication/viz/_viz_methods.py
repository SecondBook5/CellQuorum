"""CCC visualization methods: thin AnalysisMethod subclasses that delegate to _plots."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.cell_cell_communication.viz import _plots
from cellquorum.cell_cell_communication.viz._io import (
    apply_theme,
    figure_artifacts,
    load_canonical_lr_sources,
    load_curvature,
    load_topology,
    save_figure,
)
from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip

# ─── Dotplot ───────────────────────────────────────────────────────────────


class DotplotVizMethod(AnalysisMethod):
    """Render source->target x ligand-receptor dotplots from canonical LR frames."""

    name = "dotplot_viz"
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
                fig = _plots.interaction_dotplot(df, top_k=top_k)
                paths = save_figure(fig, figures_dir, f"dotplot_{label}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"LR dotplot ({label})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"dotplot_viz: {label} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("no plottable rows in any source", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"dotplot_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


# ─── Chord ─────────────────────────────────────────────────────────────────


class ChordVizMethod(AnalysisMethod):
    """Render chord/circos figures from canonical LR frames."""

    name = "chord_viz"
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
                cts = list(set(df["source"].astype(str)) | set(df["target"].astype(str)))
                palette = _plots.celltype_palette(cts)
                fig = _plots.chord_diagram(df, palette=palette, top_k=top_k)
                paths = save_figure(fig, figures_dir, f"chord_{label}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"Chord diagram ({label})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"chord_viz: {label} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("no plottable rows in any source", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"chord_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


# ─── Sankey ────────────────────────────────────────────────────────────────


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


# ─── Network ───────────────────────────────────────────────────────────────


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


# ─── Summary ───────────────────────────────────────────────────────────────


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


__all__ = [
    "DotplotVizMethod",
    "ChordVizMethod",
    "SankeyVizMethod",
    "NetworkVizMethod",
    "SummaryVizMethod",
]
