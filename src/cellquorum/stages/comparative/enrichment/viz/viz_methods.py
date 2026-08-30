"""Enrichment visualization methods: thin dispatch wrappers for the plotting library."""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.stages.comparative.enrichment.viz import plots
from cellquorum.stages.comparative.enrichment.viz.io import (
    apply_theme,
    collections_from_glob,
    figure_artifacts,
    save_figure,
)
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.visualization.figstyle import get_group_palette

# Cap on how many running-ES per-source curves to render per collection.
_MAX_RUNNING_ES_FIGS = 6

# Keep the heatmap readable when a collection has many sources.
_HEATMAP_TOP_N = 50


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


class GseaVizMethod(AnalysisMethod):
    """Render GSEA figures from the enrichment stage's GSEA CSVs."""

    name = "gsea_viz"
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

        # Discover summary CSVs, excluding the runningES companion files.
        all_cols = collections_from_glob(results_dir, "enrichment_gsea_")
        collections = [c for c in all_cols if not c.startswith("runningES_")]
        if wanted:
            collections = [c for c in collections if c in set(wanted)]
        if not collections:
            return self._skip("no enrichment_gsea_*.csv in results")

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for collection in collections:
            try:
                df = pd.read_csv(results_dir / f"enrichment_gsea_{collection}.csv")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"gsea_viz: failed to read {collection} CSV: {str(exc)[:200]}")
                continue
            if df.empty:
                warnings.append(f"gsea_viz: {collection} summary CSV is empty")
                continue

            # Figure 1: diverging NES bar.
            try:
                ax = plots.diverging_bar(
                    df, value_col="score", label_col="source", pvalue_col="padj", top_k=top_k
                )
                paths = save_figure(
                    ax.figure, figures_dir, f"gsea_diverging_{collection}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths,
                    name="enrichment_figure",
                    description=f"GSEA diverging NES bar ({collection}).",
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"gsea_viz: diverging bar failed ({collection}): {str(exc)[:200]}")

            # Figure 2: activity dotplot.
            try:
                ax = plots.activity_dotplot(
                    df, value_col="score", label_col="source", pvalue_col="padj", top_k=top_k
                )
                paths = save_figure(
                    ax.figure, figures_dir, f"gsea_activity_{collection}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths,
                    name="enrichment_figure",
                    description=f"GSEA activity dotplot ({collection}).",
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"gsea_viz: activity dotplot failed ({collection}): {str(exc)[:200]}"
                )

            # Figure 3: running-ES curves (only if the companion CSV exists).
            running_path = results_dir / f"enrichment_gsea_runningES_{collection}.csv"
            if running_path.exists():
                try:
                    running = pd.read_csv(running_path)
                    for source in list(dict.fromkeys(running["source"]))[:_MAX_RUNNING_ES_FIGS]:
                        sub = running[running["source"] == source]
                        fig = plots.running_es_curve(sub, title=str(source))
                        safe = re.sub(r"[^0-9A-Za-z_.-]", "_", str(source))
                        paths = save_figure(
                            fig,
                            figures_dir,
                            f"gsea_runningES_{collection}_{safe}",
                            formats=formats,
                            dpi=dpi,
                        )
                        artifacts += figure_artifacts(
                            paths,
                            name="enrichment_figure",
                            description=f"GSEA running-ES ({collection}, {source}).",
                        )
                        n_figures += 1
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"gsea_viz: running-ES failed ({collection}): {str(exc)[:200]}")
            else:
                warnings.append(f"gsea_viz: no running-ES CSV for {collection} (figure 3 skipped)")

        if n_figures == 0:
            return self._skip("no plottable rows in any GSEA CSV", warnings=warnings)

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"gsea_viz rendered {n_figures} figures over {collections}."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures, "collections": collections},
            backend="python",
        )


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
            return self._skip("no enrichment_gsva_*.csv in results")

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
            return self._skip("no plottable GSVA data", warnings=warnings)

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"gsva_viz rendered {n_figures} figures over {collections}."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures, "collections": collections},
            backend="python",
        )


class OraVizMethod(AnalysisMethod):
    """Render ORA figures from the enrichment stage's ORA CSVs."""

    name = "ora_viz"
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

        collections = collections_from_glob(results_dir, "enrichment_ora_")
        if wanted:
            collections = [c for c in collections if c in set(wanted)]
        if not collections:
            return self._skip("no enrichment_ora_*.csv in results")

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for collection in collections:
            try:
                df = pd.read_csv(results_dir / f"enrichment_ora_{collection}.csv")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"ora_viz: failed to read {collection} CSV: {str(exc)[:200]}")
                continue
            if df.empty:
                warnings.append(f"ora_viz: {collection} CSV is empty")
                continue

            try:
                fig = plots.ora_barplot(
                    df,
                    count_col="count",
                    label_col="source",
                    padj_col="padj",
                    facet_col="direction",
                    top_k=top_k,
                )
                paths = save_figure(
                    fig, figures_dir, f"ora_barplot_{collection}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths, name="enrichment_figure", description=f"ORA barplot ({collection})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"ora_viz: barplot failed ({collection}): {str(exc)[:200]}")

            try:
                fig = plots.ora_dotplot(
                    df,
                    ratio_col="gene_ratio",
                    count_col="count",
                    padj_col="padj",
                    label_col="source",
                    facet_col="direction",
                    top_k=top_k,
                )
                paths = save_figure(
                    fig, figures_dir, f"ora_dotplot_{collection}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths, name="enrichment_figure", description=f"ORA dotplot ({collection})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"ora_viz: dotplot failed ({collection}): {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("no plottable rows in any ORA CSV", warnings=warnings)

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"ora_viz rendered {n_figures} figures over {collections}."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures, "collections": collections},
            backend="python",
        )


__all__ = ["ActivityVizMethod", "GseaVizMethod", "GsvaVizMethod", "OraVizMethod"]
