"""ORA visualization method: clusterProfiler-style barplot and dotplot."""

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


__all__ = ["OraVizMethod"]
