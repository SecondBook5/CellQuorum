"""Categorical embedding figures (with PAGA overlay) for each requested basis."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.embeddings import compute, plots
from cellquorum.embeddings.save import apply_theme, figure_artifacts, save_figure
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class CategoricalEmbeddingMethod(AnalysisMethod):
    """Render per-group scatter + PAGA overlay on each requested embedding."""

    name = "categorical_embedding"
    stage_category = "embeddings"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        figures_dir = Path(context.paths.figures) / "embeddings"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        threshold = float(config.get("paga_threshold", 0.2))
        tags = list(config.get("embeddings", ["umap", "phate"]))
        groupby = compute.resolve_paga_groupby(
            adata,
            config.get("paga_groupby"),
            cell_type_key=config.get("cell_type_key", "cell_type"),
            cluster_key=config.get("cluster_key", "leiden"),
        )
        if groupby is None:
            return MethodSkip(
                reason="categorical_embedding skipped: no grouping column present",
                details={"method": self.name},
            )

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for tag in tags:
            spec = plots.EMBEDDING_REGISTRY.get(tag)
            if spec is None:
                warnings.append(f"categorical_embedding: unknown embedding tag '{tag}'")
                continue
            if spec["obsm"] not in adata.obsm:
                warnings.append(f"categorical_embedding: {spec['obsm']} absent (tag '{tag}')")
                continue
            try:
                fig = plots.categorical_embedding(
                    adata,
                    groupby,
                    basis=spec["obsm"],
                    axis_labels=spec["axis"],
                    paga_threshold=threshold,
                )
                paths = save_figure(
                    fig, figures_dir, f"categorical_{tag}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths,
                    name="embedding_figure",
                    description=f"{tag} categorical embedding with PAGA overlay ({groupby}).",
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"categorical_embedding: {tag} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return MethodSkip(
                reason="categorical_embedding skipped: no embeddings available to render",
                details={"method": self.name, "warnings": warnings},
            )
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"categorical_embedding rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures, "groupby": groupby},
            backend="python",
        )


__all__ = ["CategoricalEmbeddingMethod"]
