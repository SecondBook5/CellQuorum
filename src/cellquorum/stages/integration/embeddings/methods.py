"""Embeddings analysis methods: UMAP, PHATE, PAGA compute + rendering."""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.integration.embeddings import compute, overlay, plots
from cellquorum.stages.integration.embeddings.config import MagicConfig, OverlayConfig
from cellquorum.stages.integration.embeddings.plots import (
    apply_theme,
    figure_artifacts,
    save_figure,
)


def _seed(config: dict, context: object) -> int:
    """Prefer config random_state, else context.random_seed, else 1337."""
    if config.get("random_state") is not None:
        return int(config["random_state"])
    return int(getattr(context, "random_seed", 1337))


class UmapMethod(AnalysisMethod):
    """Compute UMAP coordinates from the neighbors graph."""

    name = "umap"
    stage_category = "embeddings"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        try:
            compute.compute_umap(
                adata,
                min_dist=float(config.get("umap_min_dist", 0.3)),
                random_state=_seed(config, context),
            )
        except compute.EmbeddingsComputeError as exc:
            return self._skip(f"{exc}")
        return StageResult(
            adata=adata,
            notes=["umap: wrote obsm['X_umap']"],
            metrics={"method": self.name},
            backend="python",
        )


class PhateMethod(AnalysisMethod):
    """Compute PHATE coordinates from a representation."""

    name = "phate"
    stage_category = "embeddings"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        try:
            compute.compute_phate(
                adata,
                use_rep=config.get("use_rep", "X_pca_harmony"),
                knn=int(config.get("phate_knn", 15)),
                decay=int(config.get("phate_decay", 40)),
                random_state=_seed(config, context),
            )
        except compute.EmbeddingsComputeError as exc:
            return self._skip(f"{exc}")
        return StageResult(
            adata=adata,
            notes=["phate: wrote obsm['X_phate']"],
            metrics={"method": self.name},
            backend="python",
        )


class PagaMethod(AnalysisMethod):
    """Compute PAGA connectivity over cell-type or cluster groups."""

    name = "paga"
    stage_category = "embeddings"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        groupby = compute.resolve_paga_groupby(
            adata,
            config.get("paga_groupby"),
            cell_type_key=config.get("cell_type_key", "cell_type"),
            granular_key=config.get("granular_key", "cell_type_granular"),
            cluster_key=config.get("cluster_key", "leiden"),
        )
        if groupby is None:
            return self._skip("no grouping column (cell_type/leiden) present")
        try:
            compute.compute_paga(adata, groupby=groupby)
        except compute.EmbeddingsComputeError as exc:
            return self._skip(f"{exc}")
        return StageResult(
            adata=adata,
            notes=[f"paga: wrote uns['paga'] over '{groupby}'"],
            metrics={"method": self.name, "groupby": groupby},
            backend="python",
        )


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
            granular_key=config.get("granular_key", "cell_type_granular"),
            cluster_key=config.get("cluster_key", "leiden"),
        )
        if groupby is None:
            return self._skip("no grouping column present")

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
            return self._skip("no embeddings available to render", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"categorical_embedding rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures, "groupby": groupby},
            backend="python",
        )


def _as_overlay(value: object) -> OverlayConfig:
    """Coerce the overlay config (model or dict) into an OverlayConfig."""
    if isinstance(value, OverlayConfig):
        return value
    if isinstance(value, dict):
        return OverlayConfig(**value)
    return OverlayConfig()


def _as_magic(value: object) -> MagicConfig:
    """Coerce the magic config (model or dict) into a MagicConfig."""
    if isinstance(value, MagicConfig):
        return value
    if isinstance(value, dict):
        return MagicConfig(**value)
    return MagicConfig()


class ContinuousOverlayMethod(AnalysisMethod):
    """Color each requested embedding by each requested feature; opt-in MAGIC."""

    name = "continuous_overlay"
    stage_category = "embeddings"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        figures_dir = Path(context.paths.figures) / "embeddings"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        tags = list(config.get("embeddings", ["umap", "phate"]))
        overlay_cfg = _as_overlay(config.get("overlay"))
        magic_cfg = _as_magic(config.get("magic"))
        seed = _seed(config, context)

        warnings: list[str] = []

        # Opt-in MAGIC: impute only the overlay genes, then read from that layer.
        gene_layer = None
        if magic_cfg.enabled and overlay_cfg.genes:
            try:
                imputed = overlay.impute_magic_scoped(
                    adata,
                    overlay_cfg.genes,
                    knn=magic_cfg.knn,
                    solver=magic_cfg.solver,
                    random_state=magic_cfg.random_state,
                )
                if imputed:
                    gene_layer = "magic"
            except overlay.MagicUnavailable as exc:
                warnings.append(f"continuous_overlay: MAGIC unavailable ({exc}); using raw X")

        features, feat_warnings = overlay.resolve_features(
            adata, overlay_cfg, random_state=seed, layer=gene_layer
        )
        warnings += feat_warnings
        if not features:
            return self._skip("no resolvable overlay features", warnings=warnings)

        apply_theme()
        artifacts, n_figures = [], 0
        for tag in tags:
            spec = plots.EMBEDDING_REGISTRY.get(tag)
            if spec is None or spec["obsm"] not in adata.obsm:
                warnings.append(f"continuous_overlay: embedding '{tag}' unavailable")
                continue
            coords = adata.obsm[spec["obsm"]]
            # Track used stems per embedding to detect and de-duplicate collisions.
            used_stems: dict[str, int] = {}
            for feat in features:
                try:
                    fig = plots.continuous_overlay(
                        coords, feat.values, title=feat.label, axis_labels=spec["axis"]
                    )
                    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", feat.label)
                    # De-duplicate stem collisions: append counter if already used.
                    if safe in used_stems:
                        used_stems[safe] += 1
                        disambiguated = f"{safe}_{feat.kind}_{used_stems[safe]}"
                        warnings.append(
                            f"continuous_overlay: filename collision for '{feat.label}' "
                            f"(sanitized='{safe}'), disambiguated to '{disambiguated}'"
                        )
                        safe = disambiguated
                    else:
                        used_stems[safe] = 0
                    paths = save_figure(
                        fig, figures_dir, f"overlay_{tag}_{safe}", formats=formats, dpi=dpi
                    )
                    artifacts += figure_artifacts(
                        paths,
                        name="embedding_figure",
                        description=f"{tag} overlay of {feat.kind} '{feat.label}'.",
                    )
                    n_figures += 1
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        f"continuous_overlay: {tag}/{feat.label} failed: {str(exc)[:200]}"
                    )

        if n_figures == 0:
            return self._skip("no figures rendered", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"continuous_overlay rendered {n_figures} figures."],
            warnings=warnings,
            metrics={
                "method": self.name,
                "n_figures": n_figures,
                "magic_used": gene_layer is not None,
            },
            backend="python",
        )


__all__ = [
    "CategoricalEmbeddingMethod",
    "ContinuousOverlayMethod",
    "PagaMethod",
    "PhateMethod",
    "UmapMethod",
]
