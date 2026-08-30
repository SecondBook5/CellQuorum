"""Embeddings stage: compute (UMAP/PHATE/PAGA) then render (categorical/overlay)."""

from __future__ import annotations

# Import the package so methods register themselves as a side effect.
import cellquorum.stages.integration.embeddings  # noqa: F401
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage

# Keys bridged from config.embeddings into each dispatched method's config.
_EMB_CONFIG_KEYS = (
    "enabled",
    "use_rep",
    "umap_min_dist",
    "phate_knn",
    "phate_decay",
    "paga_groupby",
    "paga_threshold",
    "random_state",
    "embeddings",
    "figure_formats",
    "dpi",
    "overlay",
    "magic",
)

# Compute methods run before render methods; order is load-bearing (adata threads
# through _run_methods_list in list order).
_DEFAULT_METHODS = [
    {"method": "umap"},
    {"method": "phate"},
    {"method": "paga"},
    {"method": "categorical_embedding"},
    {"method": "continuous_overlay"},
]


@register_stage(
    name="embeddings",
    order=200,
    config_flag="embeddings",
    config_field="embeddings",
    category="embeddings",
)
class EmbeddingsStage(MethodDispatchStage):
    """Compute embeddings and render house-style figures + overlays."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "umap")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        """Bridge config.embeddings; inject method list + structural resolver keys."""
        augmented = dict(stage_config)
        config = getattr(context, "config", None)
        emb_cfg = getattr(config, "embeddings", None)
        if emb_cfg is not None:
            for key in _EMB_CONFIG_KEYS:
                if key not in augmented:
                    value = getattr(emb_cfg, key, None)
                    if value is not None:
                        augmented[key] = value
        # Structural keys for PAGA/overlay grouping resolution.
        if "cell_type_key" not in augmented:
            annotation = getattr(config, "annotation", None)
            augmented["cell_type_key"] = getattr(annotation, "key_added", "cell_type")
        if "cluster_key" not in augmented:
            clustering = getattr(config, "clustering", None)
            augmented["cluster_key"] = getattr(clustering, "key_added", "leiden")
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [dict(m) for m in _DEFAULT_METHODS]
        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: figures + obsm are written per-method; no strict postcondition."""  # noqa: B027


__all__ = ["EmbeddingsStage"]
