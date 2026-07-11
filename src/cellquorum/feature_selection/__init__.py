"""Feature-selection (HVG) stage package."""

from __future__ import annotations

from cellquorum.feature_selection.hvg import HVGMethod
from cellquorum.feature_selection.stage import FeatureSelectionStage
from cellquorum.methods.registry import METHOD_REGISTRY


# Registry only uses cls.name, so create thin subclasses for each flavor name.
class _HVGSeuratV3(HVGMethod):
    name = "seurat_v3"


class _HVGPearsonResiduals(HVGMethod):
    name = "pearson_residuals"


# Register the HVG method under each supported flavor name so _select_method_name
# can resolve seurat / seurat_v3 / pearson_residuals to the same strategy class.
if not METHOD_REGISTRY.has("feature_selection", "seurat"):
    METHOD_REGISTRY.register(HVGMethod)
if not METHOD_REGISTRY.has("feature_selection", "seurat_v3"):
    METHOD_REGISTRY.register(_HVGSeuratV3)
if not METHOD_REGISTRY.has("feature_selection", "pearson_residuals"):
    METHOD_REGISTRY.register(_HVGPearsonResiduals)

__all__ = ["FeatureSelectionStage", "HVGMethod"]
