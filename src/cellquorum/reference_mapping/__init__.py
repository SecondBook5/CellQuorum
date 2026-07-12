"""Reference mapping package for atlas-based label transfer."""

from cellquorum.reference_mapping.config import ReferenceMappingConfig
from cellquorum.reference_mapping.stage import ReferenceMappingStage

# Guard scvi import for environments without GPU dependencies.
_has_scvi = False
try:
    import scvi  # noqa: F401

    _has_scvi = True
except ImportError:
    pass

if _has_scvi:
    from cellquorum.reference_mapping.scarches import ScArchesMethod

    __all__ = ["ReferenceMappingConfig", "ReferenceMappingStage", "ScArchesMethod"]
else:
    __all__ = ["ReferenceMappingConfig", "ReferenceMappingStage"]
