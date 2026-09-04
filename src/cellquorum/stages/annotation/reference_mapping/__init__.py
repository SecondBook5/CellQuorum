"""Reference mapping package for atlas-based label transfer."""

# Import find_spec to probe for scvi WITHOUT executing it (see the note below).
from importlib.util import find_spec

from cellquorum.stages.annotation.reference_mapping.config import ReferenceMappingConfig
from cellquorum.stages.annotation.reference_mapping.stage import ReferenceMappingStage

# Probe for scvi by asking the import system whether it is *installable*, rather
# than importing it. `import scvi` inside a try/except keeps the engine from
# crashing when the GPU extra is absent, but when scvi IS present it executes the
# module — pulling torch, lightning, and jax (7000+ modules, ~2.3s) into every
# process that touches this package. Because `config/models.py` imports this
# stage's config, that cost landed on `import cellquorum` itself, so merely
# validating a YAML file loaded PyTorch. find_spec answers the same question
# ("can scvi be imported?") without running it, which keeps the engine-wide
# skip-not-crash invariant while restoring lazy loading. scarches.py imports scvi
# inside the method body, so nothing here needs the module object.
_has_scvi = find_spec("scvi") is not None

if _has_scvi:
    from cellquorum.methods.registry import METHOD_REGISTRY
    from cellquorum.stages.annotation.reference_mapping.scarches import ScArchesMethod

    # Self-register the scarches method (guarded against double registration).
    if not METHOD_REGISTRY.has("reference_mapping", "scarches"):
        METHOD_REGISTRY.register(ScArchesMethod)

    __all__ = ["ReferenceMappingConfig", "ReferenceMappingStage", "ScArchesMethod"]
else:
    __all__ = ["ReferenceMappingConfig", "ReferenceMappingStage"]
