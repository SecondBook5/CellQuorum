from cellquorum.stages.comparative.differential_abundance.stage import DifferentialAbundanceStage
from cellquorum.core.executor import build_default_stage_registry


def test_da_stage_is_registered():
    registry = build_default_stage_registry()
    assert "differential_abundance" in registry.registered_stage_names()
    assert isinstance(registry.get("differential_abundance"), DifferentialAbundanceStage)
