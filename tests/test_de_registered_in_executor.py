from cellquorum.comparative.differential_expression.stage import DifferentialExpressionStage
from cellquorum.core.executor import build_default_stage_registry


def test_de_stage_is_registered():
    registry = build_default_stage_registry()
    assert "differential_expression" in registry.registered_stage_names()
    assert isinstance(registry.get("differential_expression"), DifferentialExpressionStage)
