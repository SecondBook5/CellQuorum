# tests/test_coexpression_wiring.py
def test_config_has_coexpression() -> None:
    from cellquorum.coexpression.config import CoexpressionConfig
    from cellquorum.config.models import CellQuorumConfig

    cfg = CellQuorumConfig()
    assert isinstance(cfg.coexpression, CoexpressionConfig)


def test_stage_registered_in_default_registry() -> None:
    from cellquorum.core.executor import build_default_stage_registry

    reg = build_default_stage_registry()
    assert "coexpression" in reg.registered_stage_names()
