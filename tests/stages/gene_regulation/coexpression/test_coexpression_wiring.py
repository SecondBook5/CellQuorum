# tests/stages/gene_regulation/coexpression/test_coexpression_wiring.py
def test_config_has_coexpression() -> None:
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.stages.gene_regulation.coexpression.config import CoexpressionConfig

    cfg = CellQuorumConfig()
    assert isinstance(cfg.coexpression, CoexpressionConfig)
