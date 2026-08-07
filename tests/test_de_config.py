from cellquorum.config.models import CellQuorumConfig, DifferentialExpressionConfig


def test_de_config_defaults():
    cfg = DifferentialExpressionConfig()
    assert cfg.enabled is True
    assert cfg.method == "pseudobulk_edger"
    assert cfg.layer == "counts"
    assert cfg.covariates == []
    assert cfg.fdr == 0.05


def test_de_config_mounted_on_cellquorum_config():
    cfg = CellQuorumConfig()
    assert isinstance(cfg.differential_expression, DifferentialExpressionConfig)


def test_de_config_rejects_unknown_field():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DifferentialExpressionConfig(covariates=[], not_a_field=1)
