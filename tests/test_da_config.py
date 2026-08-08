from cellquorum.config.models import CellQuorumConfig, DifferentialAbundanceConfig


def test_da_config_defaults():
    cfg = DifferentialAbundanceConfig()
    assert cfg.enabled is True
    assert cfg.methods == []
    assert cfg.cell_type_col == "cell_type"
    assert cfg.use_rep == "X_pca_harmony"
    assert cfg.k == 30
    assert cfg.prop == 0.1
    assert cfg.spatial_fdr == 0.1
    assert cfg.reference_celltype is None
    assert cfg.seed == 0
    assert cfg.num_iterations == 20000
    assert cfg.inclusion_prob_threshold == 0.8
    assert cfg.transform == "asin"
    assert cfg.fdr == 0.05
    assert cfg.timeout_seconds == 1800


def test_da_config_mounted_on_cellquorum_config():
    cfg = CellQuorumConfig()
    assert isinstance(cfg.differential_abundance, DifferentialAbundanceConfig)


def test_da_config_rejects_unknown_field():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DifferentialAbundanceConfig(not_a_field=1)
