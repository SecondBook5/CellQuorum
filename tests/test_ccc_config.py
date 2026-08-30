from __future__ import annotations


def test_defaults():
    from cellquorum.stages.cell_cell_communication.config import CellCellCommunicationConfig

    c = CellCellCommunicationConfig()
    assert c.enabled is True
    assert c.methods == []
    assert c.cell_type_col == "cell_type"
    assert c.sample_col == "sample_id"
    assert c.layer == "cellquorum_normalized"
    assert c.seed == 42
    assert c.resource_name == "consensus"
    assert c.expr_prop == 0.1
    assert c.min_cells == 5
    assert c.n_perms == 100
    assert c.rank is None
    assert c.tf_optimization == "robust"
    assert c.min_samples == 3
    assert c.tensor_how == "outer"
    assert abs(c.outer_fraction - 1.0 / 3.0) < 1e-9
    assert c.timeout_seconds == 1800


def test_strict_rejects_unknown_field():
    import pytest
    from pydantic import ValidationError

    from cellquorum.stages.cell_cell_communication.config import CellCellCommunicationConfig

    with pytest.raises(ValidationError):
        CellCellCommunicationConfig(not_a_field=1)


def test_nichenet_config_defaults():
    from cellquorum.stages.cell_cell_communication.config import CellCellCommunicationConfig

    cfg = CellCellCommunicationConfig()
    # prior models unset by default -> methods skip cleanly
    assert cfg.nichenet_ligand_target_matrix is None
    assert cfg.nichenet_lr_network is None
    assert cfg.nichenet_weighted_networks is None
    # validated MultiNicheNet defaults
    assert cfg.mnn_top_n_target == 250
    assert cfg.mnn_fraction_cutoff == 0.05
    assert cfg.mnn_min_sample_prop == 0.5
    assert cfg.mnn_logfc_threshold == 0.5
    assert cfg.mnn_p_val_adj is False
    # NicheNet defaults
    assert cfg.nichenet_top_ligands == 10
    assert cfg.nichenet_top_targets == 50
    assert cfg.nichenet_de_top_n == 200
    assert cfg.nichenet_sender is None
    assert cfg.nichenet_receiver is None
    assert cfg.nichenet_n_cores == 4
    assert cfg.nichenet_timeout_seconds == 7200
