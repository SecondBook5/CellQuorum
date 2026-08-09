from __future__ import annotations


def test_defaults():
    from cellquorum.cell_cell_communication.config import CellCellCommunicationConfig

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

    from cellquorum.cell_cell_communication.config import CellCellCommunicationConfig

    with pytest.raises(ValidationError):
        CellCellCommunicationConfig(not_a_field=1)
