from __future__ import annotations


def test_pipeline_config_has_ccc():
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.stages.cell_cell_communication.config import CellCellCommunicationConfig

    cfg = CellQuorumConfig()
    assert isinstance(cfg.cell_cell_communication, CellCellCommunicationConfig)
    assert cfg.cell_cell_communication.enabled is True
