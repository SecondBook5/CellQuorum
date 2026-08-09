from __future__ import annotations


def test_ccc_stage_registered():
    from cellquorum.cell_cell_communication.stage import CellCellCommunicationStage
    from cellquorum.core.executor import build_default_stage_registry

    reg = build_default_stage_registry()
    assert "cell_cell_communication" in reg.stages
    assert isinstance(reg.stages["cell_cell_communication"], CellCellCommunicationStage)


def test_pipeline_config_has_ccc():
    from cellquorum.cell_cell_communication.config import CellCellCommunicationConfig
    from cellquorum.config.models import CellQuorumConfig

    cfg = CellQuorumConfig()
    assert isinstance(cfg.cell_cell_communication, CellCellCommunicationConfig)
    assert cfg.cell_cell_communication.enabled is True
