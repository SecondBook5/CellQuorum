from __future__ import annotations


def test_ccc_stage_registered():
    from cellquorum.core.executor import build_default_stage_registry
    from cellquorum.stages.cell_cell_communication.stage import CellCellCommunicationStage

    reg = build_default_stage_registry()
    assert "cell_cell_communication" in reg.stages
    assert isinstance(reg.stages["cell_cell_communication"], CellCellCommunicationStage)


def test_pipeline_config_has_ccc():
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.stages.cell_cell_communication.config import CellCellCommunicationConfig

    cfg = CellQuorumConfig()
    assert isinstance(cfg.cell_cell_communication, CellCellCommunicationConfig)
    assert cfg.cell_cell_communication.enabled is True
