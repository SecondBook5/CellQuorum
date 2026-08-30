def test_stage_registered_in_executor():
    from cellquorum.stages.comparative.multicellular_programs.stage import MulticellularProgramsStage
    from cellquorum.core.executor import build_default_stage_registry

    reg = build_default_stage_registry()
    stage = reg.stages["multicellular_programs"]
    assert isinstance(stage, MulticellularProgramsStage)
    assert stage.name == "multicellular_programs"


def test_stage_planned_in_order():
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.core.planner import PipelinePlanner

    plan = PipelinePlanner(CellQuorumConfig()).build_plan()
    names = [s.name for s in plan.stages]
    assert "multicellular_programs" in names
