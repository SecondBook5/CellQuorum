from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.executor import build_default_stage_registry
from cellquorum.core.planner import PipelinePlanner


def test_registry_contains_embeddings():
    reg = build_default_stage_registry()
    assert reg.get("embeddings") is not None
    assert "embeddings" in reg.registered_stage_names()


def test_planner_schedules_embeddings_before_de():
    cfg = CellQuorumConfig(project={"name": "t"}, input={"h5ad": "x.h5ad"})
    planner = PipelinePlanner(cfg)
    plan = planner._build_stage_plan()
    names = [p.name for p in plan]
    assert "embeddings" in names
    assert names.index("embeddings") < names.index("differential_expression")
    # composition draws in the UMAP's colours, so it runs once the embedding exists
    assert (
        names.index("embeddings")
        < names.index("composition")
        < names.index("differential_expression")
    )
