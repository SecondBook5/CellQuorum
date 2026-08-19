import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cellquorum.trajectory.viz.config import TrajectoryVizConfig
from cellquorum.trajectory.viz.save import apply_theme, figure_artifacts, save_figure
from cellquorum.trajectory.viz.stage import TrajectoryVizStage


def test_config_defaults_are_render_only():
    c = TrajectoryVizConfig()
    assert c.enabled is True
    assert c.figure_formats == ["pdf", "png"]
    assert c.dpi == 300
    assert c.top_k == 15
    # No defaulted biology.
    assert c.genes is None and c.cluster_key is None and c.lineages is None
    assert c.pseudotime_keys is None and c.embedding_basis is None


def test_save_figure_writes_all_formats_and_tags_figure(tmp_path):
    apply_theme()
    fig = plt.figure()
    paths = save_figure(fig, tmp_path / "trajectory", "demo", formats=("pdf", "png"), dpi=72)
    assert [p.name for p in paths] == ["demo.pdf", "demo.png"]
    assert all(p.exists() for p in paths)
    arts = figure_artifacts(paths, name="trajectory_figure", description="x")
    assert all(a.kind == "figure" for a in arts)


def test_stage_default_method_list_and_noop_validate():
    stage = TrajectoryVizStage()
    assert stage.name == "trajectory_viz"
    aug = stage._augment_config(object(), {})
    methods = [m["method"] for m in aug["methods"]]
    assert methods == [
        "pseudotime_viz",
        "fate_viz",
        "driver_viz",
        "gene_trend_viz",
        "macrostate_viz",
        "velocity_viz",
        "pseudotime_heatmap",
    ]
    stage._validate_output(None)  # no raise
