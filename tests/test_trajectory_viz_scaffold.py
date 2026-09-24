import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cellquorum.stages.trajectory.viz._helpers import apply_theme, figure_artifacts, save_figure
from cellquorum.stages.trajectory.viz.config import TrajectoryVizConfig
from cellquorum.stages.trajectory.viz.stage import TrajectoryVizStage


def test_config_defaults_are_render_only():
    c = TrajectoryVizConfig()
    assert c.enabled is True
    assert c.figure_formats == ["pdf", "png"]
    assert c.dpi == 300
    assert c.top_k == 15
    # No defaulted biology.
    assert c.genes is None and c.cluster_key is None and c.lineages is None
    assert c.pseudotime_keys is None and c.embedding_basis is None
    assert c.fate_labels is None
    assert c.heatmap_categorical_keys is None
    assert c.heatmap_categorical_labels is None
    assert c.heatmap_value_labels is None
    assert c.heatmap_compute_cell_cycle is False


def test_new_config_fields_reach_methods_through_the_stage_whitelist():
    """fate_labels and the new heatmap_* fields must actually be in
    _VIZ_CONFIG_KEYS -- a config field that exists on TrajectoryVizConfig but
    is missing from that whitelist is silently dropped before any method ever
    sees it (the exact bug found live: fate_labels existed and was read by
    DriverVizMethod, but _augment_config never forwarded it).
    """
    from cellquorum.stages.trajectory.viz.stage import _VIZ_CONFIG_KEYS

    for field in (
        "fate_labels",
        "heatmap_categorical_keys",
        "heatmap_categorical_labels",
        "heatmap_value_labels",
        "heatmap_compute_cell_cycle",
    ):
        assert field in _VIZ_CONFIG_KEYS, f"{field} is on the config but not the whitelist"


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
        "activity_cascade",
    ]
    stage._validate_output(None)  # no raise
