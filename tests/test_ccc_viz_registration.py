"""Tests for CCC visualization stage registration and configuration."""

from __future__ import annotations


def test_ccc_viz_config_defaults():
    from cellquorum.stages.cell_cell_communication.viz.config import CccVizConfig

    cfg = CccVizConfig()
    assert cfg.enabled is True
    assert cfg.top_k == 15
    assert cfg.figure_formats == ["pdf", "png"]
    assert cfg.dpi == 300
    assert cfg.sources is None
    assert cfg.levels is None


def test_all_five_methods_registered():
    import cellquorum.stages.cell_cell_communication.viz  # noqa: F401
    from cellquorum.methods.registry import METHOD_REGISTRY

    for name in ("dotplot_viz", "chord_viz", "sankey_viz", "network_viz", "summary_viz"):
        assert METHOD_REGISTRY.has("ccc_viz", name)


def test_stage_default_methods_list():
    from types import SimpleNamespace

    from cellquorum.stages.cell_cell_communication.viz.stage import CccVizStage

    stage = CccVizStage()
    ctx = SimpleNamespace(config=SimpleNamespace(ccc_viz=None))
    aug = stage._augment_config(ctx, {})
    names = [m["method"] for m in aug["methods"]]
    assert names == ["dotplot_viz", "chord_viz", "sankey_viz", "network_viz", "summary_viz"]


def test_registration_idempotent():
    import importlib

    import cellquorum.stages.cell_cell_communication.viz
    from cellquorum.methods.registry import METHOD_REGISTRY

    importlib.reload(
        cellquorum.stages.cell_cell_communication.viz
    )  # re-import must not raise on duplicate register
    assert METHOD_REGISTRY.has("ccc_viz", "dotplot_viz")
