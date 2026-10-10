"""annotation_viz: the atlas marker dotplot, drawn from the annotation stage's own panel."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.annotation.viz.methods import MarkerDotplotMethod, expected_blocks

_PANEL = {"Alpha": ["G0", "G1"], "Beta": ["G2", "G3"]}


def _adata(n=90) -> ad.AnnData:
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(6)]
    X = rng.poisson(1.0, size=(n, 6)).astype("float32")
    labels = np.array(["Alpha"] * 30 + ["Beta"] * 30 + ["Gamma"] * 30)
    X[labels == "Alpha", 0:2] += 4
    X[labels == "Beta", 2:4] += 4
    a = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
    a.layers["norm"] = np.log1p(X)
    a.obs["cell_type"] = pd.Categorical(labels, categories=["Alpha", "Beta", "Gamma"])
    return a


def _ctx(tmp_path: Path):
    paths = SimpleNamespace(
        root=tmp_path,
        results=tmp_path / "results",
        figures=tmp_path / "figs",
        scratch=tmp_path / "scratch",
    )
    return SimpleNamespace(paths=paths, config=SimpleNamespace())


def _config(**over):
    return {
        "marker_panels": _PANEL,
        "group_by": "cell_type",
        "layer": "norm",
        "figure_formats": ["png"],
        **over,
    }


def test_a_group_is_expected_to_light_up_the_block_that_carries_its_name():
    assert expected_blocks(["Alpha", "Beta", "Gamma"], _PANEL) == {"Alpha": "Alpha", "Beta": "Beta"}


def test_a_panel_block_no_group_carries_is_not_highlighted_anywhere():
    assert expected_blocks(["Alpha"], _PANEL) == {"Alpha": "Alpha"}


def test_skips_and_says_why_when_there_is_no_marker_panel(tmp_path):
    res = MarkerDotplotMethod()._run(_adata(), _config(marker_panels={}), _ctx(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "marker_panels" in res.reason


def test_skips_when_the_label_column_is_missing(tmp_path):
    res = MarkerDotplotMethod()._run(_adata(), _config(group_by="nope"), _ctx(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "nope" in res.reason


def test_skips_when_the_layer_is_missing(tmp_path):
    res = MarkerDotplotMethod()._run(_adata(), _config(layer="nope"), _ctx(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "layer" in res.reason


def test_skips_when_no_panel_gene_is_in_the_data(tmp_path):
    res = MarkerDotplotMethod()._run(
        _adata(), _config(marker_panels={"Alpha": ["NOPE1", "NOPE2"]}), _ctx(tmp_path)
    )
    assert isinstance(res, MethodSkip)


def test_writes_the_figure_and_the_numbers_behind_it(tmp_path):
    res = MarkerDotplotMethod()._run(_adata(), _config(), _ctx(tmp_path))
    assert not isinstance(res, MethodSkip)
    assert (tmp_path / "figs" / "annotation" / "marker_dotplot.png").exists()
    assert (tmp_path / "results" / "annotation" / "marker_dotplot_mean.csv").exists()
    assert (tmp_path / "results" / "annotation" / "marker_dotplot_pct.csv").exists()
    assert res.artifacts


def test_the_numbers_are_per_group_per_gene_in_panel_order(tmp_path):
    MarkerDotplotMethod()._run(_adata(), _config(), _ctx(tmp_path))
    mean = pd.read_csv(tmp_path / "results" / "annotation" / "marker_dotplot_mean.csv", index_col=0)
    assert list(mean.index) == ["G0", "G1", "G2", "G3"]
    assert list(mean.columns) == ["Alpha", "Beta", "Gamma"]
    assert mean.loc["G0", "Alpha"] > mean.loc["G0", "Beta"]


def test_it_reports_which_groups_it_highlighted(tmp_path):
    res = MarkerDotplotMethod()._run(_adata(), _config(), _ctx(tmp_path))
    assert res.metrics["highlighted"] == ["Alpha", "Beta"]


def test_highlighting_can_be_switched_off(tmp_path):
    res = MarkerDotplotMethod()._run(_adata(), _config(highlight_expected=False), _ctx(tmp_path))
    assert res.metrics["highlighted"] == []


# ---- wiring -----------------------------------------------------------------------------


def test_the_stage_and_its_config_are_part_of_the_default_config():
    from cellquorum.config.models import CellQuorumConfig

    cfg = CellQuorumConfig()
    assert cfg.stages.annotation_viz is True
    assert cfg.annotation_viz.enabled is True
    assert cfg.annotation_viz.highlight_expected is True


def test_the_method_is_registered():
    import cellquorum.stages.annotation.viz  # noqa: F401
    from cellquorum.methods.registry import METHOD_REGISTRY

    assert METHOD_REGISTRY.has("annotation_viz", "marker_dotplot")


def test_the_panel_the_label_column_and_the_layer_come_from_the_annotation_block():
    from cellquorum.stages.annotation.viz.stage import AnnotationVizStage

    ctx = SimpleNamespace(
        config=SimpleNamespace(
            annotation=SimpleNamespace(
                marker_panels=_PANEL, key_added="lineage", score_layer="norm"
            ),
            annotation_viz=None,
        )
    )
    aug = AnnotationVizStage()._augment_config(ctx, {})
    assert aug["marker_panels"] == _PANEL
    assert aug["group_by"] == "lineage"
    assert aug["layer"] == "norm"
    assert [m["method"] for m in aug["methods"]] == ["marker_dotplot"]


def test_an_explicit_label_column_beats_the_annotation_block():
    from cellquorum.stages.annotation.viz.stage import AnnotationVizStage

    ctx = SimpleNamespace(
        config=SimpleNamespace(
            annotation=SimpleNamespace(
                marker_panels=_PANEL, key_added="lineage", score_layer="norm"
            ),
            annotation_viz=SimpleNamespace(group_by="canonical_population"),
        )
    )
    assert AnnotationVizStage()._augment_config(ctx, {})["group_by"] == "canonical_population"


def test_the_planner_runs_it_after_population_identity_and_before_subclustering():
    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.core.planner import build_pipeline_plan

    names = [s.name for s in build_pipeline_plan(CellQuorumConfig()).stages]
    assert names.index("population_identity") < names.index("annotation_viz")
    assert names.index("annotation_viz") < names.index("subclustering")


def test_the_executor_registry_includes_it():
    from cellquorum.core.executor import build_default_stage_registry
    from cellquorum.stages.annotation.viz.stage import AnnotationVizStage

    registry = build_default_stage_registry()
    assert isinstance(registry.get("annotation_viz"), AnnotationVizStage)
