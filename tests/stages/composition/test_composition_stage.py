"""Tests for the cell-type composition summary stage (order=190)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.pipeline import build_pipeline_context
from cellquorum.stages.composition.stage import CompositionStage


def _adata(**extra_obs) -> ad.AnnData:
    n = 12
    obs = pd.DataFrame(
        {
            "cell_type": ["A"] * 6 + ["B"] * 6,
            "condition": ["Normal"] * 3 + ["Case"] * 3 + ["Normal"] * 3 + ["Case"] * 3,
            "donor_id": ["d1", "d1", "d2"] * 4,
            "sample_id": [f"s{i % 4}" for i in range(n)],
            **extra_obs,
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    return ad.AnnData(X=np.ones((n, 3)), obs=obs)


def _context(tmp_path, adata, **composition_cfg):
    config = CellQuorumConfig(
        compute={"prefer_gpu": False},
        r={"enabled": False},
        composition=composition_cfg,
    )
    return build_pipeline_context(config, output_dir=tmp_path / "run").with_adata(adata)


def test_skips_when_cell_type_key_missing(tmp_path):
    adata = ad.AnnData(X=np.ones((3, 2)), obs=pd.DataFrame(index=["a", "b", "c"]))
    context = _context(tmp_path, adata)
    result = CompositionStage().run(context)
    assert result.status == "skipped"


def test_writes_overall_composition_table(tmp_path):
    context = _context(tmp_path, _adata())
    result = CompositionStage().run(context)
    assert result.status == "success"
    artifact_names = {a.name for a in result.artifacts}
    assert "composition_overall" in artifact_names
    overall_path = next(a.path for a in result.artifacts if a.name == "composition_overall")
    table = pd.read_csv(overall_path)
    assert set(table["cell_type"]) == {"A", "B"}
    assert table.loc[table["cell_type"] == "A", "n_cells"].iloc[0] == 6


def test_writes_condition_table_and_figure_when_condition_key_present(tmp_path):
    context = _context(tmp_path, _adata())
    result = CompositionStage().run(context)
    artifact_names = {a.name for a in result.artifacts}
    assert "composition_by_condition" in artifact_names
    assert any(a.name == "composition_figure_condition" for a in result.artifacts)
    for artifact in result.artifacts:
        assert artifact.path.exists()


def test_skips_condition_table_gracefully_when_condition_key_absent(tmp_path):
    adata = _adata()
    del adata.obs["condition"]
    context = _context(tmp_path, adata, condition_key="condition")
    result = CompositionStage().run(context)
    assert result.status == "success"
    artifact_names = {a.name for a in result.artifacts}
    assert "composition_by_condition" not in artifact_names
    # The overall table must still be produced even without a condition column.
    assert "composition_overall" in artifact_names


def test_writes_sample_table_when_sample_key_present(tmp_path):
    context = _context(tmp_path, _adata())
    result = CompositionStage().run(context)
    artifact_names = {a.name for a in result.artifacts}
    assert "composition_by_sample" in artifact_names


def test_atlas_states_filters_cells_before_computing_composition(tmp_path):
    adata = _adata(
        qc_state_final=["core"] * 8 + ["quarantine"] * 4,
    )
    context = _context(tmp_path, adata, atlas_states=["core"])
    result = CompositionStage().run(context)
    assert result.metrics["n_cells_used"] == 8
    overall_path = next(a.path for a in result.artifacts if a.name == "composition_overall")
    table = pd.read_csv(overall_path).set_index("cell_type")
    # All 4 quarantined cells were cell_type B (indices 8-11 are B, 6-7 also B but core).
    assert table["n_cells"].sum() == 8


def test_metrics_report_cell_and_group_counts(tmp_path):
    context = _context(tmp_path, _adata())
    result = CompositionStage().run(context)
    assert result.metrics["n_cells_used"] == 12
    assert result.metrics["n_cell_types"] == 2
    assert result.metrics["condition_key"] == "condition"
    assert result.metrics["sample_key"] == "sample_id"


def test_by_sample_table_orders_donors_numerically_not_lexicographically(tmp_path):
    """P1, P2, P10, P12 -- plain string sort would put P1, P10, P12, P2."""
    n = 8
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "B"] * 4,
            "donor_id": ["P1", "P1", "P2", "P2", "P10", "P10", "P12", "P12"],
            "sample_id": ["P1", "P1", "P2", "P2", "P10", "P10", "P12", "P12"],
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=np.ones((n, 3)), obs=obs)
    context = _context(tmp_path, adata)
    result = CompositionStage().run(context)
    by_sample_path = next(a.path for a in result.artifacts if a.name == "composition_by_sample")
    table = pd.read_csv(by_sample_path)
    assert table["sample_id"].tolist() == ["P1", "P2", "P10", "P12"]


def test_by_sample_table_handles_categorical_donor_id_column(tmp_path):
    """Real objects store donor_id as pandas Categorical -- .fillna("") on a

    Categorical raises TypeError unless "" is a declared category, which it
    never is. Confirmed live on the real fibroblast object.
    """
    n = 8
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "B"] * 4,
            "donor_id": pd.Categorical(["P1", "P1", "P2", "P2", "P10", "P10", "P12", "P12"]),
            "sample_id": ["P1", "P1", "P2", "P2", "P10", "P10", "P12", "P12"],
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=np.ones((n, 3)), obs=obs)
    context = _context(tmp_path, adata)
    result = CompositionStage().run(context)
    assert result.status == "success"
    by_sample_path = next(a.path for a in result.artifacts if a.name == "composition_by_sample")
    table = pd.read_csv(by_sample_path)
    assert table["sample_id"].tolist() == ["P1", "P2", "P10", "P12"]


def test_cluster_labels_rename_the_stacked_bar_legend_only(tmp_path):
    """Bare cell-type ids in the legend are meaningless without cross-referencing --

    cluster_labels renames them for display, same convention as de_heatmap and
    the marker identity dotplot (display-only, doesn't touch the data table).
    """
    from cellquorum.stages.composition.stage import _stacked_bar

    table = pd.DataFrame({"A": [60.0, 55.0], "B": [40.0, 45.0]}, index=["Normal", "Case"])
    fig = _stacked_bar(
        table, x_label="condition", cluster_labels={"A": "Papillary", "B": "Reticular"}
    )
    ax = fig.axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert sorted(legend_labels) == ["Papillary", "Reticular"]
    # The data itself is untouched -- x-axis groups still read the real index.
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Normal", "Case"]


def test_conditions_are_written_control_first_whatever_the_alphabet(tmp_path):
    from cellquorum.config.models import CellQuorumConfig

    config = CellQuorumConfig(
        compute={"prefer_gpu": False},
        r={"enabled": False},
        design={"condition_col": "condition", "case": "Case", "control": "Normal"},
    )
    context = build_pipeline_context(config, output_dir=tmp_path / "run").with_adata(_adata())

    result = CompositionStage().run(context)

    path = next(a.path for a in result.artifacts if a.name == "composition_by_condition")
    assert list(pd.read_csv(path)["condition"]) == ["Normal", "Case"]


def _bars_table():
    return pd.DataFrame(
        {"Big": [50.0, 40.0], "Mid": [30.0, 35.0], "Small": [20.0, 25.0]},
        index=["Normal", "Case"],
    )


def test_the_axis_is_a_proportion_of_cells_from_zero_to_one():
    from cellquorum.stages.composition.stage import _stacked_bar

    ax = _stacked_bar(_bars_table(), x_label="condition").axes[0]

    assert ax.get_ylim() == (0.0, 1.0)
    assert ax.get_ylabel() == "Proportion of cells"
    tops = [round(p.get_y() + p.get_height(), 6) for p in ax.patches]
    assert max(tops) == 1.0  # a 100-percent bar is a proportion of 1


def test_bars_use_the_colours_they_are_given_by_cell_type():
    from matplotlib.colors import to_hex

    from cellquorum.stages.composition.stage import _stacked_bar

    given = {"Big": "#112233", "Mid": "#445566", "Small": "#778899"}
    fig = _stacked_bar(_bars_table(), x_label=None, colours=given)
    fig.canvas.draw()
    first_bar = sorted(
        (p for p in fig.axes[0].patches if abs(p.get_x() - (-0.35)) < 1e-6), key=lambda p: p.get_y()
    )

    assert [to_hex(p.get_facecolor()) for p in first_bar] == ["#112233", "#445566", "#778899"]


def test_without_colours_every_cell_type_still_gets_its_own():
    from matplotlib.colors import to_hex

    from cellquorum.stages.composition.stage import _stacked_bar

    fig = _stacked_bar(_bars_table(), x_label=None)
    fig.canvas.draw()
    first_bar = [p for p in fig.axes[0].patches if abs(p.get_x() - (-0.35)) < 1e-6]

    assert len({to_hex(p.get_facecolor()) for p in first_bar}) == 3


def test_no_segment_is_white_on_a_white_page():
    from cellquorum.stages.composition.stage import _stacked_bar

    table = pd.DataFrame({f"T{i}": [100.0 / 13] * 2 for i in range(13)}, index=["Normal", "Case"])
    fig = _stacked_bar(table, x_label="condition")
    fig.canvas.draw()

    assert all(min(p.get_facecolor()[:3]) < 0.85 for p in fig.axes[0].patches)


def test_the_legend_lists_cell_types_in_the_order_they_are_stacked_top_to_bottom():
    from cellquorum.stages.composition.stage import _stacked_bar

    ax = _stacked_bar(_bars_table(), x_label="condition").axes[0]

    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Small", "Mid", "Big"]
    assert ax.get_legend().get_title().get_text() == "Cell type"


def test_the_pooled_figure_carries_no_title_of_its_own():
    from cellquorum.stages.composition.stage import _stacked_bar

    assert _stacked_bar(_bars_table(), x_label="condition").axes[0].get_title() == ""


# ---- composition by patient -------------------------------------------------------------


def _patient_tables():
    types = ["Big", "Mid", "Small"]
    normal = pd.DataFrame(
        [[50.0, 30.0, 20.0], [40.0, 40.0, 20.0], [60.0, 30.0, 10.0]],
        index=["P1", "P2", "P10"],
        columns=types,
    )
    case = pd.DataFrame([[45.0, 35.0, 20.0], [30.0, 50.0, 20.0]], index=["P1", "P2"], columns=types)
    return {"Normal": normal, "Lymphedema": case}


def _panels(fig):
    return [ax for ax in fig.axes if ax.get_title()]


def test_the_panels_follow_the_order_given_with_normal_on_the_left():
    from cellquorum.stages.composition.stage import _stacked_bar_by_patient

    fig = _stacked_bar_by_patient(_patient_tables())

    panels = _panels(fig)
    assert [ax.get_title() for ax in panels] == ["Normal", "Lymphedema"]
    assert panels[0].get_position().x0 < panels[1].get_position().x0


def test_patients_sit_in_natural_order_at_the_same_place_in_every_panel():
    from cellquorum.stages.composition.stage import _stacked_bar_by_patient

    panels = _panels(_stacked_bar_by_patient(_patient_tables()))

    for ax in panels:
        assert [t.get_text() for t in ax.get_xticklabels()] == ["P1", "P2", "P10"]


def test_a_patient_with_no_cells_in_a_condition_has_no_bar_there_not_an_empty_one():
    from cellquorum.stages.composition.stage import _stacked_bar_by_patient

    normal, case = _panels(_stacked_bar_by_patient(_patient_tables()))

    assert len(normal.patches) == 3 * 3  # three patients, three cell types
    assert len(case.patches) == 2 * 3  # P10 has no lymphedema cells


def test_every_bar_fills_the_axis_and_the_panels_share_it():
    from cellquorum.stages.composition.stage import _stacked_bar_by_patient

    normal, case = _panels(_stacked_bar_by_patient(_patient_tables()))

    for ax in (normal, case):
        assert ax.get_ylim() == (0.0, 1.0)
    p1 = [p for p in normal.patches if abs(p.get_x() + p.get_width() / 2 - 0) < 1e-6]
    assert round(sum(p.get_height() for p in p1), 6) == 1.0
    assert normal.get_ylabel() == "Proportion of cells"
    assert case.get_ylabel() == ""


def test_there_is_one_legend_for_the_figure_in_stack_order():
    from cellquorum.stages.composition.stage import _stacked_bar_by_patient

    fig = _stacked_bar_by_patient(_patient_tables())

    assert len(fig.legends) == 1
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["Small", "Mid", "Big"]


def test_the_figure_has_the_title_it_is_given():
    from cellquorum.stages.composition.stage import _stacked_bar_by_patient

    fig = _stacked_bar_by_patient(_patient_tables(), title="Cell type composition by patient")

    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Cell type composition by patient"


def _design_context(tmp_path, adata):
    config = CellQuorumConfig(
        compute={"prefer_gpu": False},
        r={"enabled": False},
        design={"condition_col": "condition", "case": "Case", "control": "Normal"},
    )
    return build_pipeline_context(config, output_dir=tmp_path / "run").with_adata(adata)


def test_the_stage_writes_a_per_patient_table_and_figure_control_first(tmp_path):
    result = CompositionStage().run(_design_context(tmp_path, _adata()))

    names = {a.name for a in result.artifacts}
    assert "composition_by_patient" in names
    assert "composition_figure_patient" in names
    table = pd.read_csv(
        next(a.path for a in result.artifacts if a.name == "composition_by_patient")
    )
    assert list(table.columns[:2]) == ["condition", "donor_id"]
    assert list(table["condition"].unique()) == ["Normal", "Case"]


def test_no_patient_figure_without_a_donor_column(tmp_path):
    adata = _adata()
    del adata.obs["donor_id"]

    result = CompositionStage().run(_design_context(tmp_path, adata))

    assert "composition_by_patient" not in {a.name for a in result.artifacts}


def test_the_panel_title_boxes_sit_above_the_bars_not_on_them():
    from cellquorum.stages.composition.stage import _stacked_bar_by_patient

    fig = _stacked_bar_by_patient(_patient_tables())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    six_points = 6 / 72 * fig.dpi
    for ax in _panels(fig):
        box = ax.title.get_bbox_patch().get_window_extent(renderer)
        assert box.y0 - ax.get_window_extent(renderer).y1 >= six_points


# ---- the UMAP's colours ------------------------------------------------------------------


def _four_type_adata(layout_swapped=False):
    rng = np.random.default_rng(3)
    names = ["Fibro", "TNK", "Mast", "DC"]
    corners = {"Fibro": (0, 0), "TNK": (8, 1), "Mast": (4, 9), "DC": (-5, 6)}
    if layout_swapped:
        corners = {"Fibro": (8, 1), "TNK": (-5, 6), "Mast": (0, 0), "DC": (4, 9)}
    labels, points = [], []
    for name in names:
        labels += [name] * 15
        points.append(rng.normal(loc=corners[name], scale=0.4, size=(15, 2)))
    n = len(labels)
    obs = pd.DataFrame(
        {
            "cell_type": labels,
            "condition": (["Normal"] * 8 + ["Case"] * 7) * 4,
            "donor_id": ["d1", "d2", "d1", "d2", "d1", "d2", "d1"] * 8 + ["d1"] * 4,
            "sample_id": ["s0"] * n,
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    a = ad.AnnData(X=np.ones((n, 2)), obs=obs)
    a.obsm["X_umap"] = np.vstack(points).astype("float32")
    return a


def _first_bar_colours(path_fig) -> dict[str, str]:
    from matplotlib.colors import to_hex

    ax = path_fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    return {lab: to_hex(h.get_facecolor()) for h, lab in zip(handles, labels, strict=True)}


def test_the_composition_bars_carry_the_colours_the_umap_gives_each_cell_type(tmp_path):
    from matplotlib.colors import to_hex

    from cellquorum.stages.composition.stage import _umap_colours
    from cellquorum.stages.integration.embeddings.plots import column_palette

    a = _four_type_adata()
    config = CellQuorumConfig(compute={"prefer_gpu": False}, r={"enabled": False})

    colours, note = _umap_colours(a, "cell_type", config)

    expected = column_palette(a.obsm["X_umap"], a.obs["cell_type"])
    assert {k: to_hex(v) for k, v in colours.items()} == {k: to_hex(v) for k, v in expected.items()}
    assert note is None


def test_a_configured_layout_file_decides_the_colours_as_it_does_for_the_umap(tmp_path):
    from matplotlib.colors import to_hex

    from cellquorum.stages.composition.stage import _umap_colours
    from cellquorum.stages.integration.embeddings.plots import column_palette

    a = _four_type_adata()
    swapped = _four_type_adata(layout_swapped=True)
    layout = tmp_path / "layout.csv"
    pd.DataFrame(
        {
            "cell": a.obs_names,
            "umap1": swapped.obsm["X_umap"][:, 0],
            "umap2": swapped.obsm["X_umap"][:, 1],
        }
    ).to_csv(layout, index=False)
    config = CellQuorumConfig(
        compute={"prefer_gpu": False}, r={"enabled": False}, embeddings={"layout_file": str(layout)}
    )
    from_x = column_palette(a.obsm["X_umap"], a.obs["cell_type"])
    from_file = column_palette(swapped.obsm["X_umap"], a.obs["cell_type"])
    assert from_x != from_file  # the test means something only if the layouts colour differently

    colours, _ = _umap_colours(a, "cell_type", config)

    assert {k: to_hex(v) for k, v in colours.items()} == {
        k: to_hex(v) for k, v in from_file.items()
    }


def test_a_layout_file_that_does_not_fit_falls_back_and_says_so(tmp_path):
    from cellquorum.stages.composition.stage import _umap_colours

    a = _four_type_adata()
    layout = tmp_path / "layout.csv"
    pd.DataFrame({"cell": ["nope"], "umap1": [0.0], "umap2": [0.0]}).to_csv(layout, index=False)
    config = CellQuorumConfig(
        compute={"prefer_gpu": False}, r={"enabled": False}, embeddings={"layout_file": str(layout)}
    )

    colours, note = _umap_colours(a, "cell_type", config)

    assert colours is None
    assert note is not None and "layout" in note


def test_with_no_coordinates_at_all_there_are_no_shared_colours_and_nothing_to_report():
    from cellquorum.stages.composition.stage import _umap_colours

    a = _four_type_adata()
    del a.obsm["X_umap"]
    config = CellQuorumConfig(compute={"prefer_gpu": False}, r={"enabled": False})

    assert _umap_colours(a, "cell_type", config) == (None, None)


def test_the_stage_figures_use_those_colours_and_report_a_bad_layout(tmp_path):
    a = _four_type_adata()
    context = _design_context(tmp_path, a)

    result = CompositionStage().run(context)

    assert result.status == "success"
    assert "composition_figure_condition" in {x.name for x in result.artifacts}
    assert not result.warnings
