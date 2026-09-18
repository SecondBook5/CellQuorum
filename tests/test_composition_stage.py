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
    assert legend_labels == ["Papillary", "Reticular"]
    # The data itself is untouched -- x-axis groups still read the real index.
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Normal", "Case"]
