"""Tests for the feature_selection (HVG) diagnostic figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.preprocessing.feature_selection.visualization import (
    _resolve_dispersion_column,
    write_hvg_figure,
)


def _var(columns: dict, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = {
        "means": rng.lognormal(0.0, 1.5, size=n),
        "highly_variable": np.zeros(n, dtype=bool),
    }
    base["highly_variable"][:20] = True
    base.update(columns)
    return pd.DataFrame(base, index=[f"gene_{i}" for i in range(n)])


@pytest.mark.parametrize(
    "column,expected_label",
    [
        ("dispersions_norm", "Normalized dispersion"),
        ("variances_norm", "Normalized variance"),
        ("residual_variances", "Residual variance"),
    ],
)
def test_resolve_dispersion_column_matches_each_hvg_flavor(column, expected_label):
    columns = {column: np.random.default_rng(0).normal(size=200)}
    resolved = _resolve_dispersion_column(pd.DataFrame(columns).columns)
    assert resolved == (column, expected_label)


def test_resolve_dispersion_column_prefers_dispersions_norm_when_multiple_present():
    # Should not happen in practice (one flavor runs at a time), but resolution order
    # must be deterministic rather than dict-iteration-order dependent.
    columns = pd.DataFrame(
        {"dispersions_norm": [0.0], "variances_norm": [0.0], "residual_variances": [0.0]}
    ).columns
    assert _resolve_dispersion_column(columns) == ("dispersions_norm", "Normalized dispersion")


def test_resolve_dispersion_column_none_when_no_known_column():
    assert _resolve_dispersion_column(pd.DataFrame({"means": [0.0]}).columns) is None


def test_write_hvg_figure_returns_none_without_highly_variable_column(tmp_path):
    var = pd.DataFrame({"means": [1.0, 2.0], "dispersions_norm": [0.1, 0.2]})
    result = write_hvg_figure(var, tmp_path / "hvg.png", method="seurat")
    assert result is None
    assert not (tmp_path / "hvg.png").exists()


def test_write_hvg_figure_writes_a_file_for_each_flavor(tmp_path):
    for column, method in (
        ("dispersions_norm", "seurat"),
        ("variances_norm", "seurat_v3"),
        ("residual_variances", "pearson_residuals"),
    ):
        var = _var({column: np.random.default_rng(1).normal(size=200)})
        out = tmp_path / f"hvg_{method}.png"
        result = write_hvg_figure(var, out, method=method, dpi=72)
        assert result == out
        assert out.exists()


def test_write_hvg_figure_draws_selected_genes_above_the_rest(tmp_path, monkeypatch):
    from cellquorum.stages.preprocessing.feature_selection import visualization as viz_module

    captured = {}

    def fake_save(fig, path, **kwargs):
        captured["fig"] = fig
        return [path]

    monkeypatch.setattr(viz_module, "save_cellquorum_figure", fake_save)

    var = _var({"variances_norm": np.random.default_rng(2).normal(size=200)})
    write_hvg_figure(var, tmp_path / "hvg.png", method="seurat_v3", dpi=72)

    fig = captured["fig"]
    (ax,) = fig.axes
    collections = ax.collections
    assert len(collections) == 2
    # The higher-zorder layer (drawn last, on top) must be the smaller, selected group.
    top_layer = max(collections, key=lambda c: c.get_zorder())
    assert top_layer.get_offsets().shape[0] == 20  # matches the 20 flagged HVGs above
