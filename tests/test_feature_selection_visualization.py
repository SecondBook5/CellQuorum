"""Tests for the feature_selection (HVG) diagnostic figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.preprocessing.feature_selection.visualization import (
    _resolve_dispersion_column,
    write_hvg_figure,
)


def _var(*, y_column: str, n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A synthetic var table with a real (noisy hockey-stick) mean-variance relationship."""
    rng = np.random.default_rng(seed)
    means = rng.lognormal(0.0, 2.0, size=n)
    trend = np.log1p(means)  # monotone, saturating -- enough shape for lowess to fit.
    noise = rng.lognormal(0.0, 0.5, size=n)
    values = trend * noise
    # Flag the 20 furthest-above-trend genes as HVG, same logic real selection uses.
    above_trend = values / np.maximum(trend, 1e-6)
    highly_variable = np.zeros(n, dtype=bool)
    highly_variable[np.argsort(above_trend)[-20:]] = True
    return pd.DataFrame(
        {"means": means, y_column: values, "highly_variable": highly_variable},
        index=[f"gene_{i}" for i in range(n)],
    )


@pytest.mark.parametrize(
    "column,expected_label,expected_already_log",
    [
        ("dispersions", "Dispersion (log)", True),
        ("variances", "Variance", False),
    ],
)
def test_resolve_dispersion_column_matches_each_hvg_flavor(
    column, expected_label, expected_already_log
):
    resolved = _resolve_dispersion_column(pd.DataFrame({column: [1.0]}).columns)
    assert resolved == (column, expected_label, expected_already_log)


def test_resolve_dispersion_column_prefers_dispersions_when_both_present():
    # Should not happen in practice (one flavor runs at a time), but resolution order
    # must be deterministic rather than dict-iteration-order dependent.
    columns = pd.DataFrame({"dispersions": [0.0], "variances": [0.0]}).columns
    assert _resolve_dispersion_column(columns) == ("dispersions", "Dispersion (log)", True)


def test_resolve_dispersion_column_none_when_no_known_column():
    assert _resolve_dispersion_column(pd.DataFrame({"means": [0.0]}).columns) is None


def test_write_hvg_figure_returns_none_without_highly_variable_column(tmp_path):
    var = pd.DataFrame({"means": [1.0, 2.0], "dispersions": [0.1, 0.2]})
    result = write_hvg_figure(var, tmp_path / "hvg.png", method="seurat")
    assert result is None
    assert not (tmp_path / "hvg.png").exists()


def test_write_hvg_figure_writes_a_file_for_each_flavor(tmp_path):
    for column, method in (("dispersions", "seurat"), ("variances", "seurat_v3")):
        var = _var(y_column=column)
        out = tmp_path / f"hvg_{method}.png"
        result = write_hvg_figure(var, out, method=method, dpi=72)
        assert result == out
        assert out.exists()


def test_write_hvg_figure_draws_a_fitted_trend_and_selected_genes_on_top(tmp_path, monkeypatch):
    from cellquorum.stages.preprocessing.feature_selection import visualization as viz_module

    captured = {}

    def fake_save(fig, path, **kwargs):
        captured["fig"] = fig
        return [path]

    monkeypatch.setattr(viz_module, "save_cellquorum_figure", fake_save)

    var = _var(y_column="variances")
    write_hvg_figure(var, tmp_path / "hvg.png", method="seurat_v3", dpi=72)

    fig = captured["fig"]
    (ax,) = fig.axes

    # A fitted trend line, drawn as a Line2D, distinct from the two scatter layers.
    assert len(ax.lines) == 1

    collections = ax.collections
    assert len(collections) == 2
    top_layer = max(collections, key=lambda c: c.get_zorder())
    assert top_layer.get_offsets().shape[0] == 20  # matches the 20 flagged HVGs above
