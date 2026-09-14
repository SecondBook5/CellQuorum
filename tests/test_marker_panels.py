"""Tests for annotation marker-panel figures (dotplot + feature grid)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.visualization.marker_panels import plot_marker_dotplot, plot_marker_feature_grid


def _adata(n=40, n_genes=8) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(n, n_genes)).astype(float)
    a = ad.AnnData(X=X, var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]))
    a.layers["norm"] = X.copy()
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    a.obs["cell_type"] = pd.Categorical(
        (["A"] * (n // 2)) + (["B"] * (n - n // 2)), categories=["A", "B"]
    )
    return a


def test_plot_marker_dotplot_writes_png_and_a_vector_twin(tmp_path):
    a = _adata()
    marker_panel = {"A": ["G0", "G1"], "B": ["G2", "G3"]}

    out = plot_marker_dotplot(
        a, marker_panel, groupby="cell_type", layer="norm", out_path=tmp_path / "dotplot.png"
    )

    assert out == tmp_path / "dotplot.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_marker_dotplot_drops_genes_absent_from_the_object(tmp_path):
    a = _adata()
    marker_panel = {"A": ["G0", "NOT_A_REAL_GENE"], "B": ["G2", "G3"]}

    out = plot_marker_dotplot(
        a, marker_panel, groupby="cell_type", layer="norm", out_path=tmp_path / "dotplot.png"
    )

    assert out is not None
    assert out.exists()


def test_plot_marker_dotplot_returns_none_when_no_panel_genes_present(tmp_path):
    a = _adata()
    marker_panel = {"A": ["NOPE1"], "B": ["NOPE2"]}

    out = plot_marker_dotplot(
        a, marker_panel, groupby="cell_type", layer="norm", out_path=tmp_path / "dotplot.png"
    )

    assert out is None
    assert not (tmp_path / "dotplot.png").exists()


def test_plot_marker_feature_grid_writes_png_and_a_vector_twin(tmp_path):
    a = _adata()
    markers = {"A": "G0", "B": "G2"}

    out = plot_marker_feature_grid(
        a, markers, basis="X_umap", layer="norm", out_path=tmp_path / "grid.png"
    )

    assert out == tmp_path / "grid.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_marker_feature_grid_skips_genes_absent_from_the_object(tmp_path):
    a = _adata()
    markers = {"A": "G0", "B": "NOT_A_REAL_GENE"}

    out = plot_marker_feature_grid(
        a, markers, basis="X_umap", layer="norm", out_path=tmp_path / "grid.png"
    )

    assert out is not None
    assert out.exists()


def test_plot_marker_feature_grid_returns_none_when_no_markers_present(tmp_path):
    a = _adata()
    markers = {"A": "NOPE1", "B": "NOPE2"}

    out = plot_marker_feature_grid(
        a, markers, basis="X_umap", layer="norm", out_path=tmp_path / "grid.png"
    )

    assert out is None
    assert not (tmp_path / "grid.png").exists()
