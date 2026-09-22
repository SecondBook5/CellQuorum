"""Tests for the metacell gene x metacell heatmap figure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.metacells.viz.heatmap import (
    _build_heatmap_figure,
    ordered_metacell_columns,
    plot_metacell_gene_heatmap,
    top_variable_genes,
)


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metacell_id": ["MC1", "MC2", "MC3"],
            "dominant_group": ["Fibroblast", "Endothelial", "Fibroblast"],
            "n_cells": [80, 95, 60],
        }
    )


def _expr() -> pd.DataFrame:
    return pd.DataFrame(
        {"MC1": [1.0, 2.0], "MC2": [0.1, 0.2], "MC3": [1.5, 1.8]},
        index=["GENE_A", "GENE_B"],
    )


def test_ordered_columns_group_by_dominant_type() -> None:
    order = ordered_metacell_columns(_nodes())
    assert order.index("MC1") < order.index("MC2") or order.index("MC3") < order.index("MC2")
    # Both Fibroblast metacells sit together, Endothelial elsewhere.
    fibro_positions = [order.index("MC1"), order.index("MC3")]
    endo_position = order.index("MC2")
    assert (max(fibro_positions) < endo_position) or (min(fibro_positions) > endo_position)


def test_writes_png_and_pdf(tmp_path: Path) -> None:
    paths = plot_metacell_gene_heatmap(_expr(), _nodes(), tmp_path)
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".pdf", ".png"]


def test_empty_expr_returns_empty(tmp_path: Path) -> None:
    assert plot_metacell_gene_heatmap(pd.DataFrame(), _nodes(), tmp_path) == []


def _many_genes_expr(n_genes: int = 30) -> pd.DataFrame:
    """Genes with strictly increasing variance -- GENE_{n-1} is the highest-variance
    gene, GENE_0 the lowest, so the expected top-k set is unambiguous."""

    rows = {f"GENE_{i}": [float(i), -float(i), 0.0] for i in range(n_genes)}
    expr = pd.DataFrame(rows).T
    expr.columns = ["MC1", "MC2", "MC3"]
    return expr


def test_top_variable_genes_keeps_highest_variance_rows() -> None:
    expr = _many_genes_expr(30)
    top = top_variable_genes(expr, 5)
    assert set(top.index) == {f"GENE_{i}" for i in range(25, 30)}


def test_top_variable_genes_noop_when_fewer_genes_than_n() -> None:
    expr = _many_genes_expr(3)
    top = top_variable_genes(expr, 50)
    assert set(top.index) == set(expr.index)


def test_heatmap_only_plots_top_variance_genes() -> None:
    """Reproduces the reviewer's 20,000-gene finding at small scale: with more
    genes than n_genes, only the top-variance genes should reach the figure's
    y-axis, not every gene in `expr`."""

    n_total, n_keep = 30, 5
    expr = _many_genes_expr(n_total)
    fig = _build_heatmap_figure(expr, _nodes(), n_genes=n_keep)
    ax = fig.axes[0]
    ytick_labels = {t.get_text() for t in ax.get_yticklabels()}
    assert ytick_labels == {f"GENE_{i}" for i in range(n_total - n_keep, n_total)}
    assert len(ytick_labels) == n_keep
