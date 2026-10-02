"""Tests for the two metacell QC scatter figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.metacells.viz.qc_scatter import (
    plot_metacell_qc_size_comparison,
    plot_metacell_qc_unclassified_vs_size,
)


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metacell_id": ["MC1", "MC2"],
            "dominant_group": ["Fibroblast", "Endothelial"],
            "n_cells": [80, 95],
        }
    )


def _assignments() -> pd.DataFrame:
    rows = []
    for i in range(80):
        rows.append(
            {"cell_id": f"MC1_{i}", "metacell_id": "MC1", "group": "Fibroblast", "n_umis": 1000 + i}
        )
    for i in range(95):
        rows.append(
            {"cell_id": f"MC2_{i}", "metacell_id": "MC2", "group": "Endothelial", "n_umis": 800 + i}
        )
    for i in range(10):
        rows.append(
            {
                "cell_id": f"out_{i}",
                "metacell_id": "Outliers",
                "group": "Fibroblast" if i % 2 else "Endothelial",
                "n_umis": 150 + i,
            }
        )
    return pd.DataFrame(rows)


def test_unclassified_vs_size_writes_png_and_pdf(tmp_path: Path) -> None:
    paths = plot_metacell_qc_unclassified_vs_size(_assignments(), _nodes(), tmp_path)
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".pdf", ".png"]


def test_size_comparison_writes_png_and_pdf(tmp_path: Path) -> None:
    paths = plot_metacell_qc_size_comparison(_assignments(), _nodes(), tmp_path)
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".pdf", ".png"]


def test_size_comparison_empty_when_no_outliers(tmp_path: Path) -> None:
    df = _assignments()
    df = df[df["metacell_id"] != "Outliers"]
    assert plot_metacell_qc_size_comparison(df, _nodes(), tmp_path) == []
