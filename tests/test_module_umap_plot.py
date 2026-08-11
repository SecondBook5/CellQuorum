# tests/test_module_umap_plot.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.coexpression.module_umap_plot import plot_module_umap


def _write_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "gene": [f"G{i}" for i in range(6)],
            "UMAP1": [0.1, 0.2, 0.9, 1.0, 0.5, 0.6],
            "UMAP2": [0.1, 0.2, 0.9, 1.0, 0.5, 0.6],
            "module": ["M1", "M1", "M2", "M2", "M1", "M2"],
            "color": ["red", "red", "blue", "blue", "red", "blue"],
            "hub": ["hub", "other", "hub", "other", "other", "hub"],
            "kME": [0.9, 0.4, 0.8, 0.3, 0.5, 0.7],
        }
    ).to_csv(path, index=False)


def test_writes_png_and_pdf(tmp_path: Path) -> None:
    csv = tmp_path / "module_umap.csv"
    _write_csv(csv)
    paths = plot_module_umap(csv, tmp_path, tag="demo")
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".pdf", ".png"]
    for p in paths:
        assert p.is_file() and p.stat().st_size > 0


def test_empty_or_all_grey_returns_empty(tmp_path: Path) -> None:
    csv = tmp_path / "module_umap.csv"
    pd.DataFrame(
        {
            "gene": ["G0"],
            "UMAP1": [0.0],
            "UMAP2": [0.0],
            "module": ["grey"],
            "color": ["grey"],
            "hub": ["other"],
            "kME": [0.1],
        }
    ).to_csv(csv, index=False)
    assert plot_module_umap(csv, tmp_path) == []


def test_uses_categorical_palette(tmp_path: Path) -> None:
    from cellquorum.visualization import figstyle

    csv = tmp_path / "module_umap.csv"
    _write_csv(csv)
    # Should not raise and should complete using the house palette
    # (smoke: default palette is None → CATEGORICAL_PALETTE).
    paths = plot_module_umap(csv, tmp_path, palette=None)
    assert len(paths) == 2
    assert isinstance(figstyle.CATEGORICAL_PALETTE, list)
