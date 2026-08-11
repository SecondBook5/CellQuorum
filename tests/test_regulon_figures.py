"""Tests for the house-styled SCENIC regulon figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.grn import regulon_figures as rf


def _auc(n_cells: int = 60, n_reg: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = rng.random((n_cells, n_reg))
    cols = [f"TF{j}_(+)" for j in range(n_reg)]
    idx = [f"cell_{i}" for i in range(n_cells)]
    return pd.DataFrame(data, index=idx, columns=cols)


def _groups(auc: pd.DataFrame) -> pd.Series:
    labels = ["A" if i % 2 == 0 else "B" for i in range(auc.shape[0])]
    return pd.Series(labels, index=auc.index)


def test_rss_shape_and_bounds() -> None:
    auc = _auc()
    rss = rf.regulon_specificity_scores(auc, _groups(auc))
    assert set(rss.index) == {"A", "B"}
    assert list(rss.columns) == list(auc.columns)
    assert np.isfinite(rss.to_numpy()).all()


def test_rss_panels_writes_png_and_pdf(tmp_path: Path) -> None:
    auc = _auc()
    paths = rf.plot_rss_panels(auc, _groups(auc), tmp_path, top_n=3)
    suffixes = {Path(p).suffix for p in paths}
    assert {".png", ".pdf"} <= suffixes


def test_regulon_clustermap_writes_files(tmp_path: Path) -> None:
    auc = _auc()
    paths = rf.plot_regulon_clustermap(auc, _groups(auc), tmp_path, top_n=3)
    assert paths and all(Path(p).exists() for p in paths)


def test_regulon_umap_writes_files(tmp_path: Path) -> None:
    auc = _auc()
    rng = np.random.default_rng(1)
    umap = pd.DataFrame(rng.random((auc.shape[0], 2)), index=auc.index, columns=["UMAP1", "UMAP2"])
    paths = rf.plot_regulon_umap(auc, umap, tmp_path, groups=_groups(auc), top_n=4)
    assert paths and all(Path(p).exists() for p in paths)


def test_cell_clustermap_writes_files(tmp_path: Path) -> None:
    auc = _auc()
    ann = _groups(auc).to_frame("group")
    paths = rf.plot_regulon_cell_clustermap(auc, ann, tmp_path, top_n=3)
    assert paths and all(Path(p).exists() for p in paths)


def test_plots_return_empty_on_empty_auc(tmp_path: Path) -> None:
    empty = pd.DataFrame()
    assert rf.plot_rss_panels(empty, pd.Series(dtype=str), tmp_path) == []
    assert rf.plot_regulon_clustermap(empty, pd.Series(dtype=str), tmp_path) == []


def test_no_theme_import() -> None:
    src = (
        Path(__file__).resolve().parents[1] / "src/cellquorum/grn/regulon_figures.py"
    ).read_text()
    assert "crrt" not in src
    assert "get_stage_colors_for_cancer" not in src
    assert "plot_rss_vs_rcond" not in src
