"""Tests for the house-styled SCENIC regulon figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.stages.gene_regulation.grn import regulon_figures as rf


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


def test_rss_orders_numeric_cluster_ids_numerically_not_lexicographically() -> None:
    """Leiden/CHOIR group labels are numeric-looking strings ("0".."15"); a plain
    string sort puts "10" before "2", scrambling every downstream figure's column
    order. Real bug, confirmed in a live GRN dotplot before this fix."""
    n = 20
    auc = _auc(n_cells=n)
    labels = [str(i % 12) for i in range(n)]  # cycles through "0".."11"
    groups = pd.Series(labels, index=auc.index)
    rss = rf.regulon_specificity_scores(auc, groups)
    assert list(rss.index) == [str(i) for i in range(12)]


def test_annotation_palette_orders_numeric_categories_numerically() -> None:
    cats = ["10", "2", "1", "0", "11"]
    pal = rf._annotation_palette("cluster", cats)
    assert list(pal.keys()) == ["0", "1", "2", "10", "11"]


def test_rss_panels_writes_png_and_pdf(tmp_path: Path) -> None:
    auc = _auc()
    paths = rf.plot_rss_panels(auc, _groups(auc), tmp_path, top_n=3)
    suffixes = {Path(p).suffix for p in paths}
    assert {".png", ".pdf"} <= suffixes


def test_regulon_clustermap_writes_files(tmp_path: Path) -> None:
    auc = _auc()
    paths = rf.plot_regulon_clustermap(auc, _groups(auc), tmp_path, top_n=3)
    assert paths and all(Path(p).exists() for p in paths)


def test_regulon_clustermap_cluster_labels_rename_yticklabels(tmp_path: Path, monkeypatch) -> None:
    """Bare cluster ids ("0", "1", ...) on the y-axis are meaningless without

    cross-referencing -- confirmed live: the real fibroblast GRN clustermap
    shows raw fibroblast_subcluster ids (0-6), not names, unlike every
    other stage's figures. cluster_labels renames the y-axis text only.
    """
    import seaborn as sns

    captured = {}
    real_clustermap = sns.clustermap

    def spy(data, **kwargs):
        captured["index"] = list(data.index)
        return real_clustermap(data, **kwargs)

    monkeypatch.setattr(sns, "clustermap", spy)
    auc = _auc()
    rf.plot_regulon_clustermap(
        auc, _groups(auc), tmp_path, top_n=3, cluster_labels={"A": "Papillary", "B": "Reticular"}
    )
    assert set(captured["index"]) == {"Papillary", "Reticular"}


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


def test_cell_clustermap_cluster_labels_rename_the_annotation_legend(
    tmp_path: Path, monkeypatch
) -> None:
    """The row-color legend's swatch labels come straight from the raw

    annotation values -- same bare-id problem as the clustermap y-axis,
    just one file over (the companion *_legend_<col>.pdf).
    """
    captured = {}
    real_palette = rf._annotation_palette

    def spy(col, cats):
        captured[col] = cats
        return real_palette(col, cats)

    monkeypatch.setattr(rf, "_annotation_palette", spy)
    auc = _auc()
    ann = _groups(auc).to_frame("group")
    rf.plot_regulon_cell_clustermap(
        auc,
        ann,
        tmp_path,
        top_n=3,
        cluster_labels={"A": "Papillary", "B": "Reticular"},
        cluster_col="group",
    )
    assert set(captured["group"]) == {"Papillary", "Reticular"}


def test_rss_panels_cluster_labels_rename_titles(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.axes

    titles = []
    real_set_title = matplotlib.axes.Axes.set_title

    def spy(self, label, *a, **kw):
        titles.append(label)
        return real_set_title(self, label, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", spy)
    auc = _auc()
    rf.plot_rss_panels(
        auc, _groups(auc), tmp_path, top_n=3, cluster_labels={"A": "Papillary", "B": "Reticular"}
    )
    assert any("Papillary" in t for t in titles)
    assert any("Reticular" in t for t in titles)
    assert not any(t.endswith("= A") or t.endswith("= B") for t in titles)


def test_plots_return_empty_on_empty_auc(tmp_path: Path) -> None:
    empty = pd.DataFrame()
    assert rf.plot_rss_panels(empty, pd.Series(dtype=str), tmp_path) == []
    assert rf.plot_regulon_clustermap(empty, pd.Series(dtype=str), tmp_path) == []


def test_no_theme_import() -> None:
    src = (
        Path(__file__).resolve().parents[1]
        / "src/cellquorum/stages/gene_regulation/grn/regulon_figures.py"
    ).read_text()
    assert "crrt" not in src
    assert "get_stage_colors_for_cancer" not in src
    assert "plot_rss_vs_rcond" not in src
