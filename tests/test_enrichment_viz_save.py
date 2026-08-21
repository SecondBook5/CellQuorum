"""Tests for the enrichment-viz house-style save helper."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cellquorum.comparative.enrichment.viz.io import apply_theme, figure_artifacts, save_figure


def test_save_figure_creates_parent_dirs_and_all_formats(tmp_path):
    out_dir = tmp_path / "figures" / "enrichment"  # does NOT exist yet
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, out_dir, "gsea_diverging_hallmark", formats=("pdf", "png"), dpi=150)
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert all(p.exists() for p in paths)
    assert paths[0].name == "gsea_diverging_hallmark.pdf"


def test_save_figure_closes_figure(tmp_path):
    fig, ax = plt.subplots()
    n_before = len(plt.get_fignums())
    save_figure(fig, tmp_path, "x", formats=("png",))
    assert len(plt.get_fignums()) == n_before - 1


def test_figure_artifacts_are_tagged_figure(tmp_path):
    p = tmp_path / "a.png"
    p.write_text("x")
    arts = figure_artifacts([p], name="enrichment_figure", description="test")
    assert len(arts) == 1
    assert arts[0].kind == "figure"
    assert arts[0].path == p


def test_apply_theme_sets_svg_fonttype_none():
    import matplotlib as mpl

    apply_theme()
    assert mpl.rcParams["svg.fonttype"] == "none"
    assert mpl.rcParams["pdf.fonttype"] == 42
