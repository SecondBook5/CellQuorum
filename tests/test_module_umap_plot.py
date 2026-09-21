# tests/test_module_umap_plot.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.gene_regulation.coexpression.module_umap_plot import (
    plot_module_umap,
)


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


def test_hub_labels_do_not_overlap_the_legend(tmp_path: Path) -> None:
    """The legend sits at loc="upper left" -- a hub gene whose UMAP coordinates

    land in that corner gets a label there too, and adjustText can only
    repel a label away from an artist that already exists on the axes when
    it runs. Built after the repulsion (the original order), the legend was
    invisible to adjustText and a real run showed labels landing directly
    under it. Regression: several hub genes placed right in the legend's
    corner, checked post-render.
    """
    import matplotlib.pyplot as plt

    from cellquorum.stages.gene_regulation.coexpression.module_umap_plot import (
        _build_figure,
    )
    from cellquorum.visualization import figstyle

    df = pd.DataFrame(
        {
            "gene": ["CCN5", "COL23A1", "MFAP5", "NFKB1", "SOD2", "NAMPT", "OTHER1", "OTHER2"],
            # First three hub genes sit right where loc="upper left" draws the
            # legend; the rest are spread out so the module/color plumbing
            # has real variety to work with.
            "UMAP1": [0.05, 0.08, 0.03, 5.0, 5.2, 5.1, 2.5, 2.6],
            "UMAP2": [4.9, 4.95, 4.85, 0.5, 0.6, 0.4, 2.5, 2.4],
            "module": ["M1", "M1", "M1", "M2", "M2", "M2", "M1", "M2"],
            "hub": ["hub", "hub", "hub", "hub", "hub", "hub", "other", "other"],
            "kME": [0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.4, 0.4],
        }
    )
    mods = ["M1", "M2"]
    mod_color = {"M1": figstyle.CATEGORICAL_PALETTE[0], "M2": figstyle.CATEGORICAL_PALETTE[1]}
    top_hub = {"M1": "CCN5", "M2": "NFKB1"}

    fig, leg, labels = _build_figure(df, mods=mods, mod_color=mod_color, top_hub=top_hub, tag="")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = leg.get_window_extent(renderer)

    overlapping = [
        t.get_text() for t in labels if t.get_window_extent(renderer).overlaps(legend_bbox)
    ]
    assert overlapping == [], f"hub label(s) rendered under the legend: {overlapping}"

    # The three pushed-away labels must not have landed on top of each other
    # either -- pushing them all to the identical spot would just trade one
    # overlap for another.
    boxes = [t.get_window_extent(renderer) for t in labels]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), (
                f"labels {labels[i].get_text()!r} and {labels[j].get_text()!r} overlap "
                "each other after being pushed off the legend"
            )
    plt.close("all")


def test_uses_categorical_palette(tmp_path: Path) -> None:
    from cellquorum.visualization import figstyle

    csv = tmp_path / "module_umap.csv"
    _write_csv(csv)
    # Should not raise and should complete using the house palette
    # (smoke: default palette is None → CATEGORICAL_PALETTE).
    paths = plot_module_umap(csv, tmp_path, palette=None)
    assert len(paths) == 2
    assert isinstance(figstyle.CATEGORICAL_PALETTE, list)
