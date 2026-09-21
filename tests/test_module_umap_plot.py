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


def test_legend_renders_entirely_outside_the_axes_data_area(tmp_path: Path) -> None:
    """A legend drawn at loc="upper left" (inside the axes) sits on top of

    whatever data happens to be in that corner -- module points are not
    evenly distributed, so this reliably covered real points for whichever
    module landed there (confirmed live: the real fibroblast module UMAP
    had a module's own cluster peeking out from under the legend box). The
    legend must be anchored OUTSIDE the axes entirely, so it is structurally
    impossible for it to overlap a data point regardless of layout -- this
    also makes the earlier "push a label off the legend" mechanism
    unnecessary, since a hub label (always inside the data's own coordinate
    range) and a legend outside it can never occupy the same region.
    """
    import matplotlib.pyplot as plt

    from cellquorum.stages.gene_regulation.coexpression.module_umap_plot import (
        _build_figure,
    )
    from cellquorum.visualization import figstyle

    df = pd.DataFrame(
        {
            "gene": ["CCN5", "COL23A1", "MFAP5", "NFKB1", "SOD2", "NAMPT", "OTHER1", "OTHER2"],
            # First three hub genes sit right where loc="upper left" would
            # have drawn an inside legend, to prove the outside placement
            # holds even in the worst-case layout.
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
    axes_bbox = fig.axes[0].get_window_extent(renderer)

    assert not legend_bbox.overlaps(
        axes_bbox
    ), "legend overlaps the axes' own data area -- it must render fully outside it"
    # And since the legend can't touch the data area at all, no hub label
    # (which only ever renders inside it) can overlap the legend either.
    overlapping = [
        t.get_text() for t in labels if t.get_window_extent(renderer).overlaps(legend_bbox)
    ]
    assert overlapping == [], f"hub label(s) rendered on top of the legend: {overlapping}"
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
