"""Two-panel figure for the resolution-stability diagnostic.

Read together: a resolution where cluster count jumps but stability collapses is
over-splitting; a plateau in stability across resolutions is a defensible operating
range. Neither panel alone tells that story -- cluster count says nothing about whether
a split is real, and stability alone says nothing about how fine the partition is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.visualization.figstyle import (
    CELLQUORUM_RED,
    NORMAL_BLUE,
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    save_cellquorum_figure,
)


def write_resolution_diagnostic_figure(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    dpi: int = 150,
) -> Path | None:
    """Render cluster count and bootstrap stability across the resolution sweep.

    Args:
        summary: As returned by
            :func:`cellquorum.stages.clustering.resolution_diagnostic.summarize_resolution_stability`.
        output_path: Destination PNG path; a vector twin is written beside it.
        dpi: Raster resolution.

    Returns:
        ``output_path`` on success, or None (writes nothing) when ``summary`` is empty.
    """
    if summary.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_cellquorum_theme()

    ordered = summary.sort_values("resolution")
    x = ordered["resolution"].to_numpy(dtype=float)

    fig, (ax_count, ax_stability) = plt.subplots(2, 1, figsize=(6.0, 6.4), sharex=True)

    ax_count.step(x, ordered["n_clusters"].to_numpy(dtype=float), where="mid", color=NORMAL_BLUE)
    ax_count.scatter(x, ordered["n_clusters"], color=NORMAL_BLUE, s=18, zorder=3)
    ax_count.set_ylabel("Clusters found")
    ax_count.set_title("Resolution stability diagnostic", fontweight="bold", pad=10)
    apply_cellquorum_axis_style(ax_count)

    median = ordered["median_jaccard"].to_numpy(dtype=float)
    q25 = ordered["q25"].to_numpy(dtype=float)
    q75 = ordered["q75"].to_numpy(dtype=float)

    ax_stability.fill_between(x, q25, q75, color=CELLQUORUM_RED, alpha=0.18, linewidth=0)
    ax_stability.plot(x, median, color=CELLQUORUM_RED, linewidth=1.1, marker="o", markersize=3.5)
    ax_stability.set_ylim(0.0, 1.02)
    ax_stability.set_xlabel("Leiden resolution")
    ax_stability.set_ylabel("Bootstrap stability (Jaccard)")
    apply_cellquorum_axis_style(ax_stability)

    fig.tight_layout()
    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    return output_path


__all__ = ["write_resolution_diagnostic_figure"]
