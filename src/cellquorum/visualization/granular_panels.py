"""Per-coarse-type fine-label (granular) embedding panels.

One figure per coarse cell type, colored by ITS OWN fine subtypes -- not one
global map with ~80 fine labels competing for the same small categorical
palette. A coarse type whose fine label never differs from its own coarse
label (no real subdivision exists in the reference atlas for it) is skipped
rather than drawn as a single flat color, and reported separately so the gap
itself is visible instead of silently absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cellquorum.stages.integration.embeddings.plots import apply_theme
from cellquorum.visualization.figstyle import cell_type_palette, save_figure

if TYPE_CHECKING:
    import pandas as pd


def plot_granular_panels_per_coarse_type(
    coords: np.ndarray,
    fine_labels: pd.Series,
    coarse_labels: pd.Series,
    *,
    out_dir: Path,
    return_skipped: bool = False,
    point_size: float = 2.0,
) -> dict[str, Path] | tuple[dict[str, Path], list[str]]:
    """
    Plot one embedding panel per coarse type, colored by its own fine subtypes.

    Args:
        coords: Cells x 2 embedding coordinates for the FULL cohort.
        fine_labels: Per-cell fine label (e.g. ``cell_type_granular``), same
            length as ``coords``.
        coarse_labels: Per-cell coarse label (e.g.
            ``cell_type_granular_coarse``), same length as ``coords``.
        out_dir: Directory to write figures into (created if needed).
        return_skipped: When True, also return the list of coarse types
            skipped for having no real fine subdivision.
        point_size: Marker size for the foreground (in-type) scatter.

    Returns:
        ``{coarse_type: png_path}`` for every coarse type with more than one
        fine label. When ``return_skipped`` is True, returns
        ``(written, no_resolution)`` instead, where ``no_resolution`` lists
        coarse types skipped for lacking real subdivision (sorted).
    """
    import pandas as pd

    out_dir = Path(out_dir)
    coords = np.asarray(coords)[:, :2]
    fine_labels = pd.Series(np.asarray(fine_labels), name="fine")
    coarse_labels = pd.Series(np.asarray(coarse_labels), name="coarse")

    written: dict[str, Path] = {}
    no_resolution: list[str] = []

    for coarse_type in sorted(coarse_labels.unique()):
        mask = (coarse_labels == coarse_type).to_numpy()
        fine_in_type = fine_labels[mask]
        distinct = sorted(fine_in_type.unique())
        if len(distinct) <= 1:
            no_resolution.append(coarse_type)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        apply_theme()

        import matplotlib.pyplot as plt

        palette = cell_type_palette(distinct)
        fig, ax = plt.subplots(figsize=(6.5, 6.0))

        # Grey context: every other cell, so the panel still reads as "where
        # in the atlas is this" rather than a population floating in a void.
        ax.scatter(
            coords[~mask, 0],
            coords[~mask, 1],
            c="#e8e9eb",
            s=point_size * 0.6,
            lw=0,
            rasterized=True,
        )
        type_coords = coords[mask]
        for label in distinct:
            label_mask = (fine_in_type == label).to_numpy()
            ax.scatter(
                type_coords[label_mask, 0],
                type_coords[label_mask, 1],
                c=[palette[label]],
                s=point_size,
                lw=0,
                rasterized=True,
                label=f"{label} ({int(label_mask.sum()):,})",
            )

        # Percentile bounding box, not true min/max: a handful of far-flung
        # outlier cells (often real signal -- e.g. cells whose expression
        # doesn't match their label) must not blow out the zoom for everyone
        # else. Outliers still plot; they just may sit outside the crop.
        x_lo, x_hi = np.percentile(type_coords[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(type_coords[:, 1], [1, 99])
        pad = 0.08 * (x_hi - x_lo + 1e-9)
        ax.set_xlim(x_lo - pad, x_hi + pad)
        pad_y = 0.08 * (y_hi - y_lo + 1e-9)
        ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{coarse_type} -- fine subtypes", fontsize=11, weight="bold")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

        safe_name = coarse_type.replace("/", "_").replace(" ", "_")
        paths = save_figure(fig, out_dir, f"granular_{safe_name}")
        written[coarse_type] = next(p for p in paths if p.suffix == ".png")

    if return_skipped:
        return written, sorted(no_resolution)
    return written


__all__ = ["plot_granular_panels_per_coarse_type"]
