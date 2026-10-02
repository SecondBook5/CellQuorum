"""Per-coarse-type fine-label (granular) embedding panels.

One figure per coarse cell type, colored by ITS OWN fine subtypes -- not one
global map with ~80 fine labels competing for the same small categorical
palette. A coarse type whose fine label never differs from its own coarse
label (no real subdivision exists in the reference atlas for it) is skipped
rather than drawn as a single flat color, and reported separately so the gap
itself is visible instead of silently absent.

Reuses the embeddings stage's own categorical_embedding (house style, PAGA
overlay, count-labeled legend) rather than a bespoke scatter loop -- a
local PAGA computed on just this coarse type's own cells shows whether its
fine subtypes are genuinely continuous/connected or discrete, the same way
the global atlas's own PAGA does at the coarse level.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cellquorum.stages.integration.embeddings.plots import categorical_embedding
from cellquorum.visualization.figstyle import apply_theme, save_figure

if TYPE_CHECKING:
    import pandas as pd


def _robust_bounds(values: np.ndarray, k: float = 6.0) -> tuple[float, float]:
    """Median +/- k*MAD -- stays tight even with ~10%+ outlier contamination.

    A handful of far-flung cells (often real signal -- e.g. cells whose
    expression doesn't match their label) must not blow out the crop for
    everyone else. A percentile bound is not aggressive enough once the
    outlier fraction is much above 1%; MAD's breakdown point is 50%.
    """
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center))) or float(np.std(values)) or 1.0
    return center - k * mad, center + k * mad


def plot_granular_panels_per_coarse_type(
    coords: np.ndarray,
    fine_labels: pd.Series,
    coarse_labels: pd.Series,
    *,
    out_dir: Path,
    return_skipped: bool = False,
    paga_overlay: bool = True,
    paga_threshold: float = 0.2,
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
        paga_overlay: Whether to compute and draw PAGA connectivity between
            this coarse type's own fine subtypes.
        paga_threshold: Minimum connectivity for a drawn PAGA edge.

    Returns:
        ``{coarse_type: png_path}`` for every coarse type with more than one
        fine label. When ``return_skipped`` is True, returns
        ``(written, no_resolution)`` instead, where ``no_resolution`` lists
        coarse types skipped for lacking real subdivision (sorted).
    """
    import anndata as ad
    import pandas as pd
    import scanpy as sc

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
        apply_theme("notebook")

        sub = ad.AnnData(
            X=np.zeros((int(mask.sum()), 1), dtype="float32"),
            obs=pd.DataFrame(
                {"fine": pd.Categorical(fine_in_type.to_numpy(), categories=distinct)}
            ),
        )
        sub.obsm["X_panel"] = coords[mask]

        if paga_overlay and len(distinct) >= 2:
            try:
                sc.tl.paga(sub, groups="fine")
            except Exception:  # noqa: BLE001 -- PAGA is a bonus overlay, never fatal here
                pass

        fig = categorical_embedding(
            sub,
            "fine",
            basis="X_panel",
            axis_labels=("UMAP1", "UMAP2"),
            title=f"{coarse_type} -- fine subtypes",
            paga_overlay=paga_overlay,
            paga_threshold=paga_threshold,
        )
        ax = fig.axes[0]
        x_lo, x_hi = _robust_bounds(sub.obsm["X_panel"][:, 0])
        y_lo, y_hi = _robust_bounds(sub.obsm["X_panel"][:, 1])
        pad = 0.08 * (x_hi - x_lo + 1e-9)
        pad_y = 0.08 * (y_hi - y_lo + 1e-9)
        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.set_ylim(y_lo - pad_y, y_hi + pad_y)

        safe_name = coarse_type.replace("/", "_").replace(" ", "_")
        paths = save_figure(fig, out_dir, f"granular_{safe_name}")
        written[coarse_type] = next(p for p in paths if p.suffix == ".png")

    if return_skipped:
        return written, sorted(no_resolution)
    return written


__all__ = ["plot_granular_panels_per_coarse_type"]
