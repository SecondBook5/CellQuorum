"""Reference-gallery fixtures for the trajectory figures.

Each entry builds one figure from small synthetic inputs with a fixed seed, so the
drawing is identical on every run and any change to the figure code shows up as a
pixel difference against the stored baseline in ``docs/gallery/assets``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.stages.trajectory.viz import _helpers as helpers
from cellquorum.stages.trajectory.viz.activity_cascade import cascade_heatmap
from cellquorum.stages.trajectory.viz.heatmap import condition_split_heatmap


def pseudotime_scatter() -> Figure:
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(400, 2))
    values = np.clip(np.abs(coords[:, 0]) * 0.02, 0, 0.03)
    values[:3] = 1.0  # outlier cells the robust color scale must not stretch for
    return helpers.embedding_scatter(
        coords, values, title="Synthetic pseudotime", cbar_label="pseudotime"
    )


def driver_bar() -> Figure:
    genes = [f"Gene{i}" for i in range(12)]
    values = np.linspace(-0.8, 0.9, 12)
    return helpers.signed_diverging_bar(genes, values, title="Synthetic fate drivers")


def driver_heatmap() -> Figure:
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(10, 3))
    rows = [f"Gene{i}" for i in range(10)]
    return helpers.matrix_heatmap(
        matrix,
        rows,
        ["Fate A", "Fate B", "Fate C"],
        title="Synthetic driver correlation",
        cbar_label="correlation",
        row_group_breaks=[4],
    )


def transition_matrix() -> Figure:
    names = ["Basal", "Transit", "Mature", "Stalled"]
    matrix = np.array(
        [
            [0.80, 0.15, 0.03, 0.02],
            [0.10, 0.70, 0.15, 0.05],
            [0.02, 0.08, 0.88, 0.02],
            [0.05, 0.10, 0.05, 0.80],
        ]
    )
    stationary = pd.Series([0.15, 0.25, 0.45, 0.15], index=names)
    return helpers.coarse_transition_heatmap(
        pd.DataFrame(matrix, index=names, columns=names),
        stationary,
        title="Synthetic coarse transition matrix",
    )


def fate_violin() -> Figure:
    groups = {str(key): np.random.default_rng(10 + key).beta(2 + key, 3, 60) for key in (0, 1, 4)}
    return helpers.grouped_violin(
        groups,
        title="Synthetic fate probability",
        ylabel="probability",
        labels={"0": "Basal", "1": "Transit", "4": "Mature"},
    )


def pseudotime_heatmap() -> Figure:
    n_bins, n_genes = 20, 8
    ramp = np.linspace(0, 1, n_bins)
    peaks = np.linspace(0.1, 0.9, n_genes)
    base = np.exp(-((ramp[:, None] - peaks[None, :]) ** 2) / 0.02)
    profiles = {"Normal": base, "Disease": np.roll(base, 3, axis=0) * 0.8}
    state = (ramp > 0.5).astype(int)
    tracks = [
        {
            "label": "State",
            "cats": ["Basal", "Mature"],
            "colors": ["#4C72B0", "#DD8452"],
            "tracks": {"Normal": state, "Disease": state},
            "present_codes": [0, 1],
        }
    ]
    return condition_split_heatmap(
        profiles,
        {"Normal": ramp, "Disease": ramp * 0.9},
        tracks,
        [f"Gene{i}" for i in range(n_genes)],
        np.arange(n_genes),
        condition_order=["Normal", "Disease"],
        title="Synthetic pseudotime heatmap",
    )


def activity_cascade() -> Figure:
    rng = np.random.default_rng(3)
    pseudotime = np.sort(rng.uniform(0, 1, 300))
    peaks = np.linspace(0.1, 0.9, 10)
    activity = pd.DataFrame(
        {
            f"HALLMARK_PATHWAY_{i}": np.exp(-((pseudotime - peak) ** 2) / 0.03)
            + rng.normal(0, 0.05, 300)
            for i, peak in enumerate(peaks)
        }
    )
    fig = cascade_heatmap(
        activity, pseudotime, n_bins=20, title="Synthetic activity cascade", xlab="dpt"
    )
    if fig is None:
        raise RuntimeError("Synthetic cascade fixture produced no associated sources")
    return fig


def _themed(build: Callable[[], Figure]) -> Callable[[], Figure]:
    """Apply the trajectory stage's own theme first, as the real pipeline does."""

    def themed() -> Figure:
        helpers.apply_theme()
        return build()

    return themed


FIGURES: dict[str, Callable[[], Figure]] = {
    "trajectory_pseudotime_scatter": _themed(pseudotime_scatter),
    "trajectory_driver_bar": _themed(driver_bar),
    "trajectory_driver_heatmap": _themed(driver_heatmap),
    "trajectory_transition_matrix": _themed(transition_matrix),
    "trajectory_fate_violin": _themed(fate_violin),
    "trajectory_pseudotime_heatmap": _themed(pseudotime_heatmap),
    "trajectory_activity_cascade": _themed(activity_cascade),
}
