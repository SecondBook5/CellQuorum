"""Reference-gallery fixtures for the annotation-diagnostic and clustering-resolution figures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from _gallery_io import via_files
from matplotlib.figure import Figure

from cellquorum.stages.annotation.diagnostics import viz as annotation_viz
from cellquorum.stages.clustering.resolution_diagnostic_viz import (
    write_resolution_diagnostic_figure,
)
from cellquorum.visualization.figstyle import apply_cellquorum_theme

CELL_TYPES = ["Fibroblasts", "LEC", "Pericytes", "T/NK", "Macrophages", "Keratinocytes"]


def _obs() -> pd.DataFrame:
    """Per-cell scDiagnostics columns; the rarer types are the less confident ones."""
    rng = np.random.default_rng(0)
    sizes = [400, 250, 150, 120, 80, 40]
    rows = []
    for cell_type, n in zip(CELL_TYPES, sizes, strict=True):
        looseness = 1.0 - n / 500
        anomaly = np.clip(rng.normal(0.15 + 0.4 * looseness, 0.12, n), 0, 1)
        confidence = float(np.clip(0.95 - 0.5 * looseness, 0.2, 1.0))
        rows.append(
            pd.DataFrame(
                {
                    "cell_type": cell_type,
                    "scdiag_anomaly": anomaly,
                    "scdiag_anomaly_flag": anomaly > 0.5,
                    "scdiag_entropy": np.clip(rng.normal(0.8 - confidence * 0.7, 0.1, n), 0, 1),
                    "scdiag_knn_prob_group": confidence,
                    "scdiag_knn_prob_group_n": n,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def anomaly() -> Path:
    return via_files(
        lambda out: [annotation_viz.plot_anomaly_by_cell_type(_obs(), "cell_type", out / "a.png")]
    )()


def entropy() -> Path:
    return via_files(
        lambda out: [annotation_viz.plot_entropy_by_cell_type(_obs(), "cell_type", out / "e.png")]
    )()


def knn_confidence() -> Path:
    return via_files(
        lambda out: [annotation_viz.plot_knn_confidence(_obs(), "cell_type", out / "k.png")]
    )()


def resolution_diagnostic() -> Path:
    summary = pd.DataFrame(
        {
            "resolution": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
            "n_clusters": [2, 3, 4, 5, 7, 9, 12],
            "median_jaccard": [0.97, 0.95, 0.93, 0.88, 0.55, 0.40, 0.31],
            "q25": [0.94, 0.9, 0.88, 0.8, 0.4, 0.3, 0.2],
            "q75": [0.99, 0.98, 0.97, 0.93, 0.7, 0.5, 0.42],
            "n_bootstrap_observations": 40,
        }
    )
    return via_files(
        lambda out: [write_resolution_diagnostic_figure(summary, out / "resolution.png")]
    )()


THEME = apply_cellquorum_theme

FIGURES: dict[str, Callable[[], Figure | Path]] = {
    "annotation_anomaly_by_cell_type": anomaly,
    "annotation_entropy_by_cell_type": entropy,
    "annotation_knn_confidence": knn_confidence,
    "clustering_resolution_diagnostic": resolution_diagnostic,
}
