"""Reference-gallery fixtures for the in-silico perturbation figures."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from _gallery_io import via_files
from matplotlib.figure import Figure

from cellquorum.stages.gene_regulation.perturbation import perturbation_figures as pf
from cellquorum.visualization.figstyle import apply_theme

LABELS = {"0": "Perivascular", "1": "Mechanosensory", "2": "Reticular"}


def _embedding_and_shift() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Three clusters on a 2-D embedding with a knock-out shift pointing toward cluster 0."""
    rng = np.random.default_rng(0)
    centers = {"0": (-3.0, 0.0), "1": (3.0, 1.0), "2": (0.0, -3.0)}
    groups = pd.Series(np.repeat(list(centers), 100), index=[f"c{i}" for i in range(300)])
    coords = np.array([rng.normal(centers[g], 0.8) for g in groups])
    toward = np.array(centers["0"]) - coords
    toward /= np.linalg.norm(toward, axis=1, keepdims=True)
    shift = toward * 0.25 + rng.normal(0, 0.03, coords.shape)
    embedding = pd.DataFrame(coords, index=groups.index, columns=["DIM1", "DIM2"])
    return embedding, pd.DataFrame(shift, index=groups.index, columns=["d0", "d1"]), groups


def target_ranking() -> Path:
    ranking = pd.DataFrame(
        {
            "tf": [f"TF{i}" for i in range(25)],
            "score": np.linspace(1.0, 0.1, 25),
            "n_cells": 100,
            "direction": "directional",
        }
    )
    return via_files(lambda out: pf.plot_target_ranking(ranking, out, n_top=15))()


def shift_field() -> Path:
    embedding, shift, groups = _embedding_and_shift()
    return via_files(
        lambda out: pf.plot_ko_shift_field(
            shift, embedding, out, tf="PROX1", groups=groups, group_labels=LABELS
        )
    )()


def shift_grid() -> Path:
    embedding, shift, groups = _embedding_and_shift()
    return via_files(
        lambda out: pf.plot_ko_shift_grid(
            shift, embedding, out, tf="PROX1", groups=groups, group_labels=LABELS, n_grid=14
        )
    )()


def fate_summary() -> Path:
    fate = pd.DataFrame(
        {
            "cluster": [f"C{i}" for i in range(8)],
            "delta": [0.18, 0.09, 0.02, -0.04, -0.11, -0.19, 0.05, -0.02],
        }
    )
    return via_files(lambda out: pf.plot_ko_fate_summary(fate, out, tf="PROX1"))()


def grn_connectivity() -> Path:
    tfs = [f"TF{i}" for i in range(30)]
    grn = pd.DataFrame(
        {
            "cluster": ["c0"] * 30 + ["c1"] * 30,
            "tf": tfs + tfs,
            "n_targets": list(range(30)) + list(range(29, -1, -1)),
        }
    )
    return via_files(lambda out: pf.plot_grn_connectivity(grn, out, n_top=15))()


THEME = partial(apply_theme, "compact")

FIGURES: dict[str, Callable[[], Figure | Path]] = {
    "perturbation_target_ranking": target_ranking,
    "perturbation_shift_field": shift_field,
    "perturbation_shift_grid": shift_grid,
    "perturbation_fate_summary": fate_summary,
    "perturbation_grn_connectivity": grn_connectivity,
}
