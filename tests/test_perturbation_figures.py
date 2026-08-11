"""Tests for perturbation figures (synthetic input; PNG+PDF; empty -> [])."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.perturbation import perturbation_figures as pf


def _ranking(n: int = 25) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tf": [f"TF{i}" for i in range(n)],
            "score": np.linspace(1.0, 0.1, n),
            "n_cells": 100,
            "direction": "directional",
        }
    )


def test_target_ranking_writes_png_and_pdf(tmp_path: Path) -> None:
    paths = pf.plot_target_ranking(_ranking(), tmp_path, n_top=10)
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes == [".pdf", ".png"]
    assert all(p.exists() for p in paths)


def test_target_ranking_empty_returns_empty(tmp_path: Path) -> None:
    assert pf.plot_target_ranking(pd.DataFrame(columns=["tf", "score"]), tmp_path) == []


def test_shift_field_writes_and_aligns(tmp_path: Path) -> None:
    idx = [f"c{i}" for i in range(50)]
    shift = pd.DataFrame({"dx": np.random.rand(50), "dy": np.random.rand(50)}, index=idx)
    emb = pd.DataFrame({"DIM1": np.random.rand(50), "DIM2": np.random.rand(50)}, index=idx)
    paths = pf.plot_ko_shift_field(shift, emb, tmp_path, tf="PROX1")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]


def test_shift_field_no_overlap_returns_empty(tmp_path: Path) -> None:
    shift = pd.DataFrame({"dx": [1.0], "dy": [1.0]}, index=["a"])
    emb = pd.DataFrame({"DIM1": [1.0], "DIM2": [1.0]}, index=["b"])
    assert pf.plot_ko_shift_field(shift, emb, tmp_path, tf="X") == []


def test_fate_summary_writes(tmp_path: Path) -> None:
    fate = pd.DataFrame({"cluster": ["A", "B", "C"], "delta": [0.1, -0.2, 0.05]})
    paths = pf.plot_ko_fate_summary(fate, tmp_path, tf="PROX1")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]


def test_grn_connectivity_writes(tmp_path: Path) -> None:
    grn = pd.DataFrame({"tf": [f"TF{i}" for i in range(30)], "degree": range(30)})
    paths = pf.plot_grn_connectivity(grn, tmp_path, n_top=15)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]


def test_grn_connectivity_empty_returns_empty(tmp_path: Path) -> None:
    assert pf.plot_grn_connectivity(pd.DataFrame(columns=["tf", "degree"]), tmp_path) == []
