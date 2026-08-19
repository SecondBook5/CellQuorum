"""Tests for perturbation figures (synthetic input; PNG+PDF; empty -> [])."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.gene_regulation.perturbation import perturbation_figures as pf


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
    np.random.seed(0)
    idx = [f"c{i}" for i in range(50)]
    # Producer's real schema: d0/d1, not dx/dy (proves positional read of FIX 1)
    shift = pd.DataFrame({"d0": np.random.rand(50), "d1": np.random.rand(50)}, index=idx)
    emb = pd.DataFrame({"DIM1": np.random.rand(50), "DIM2": np.random.rand(50)}, index=idx)
    paths = pf.plot_ko_shift_field(shift, emb, tmp_path, tf="PROX1")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    assert all(p.exists() for p in paths)


def test_shift_field_no_overlap_returns_empty(tmp_path: Path) -> None:
    shift = pd.DataFrame({"d0": [1.0], "d1": [1.0]}, index=["a"])
    emb = pd.DataFrame({"DIM1": [1.0], "DIM2": [1.0]}, index=["b"])
    assert pf.plot_ko_shift_field(shift, emb, tmp_path, tf="X") == []


def test_shift_grid_writes_and_masks(tmp_path: Path) -> None:
    np.random.seed(1)
    idx = [f"c{i}" for i in range(200)]
    shift = pd.DataFrame(
        {"d0": np.random.randn(200) * 0.1, "d1": np.random.randn(200) * 0.1}, index=idx
    )
    emb = pd.DataFrame({"DIM1": np.random.rand(200), "DIM2": np.random.rand(200)}, index=idx)
    groups = pd.Series(np.random.choice(["A", "B"], 200), index=idx)
    paths = pf.plot_ko_shift_grid(shift, emb, tmp_path, tf="PROX1", groups=groups, n_grid=12)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    assert all(p.exists() for p in paths)


def test_shift_grid_no_overlap_returns_empty(tmp_path: Path) -> None:
    shift = pd.DataFrame({"d0": [1.0], "d1": [1.0]}, index=["a"])
    emb = pd.DataFrame({"DIM1": [1.0], "DIM2": [1.0]}, index=["b"])
    assert pf.plot_ko_shift_grid(shift, emb, tmp_path, tf="X") == []


def test_fate_summary_writes(tmp_path: Path) -> None:
    fate = pd.DataFrame({"cluster": ["A", "B", "C"], "delta": [0.1, -0.2, 0.05]})
    paths = pf.plot_ko_fate_summary(fate, tmp_path, tf="PROX1")
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]


def test_grn_connectivity_writes(tmp_path: Path) -> None:
    # Producer's real per-cluster schema: cluster/tf/n_targets (multiple clusters per
    # TF so the groupby-sum in FIX 2 is exercised)
    tfs = [f"TF{i}" for i in range(30)]
    grn = pd.DataFrame(
        {
            "cluster": ["c0"] * 30 + ["c1"] * 30,
            "tf": tfs + tfs,
            "n_targets": list(range(30)) + list(range(30)),
        }
    )
    paths = pf.plot_grn_connectivity(grn, tmp_path, n_top=15)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    assert all(p.exists() for p in paths)


def test_grn_connectivity_empty_returns_empty(tmp_path: Path) -> None:
    empty = pd.DataFrame(columns=["cluster", "tf", "n_targets"])
    assert pf.plot_grn_connectivity(empty, tmp_path) == []


def test_grn_connectivity_accepts_degree_schema(tmp_path: Path) -> None:
    # Backward-compat: the old consumer `degree` schema must still render (FIX 2)
    grn = pd.DataFrame({"tf": [f"TF{i}" for i in range(30)], "degree": range(30)})
    paths = pf.plot_grn_connectivity(grn, tmp_path, n_top=15)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    assert all(p.exists() for p in paths)
