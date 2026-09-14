"""Tests for per-coarse-type granular (fine-label) embedding panels.

One figure per coarse cell type showing its OWN fine subtypes, not one
crowded global map of all ~80 fine labels at once. A coarse type whose fine
label never differs from its coarse label (no real subdivision in the
reference atlas) is skipped, not drawn as a flat single-color panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.visualization.granular_panels import plot_granular_panels_per_coarse_type


def _coords(n: int) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, 2))


def test_writes_one_panel_per_coarse_type_with_real_subdivision(tmp_path):
    n = 12
    coarse = pd.Series((["Fibroblasts"] * 6) + (["LEC"] * 6))
    fine = pd.Series((["Fibroblast_A"] * 3 + ["Fibroblast_B"] * 3) + (["LEC"] * 6))

    written = plot_granular_panels_per_coarse_type(_coords(n), fine, coarse, out_dir=tmp_path)

    # Fibroblasts has 2 fine labels -> gets a panel. LEC has 1 (== its own
    # coarse label) -> no real subdivision, skipped.
    assert set(written) == {"Fibroblasts"}
    assert written["Fibroblasts"].exists()
    assert written["Fibroblasts"].with_suffix(".pdf").exists()


def test_returns_empty_when_no_coarse_type_has_subdivision(tmp_path):
    n = 6
    coarse = pd.Series(["LEC"] * 3 + ["Neutrophils"] * 3)
    fine = pd.Series(["LEC"] * 3 + ["Neutrophils"] * 3)

    written = plot_granular_panels_per_coarse_type(_coords(n), fine, coarse, out_dir=tmp_path)

    assert written == {}


def test_no_resolution_types_are_reported(tmp_path):
    n = 12
    coarse = pd.Series((["Fibroblasts"] * 6) + (["LEC"] * 6))
    fine = pd.Series((["Fibroblast_A"] * 3 + ["Fibroblast_B"] * 3) + (["LEC"] * 6))

    written, no_resolution = plot_granular_panels_per_coarse_type(
        _coords(n), fine, coarse, out_dir=tmp_path, return_skipped=True
    )

    assert no_resolution == ["LEC"]
