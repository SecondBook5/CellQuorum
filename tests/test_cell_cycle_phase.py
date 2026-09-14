"""TDD tests for cellquorum.stats.cell_cycle_phase.

Derives a categorical G1/S/G2M call from S/G2M scores, matching scanpy's own
score_genes_cell_cycle convention: both scores <= 0 -> G1, else the higher
of the two wins.
"""

from __future__ import annotations

import pandas as pd

from cellquorum.stats.cell_cycle_phase import assign_cell_cycle_phase


def test_both_scores_nonpositive_is_g1():
    s = pd.Series([0.0, -0.2], index=["a", "b"])
    g2m = pd.Series([-0.1, 0.0], index=["a", "b"])

    phase = assign_cell_cycle_phase(s, g2m)

    assert phase.tolist() == ["G1", "G1"]


def test_s_score_dominant_is_s_phase():
    s = pd.Series([0.5], index=["a"])
    g2m = pd.Series([0.1], index=["a"])

    phase = assign_cell_cycle_phase(s, g2m)

    assert phase.tolist() == ["S"]


def test_g2m_score_dominant_is_g2m_phase():
    s = pd.Series([0.1], index=["a"])
    g2m = pd.Series([0.5], index=["a"])

    phase = assign_cell_cycle_phase(s, g2m)

    assert phase.tolist() == ["G2M"]


def test_phase_is_indexed_like_the_input_scores():
    s = pd.Series([0.1, 0.2], index=["cell_x", "cell_y"])
    g2m = pd.Series([0.0, 0.0], index=["cell_x", "cell_y"])

    phase = assign_cell_cycle_phase(s, g2m)

    assert list(phase.index) == ["cell_x", "cell_y"]
    assert phase.name == "phase"
