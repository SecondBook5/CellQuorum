"""Tests for PCA knee-based component selection."""

from __future__ import annotations

import numpy as np

from cellquorum.preprocessing.dimensionality.knee import select_n_pcs


def test_selects_elbow_on_clear_knee():
    # Sharp decay then flat tail: knee should land in the low single digits.
    vr = np.array([0.5, 0.25, 0.12, 0.02, 0.015, 0.012, 0.011, 0.010], dtype=float)
    n = select_n_pcs(vr, max_pcs=50)
    assert 1 <= n <= 5


def test_respects_max_pcs_cap():
    vr = np.linspace(0.1, 0.01, num=100)
    n = select_n_pcs(vr, max_pcs=10)
    assert n <= 10


def test_returns_at_least_one():
    vr = np.array([0.9, 0.1], dtype=float)
    assert select_n_pcs(vr, max_pcs=50) >= 1


def test_flat_curve_falls_back_to_cap():
    # No discernible knee -> fall back to min(len, max_pcs).
    vr = np.full(20, 0.05, dtype=float)
    assert select_n_pcs(vr, max_pcs=15) == 15
