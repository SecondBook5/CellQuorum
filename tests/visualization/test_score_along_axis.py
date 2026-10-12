"""A score along an ordering of cells, one smoothed curve per condition."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.score_along_axis import score_along_axis  # noqa: E402


def _cells(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    axis = rng.random(n)
    condition = np.where(rng.random(n) < 0.5, "Normal", "Lymphedema")
    # the score rises along the axis, and lymphedema cells sit higher at every position
    score = 2.0 * axis + np.where(condition == "Lymphedema", 0.8, 0.0) + rng.normal(0, 0.3, n)
    return pd.Series(axis), pd.Series(score), pd.Series(condition)


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _draw(**kwargs):
    axis, score, condition = _cells()
    base = dict(
        case="Lymphedema", control="Normal", axis_label="EndoMT index", score_label="TEAD activity"
    )
    base.update(kwargs)
    return score_along_axis(axis, score, condition, **base)


def _curves(fig):
    return {
        ln.get_gid(): ln for ln in fig.axes[0].lines if (ln.get_gid() or "").startswith("curve|")
    }


def test_one_smoothed_curve_per_condition_control_first() -> None:
    fig = _draw()
    curves = _curves(fig)
    assert list(curves) == ["curve|Normal", "curve|Lymphedema"]


def test_each_curve_follows_a_score_that_rises_along_the_axis() -> None:
    fig = _draw()
    for line in _curves(fig).values():
        y = np.asarray(line.get_ydata())
        assert y[-1] > y[0]


def test_lymphedema_curve_sits_above_the_normal_one_for_these_data() -> None:
    curves = _curves(_draw())
    normal, lymphedema = (
        np.asarray(curves[f"curve|{c}"].get_ydata()) for c in ("Normal", "Lymphedema")
    )
    assert float(np.mean(lymphedema - normal)) > 0.3


def test_every_cell_is_drawn_as_a_faint_point() -> None:
    fig = _draw()
    points = sum(len(c.get_offsets()) for c in fig.axes[0].collections if c.get_gid() == "cells")
    assert points == 300


def test_a_band_of_resampling_uncertainty_is_drawn_for_each_curve() -> None:
    fig = _draw()
    bands = [c for c in fig.axes[0].collections if (c.get_gid() or "").startswith("band|")]
    assert len(bands) == 2


def test_the_axes_are_named() -> None:
    ax = _draw().axes[0]
    assert ax.get_xlabel() == "EndoMT index"
    assert ax.get_ylabel() == "TEAD activity"


def test_the_same_seed_draws_the_same_curves() -> None:
    first = {k: np.asarray(v.get_ydata()) for k, v in _curves(_draw(seed=3)).items()}
    second = {k: np.asarray(v.get_ydata()) for k, v in _curves(_draw(seed=3)).items()}
    for key in first:
        np.testing.assert_allclose(first[key], second[key])


def test_a_condition_with_too_few_cells_is_named_not_silently_dropped() -> None:
    axis, score, condition = _cells()
    condition = condition.where(condition != "Normal", "Lymphedema")
    condition.iloc[:3] = "Normal"
    with pytest.raises(ValueError, match="Normal"):
        score_along_axis(
            axis,
            score,
            condition,
            case="Lymphedema",
            control="Normal",
            axis_label="a",
            score_label="s",
            min_cells=10,
        )
