"""A score or index drawn as overlaid histograms, one per condition."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.score_histogram import score_histogram  # noqa: E402


def _cells(n_control: int = 200, n_case: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    value = np.concatenate([rng.beta(2, 5, n_control), rng.beta(5, 2, n_case)])
    condition = np.array(["Normal"] * n_control + ["Lymphedema"] * n_case)
    return pd.Series(value), pd.Series(condition)


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _draw(**kwargs):
    value, condition = _cells()
    base = dict(case="Lymphedema", control="Normal", value_label="EndoMT index")
    base.update(kwargs)
    return score_histogram(value, condition, **base)


def _heights(fig, level):
    return np.array([p.get_height() for p in fig.axes[0].patches if p.get_gid() == f"bars|{level}"])


def test_one_set_of_bars_per_condition_control_first() -> None:
    fig = _draw()
    gids = [p.get_gid() for p in fig.axes[0].patches]
    assert gids[0] == "bars|Normal" and gids[-1] == "bars|Lymphedema"
    assert set(gids) == {"bars|Normal", "bars|Lymphedema"}


def test_bar_heights_are_the_cell_counts_of_each_condition() -> None:
    fig = _draw()
    assert _heights(fig, "Normal").sum() == 200
    assert _heights(fig, "Lymphedema").sum() == 300


def test_both_conditions_share_the_same_bin_edges() -> None:
    fig = _draw()
    edges = {
        level: [p.get_x() for p in fig.axes[0].patches if p.get_gid() == f"bars|{level}"]
        for level in ("Normal", "Lymphedema")
    }
    np.testing.assert_allclose(edges["Normal"], edges["Lymphedema"])


def test_the_number_of_bins_is_the_one_asked_for() -> None:
    assert len(_heights(_draw(bins=12), "Normal")) == 12


def test_share_scales_each_condition_to_one_whatever_its_size() -> None:
    fig = _draw(stat="share")
    assert _heights(fig, "Normal").sum() == pytest.approx(1.0)
    assert _heights(fig, "Lymphedema").sum() == pytest.approx(1.0)
    assert fig.axes[0].get_ylabel() == "Share of cells"


def test_the_legend_gives_the_number_of_cells_per_condition() -> None:
    fig = _draw()
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert texts == ["Normal (n = 200)", "Lymphedema (n = 300)"]


def test_the_axes_are_named() -> None:
    ax = _draw().axes[0]
    assert ax.get_xlabel() == "EndoMT index"
    assert ax.get_ylabel() == "Cells"


def test_a_condition_with_no_cells_is_named() -> None:
    value, condition = _cells()
    with pytest.raises(ValueError, match="Lymphedema"):
        score_histogram(
            value,
            condition.where(condition != "Lymphedema", "Normal"),
            case="Lymphedema",
            control="Normal",
            value_label="x",
        )


def test_an_unknown_stat_is_refused() -> None:
    with pytest.raises(ValueError, match="stat"):
        _draw(stat="density")
