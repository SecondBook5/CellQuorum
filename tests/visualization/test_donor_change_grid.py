"""The per-donor change grid: each donor's change, with the donor-level evidence drawn in."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cellquorum.visualization.donor_change_grid import donor_change_grid

DONORS = ["P1", "P2", "P10", "P4", "P5", "P6", "P7", "P8", "P12"]


def _samples(donors=DONORS, drop_control_of: str | None = None):
    """One control and one case sample per donor; three programs with known per-donor changes."""
    up = np.array([0.04, 0.06, 0.05, 0.17, 0.03, 0.05, 0.06, 0.04, 0.05])  # every donor higher
    mixed = np.array([0.05, 0.04, 0.03, 0.02, 0.01, -0.03, -0.02, -0.04, -0.01])  # 5 up, 4 down
    down = -np.array([0.05, 0.04, 0.06, 0.05, 0.03, 0.04, 0.07, 0.05, 0.04])  # every donor lower
    rows, donor, cond = {}, [], []
    for i, d in enumerate(donors):
        for arm, sign in (("Normal", 0.0), ("Lymphedema", 1.0)):
            if d == drop_control_of and arm == "Normal":
                continue
            rows[f"{d}_{arm}"] = [0.2 + sign * up[i], 0.3 + sign * mixed[i], 0.4 + sign * down[i]]
            donor.append(d)
            cond.append(arm)
    values = pd.DataFrame.from_dict(rows, orient="index", columns=["Up", "Mixed", "Down"])
    return values, pd.Series(donor, index=values.index), pd.Series(cond, index=values.index)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _draw(**kwargs):
    values, donors, conditions = _samples(drop_control_of=kwargs.pop("drop_control_of", None))
    fig, ax = donor_change_grid(
        values,
        donors,
        conditions,
        case="Lymphedema",
        control="Normal",
        colorbar_label="Change in score",
        **kwargs,
    )
    return fig, ax


def test_donor_grid_shows_paired_changes_agreement_and_scale() -> None:
    fig, ax = _draw()

    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["P1", "P2", "P4", "P5", "P6", "P7", "P8", "P10", "P12", "Mean"]

    grid = np.asarray(ax.images[0].get_array())
    assert grid[0, 0] == pytest.approx(0.04)  # Up, P1
    assert grid[0, 3] == pytest.approx(0.03)  # Up, P5
    assert grid[1, 5] == pytest.approx(-0.02)  # Mixed, P7
    assert grid[0, -1] == pytest.approx(
        np.mean([0.04, 0.06, 0.05, 0.17, 0.03, 0.05, 0.06, 0.04, 0.05])
    )

    texts = {t.get_text() for t in ax.texts}
    assert "9/9*" in texts
    assert "5/9" in texts

    outlines = [p for p in ax.patches if not p.get_fill()]
    assert len(outlines) == 4
    assert {round(p.get_y() + p.get_height() / 2) for p in outlines} == {1}  # all in the Mixed row

    norm = ax.images[0].norm
    assert norm.vmin == -norm.vmax
    assert norm.vmax >= 0.17
    assert norm.vmax == pytest.approx(0.2)  # the next round number above the largest change

    words = " ".join(t.get_text() for t in fig.findobj(plt.Text))
    assert "dashed outline" in words.lower()
    assert "FDR" in words


def test_the_scale_can_be_set_by_the_caller() -> None:
    _, ax = _draw(vmax=0.5)
    assert ax.images[0].norm.vmax == 0.5


def test_a_donor_with_one_arm_missing_is_left_blank_not_read_as_no_change() -> None:
    _, ax = _draw(drop_control_of="P7")
    grid = np.asarray(ax.images[0].get_array())
    column = [t.get_text() for t in ax.get_xticklabels()].index("P7")
    assert np.ma.getmaskarray(ax.images[0].get_array())[:, column].all()
    assert np.isfinite(grid[:, -1]).all()
