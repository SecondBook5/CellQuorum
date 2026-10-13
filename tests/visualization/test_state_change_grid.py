"""The state-by-program change grid: one change per program and state, with donor agreement."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cellquorum.visualization.donor_change_grid import state_change_grid

DONORS = [f"P{i}" for i in range(1, 10)]
MIXED = np.array([0.05, 0.04, 0.03, 0.02, 0.01, -0.03, -0.02, -0.04, -0.01])  # 5 up, 4 down


def _units():
    """Per donor, arm and state means. State B has a control sample for donors P1-P4 only."""
    rows, donor, cond, state = {}, [], [], []
    for i, d in enumerate(DONORS):
        for st in ("A", "B"):
            for arm, sign in (("Normal", 0.0), ("Lymphedema", 1.0)):
                if st == "B" and arm == "Normal" and i >= 4:
                    continue
                rows[f"{d}|{arm}|{st}"] = [
                    0.2 + sign * 0.12,
                    0.3 - sign * 0.12,
                    0.4 + sign * MIXED[i],
                ]
                donor.append(d)
                cond.append(arm)
                state.append(st)
    values = pd.DataFrame.from_dict(rows, orient="index", columns=["Up", "Down", "Mixed"])
    idx = values.index
    return (
        values,
        pd.Series(donor, index=idx),
        pd.Series(cond, index=idx),
        pd.Series(state, index=idx),
    )


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _draw(**kwargs):
    values, donors, conditions, states = _units()
    return state_change_grid(
        values,
        donors,
        conditions,
        states,
        case="Lymphedema",
        control="Normal",
        colorbar_label="Change in score",
        **kwargs,
    )


def test_programs_run_down_and_states_across_in_the_order_given() -> None:
    _, ax = _draw(group_order=["B", "A"])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["B", "A"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Up", "Down", "Mixed"]


def test_each_cell_is_the_mean_donor_change_in_that_state() -> None:
    _, ax = _draw(group_order=["A", "B"])
    grid = np.asarray(ax.images[0].get_array())
    assert grid[0, 0] == pytest.approx(0.12)
    assert grid[1, 0] == pytest.approx(-0.12)
    assert grid[2, 0] == pytest.approx(MIXED.mean())
    assert grid[2, 1] == pytest.approx(MIXED[:4].mean())  # only P1-P4 have both arms in B


def test_each_cell_gives_its_change_and_the_donors_that_agree() -> None:
    _, ax = _draw(group_order=["A", "B"])
    texts = {t.get_text() for t in ax.texts}
    assert "+0.12\n9/9*" in texts
    assert "-0.12\n9/9*" in texts
    assert f"{MIXED.mean():+.2f}\n5/9" in texts
    assert "+0.12\n4/4" in texts


def test_the_correction_runs_over_every_cell_of_the_grid() -> None:
    # Two unanimous nine-donor cells reach 0.0039 doubled; pooled over six cells that is 0.012.
    _, ax = _draw(group_order=["A", "B"])
    starred = [t.get_text() for t in ax.texts if t.get_text().endswith("*")]
    assert len(starred) == 2


def test_cells_with_too_few_donors_to_reach_significance_are_outlined() -> None:
    _, ax = _draw(group_order=["A", "B"])
    outlines = [p for p in ax.patches if not p.get_fill()]
    assert len(outlines) == 3  # the three cells of state B, four donors each
    assert {round(p.get_x() + p.get_width() / 2) for p in outlines} == {1}


def test_the_colour_scale_is_symmetric_and_covers_every_change() -> None:
    _, ax = _draw(group_order=["A", "B"])
    norm = ax.images[0].norm
    assert norm.vmin == -norm.vmax
    assert norm.vmax == pytest.approx(0.15)  # the next round number above 0.12


def test_the_figure_says_what_the_marks_mean() -> None:
    fig, _ = _draw()
    words = " ".join(t.get_text() for t in fig.findobj(plt.Text)).lower()
    assert "dashed outline" in words
    assert "fdr" in words
