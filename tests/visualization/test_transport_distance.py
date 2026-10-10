"""Distance between each donor's two conditions, against what shuffled labels give."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.transport_distance import transport_distance  # noqa: E402


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": ["Structural"] * 3 + ["Inflamed"] * 2,
            "donor": ["P2", "P4", "P7", "P2", "P4"],
            "n_normal": [77, 27, 20, 76, 32],
            "n_le": [68, 43, 68, 70, 33],
            "w2": [5.5, 5.4, 5.0, 5.7, 5.1],
            "null_mean": [4.9, 4.8, 4.8, 5.5, 5.0],
            "null_p95": [5.0, 5.1, 5.5, 5.9, 5.3],
            "p_perm": [0.002, 0.002, 0.31, 0.002, 0.30],
        }
    )


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _axis(fig):
    return next(ax for ax in fig.axes if ax.get_label() == "transport")


def _gids(fig):
    return {c.get_gid() for c in _axis(fig).collections}


def test_every_donor_state_pair_has_an_observed_distance_and_a_null() -> None:
    fig = transport_distance(_table())
    gids = _gids(fig)
    assert {"obs|Structural|P2", "obs|Inflamed|P4", "null|Structural|P7"} <= gids


def test_states_follow_the_order_given_not_the_table_order() -> None:
    fig = transport_distance(_table(), state_order=["Structural", "Inflamed"])
    heads = [t.get_text() for t in _axis(fig).texts if t.get_gid() == "state"]
    assert [h.split("  ")[0] for h in heads] == ["Structural", "Inflamed"]


def test_a_donor_whose_distance_does_not_beat_shuffled_labels_is_an_open_marker() -> None:
    fig = transport_distance(_table(), alpha=0.05)
    fig.canvas.draw()
    called = next(c for c in _axis(fig).collections if c.get_gid() == "obs|Structural|P2")
    not_called = next(c for c in _axis(fig).collections if c.get_gid() == "obs|Structural|P7")
    assert len(called.get_facecolor()) == 1
    assert len(not_called.get_facecolor()) == 0


def test_the_state_header_counts_donors_above_the_null() -> None:
    fig = transport_distance(_table(), alpha=0.05)
    heads = {
        t.get_text().split("  ")[0]: t.get_text()
        for t in _axis(fig).texts
        if t.get_gid() == "state"
    }
    assert "2 of 3" in heads["Structural"]
    assert "1 of 2" in heads["Inflamed"]


def test_cell_counts_per_condition_are_written_beside_each_donor() -> None:
    fig = transport_distance(_table())
    counts = [t.get_text() for t in _axis(fig).texts if t.get_gid() == "cells|Structural|P7"]
    assert counts == ["20 | 68"]


def test_a_missing_column_is_named() -> None:
    with pytest.raises(ValueError, match="null_p95"):
        transport_distance(_table().drop(columns="null_p95"))


def test_the_axis_names_the_measure_it_is_given() -> None:
    fig = transport_distance(_table(), measure_label="EndoMT score change")
    assert "EndoMT score change" in _axis(fig).get_xlabel()
    assert "Wasserstein" not in _axis(fig).get_xlabel()
