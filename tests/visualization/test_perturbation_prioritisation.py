"""Which populations a condition changes most: donor-aware distance beside Augur and its null."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.perturbation_prioritisation import (  # noqa: E402
    perturbation_prioritisation,
)


def _scdist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dist": [6.0, 2.4, 0.5, 0.0],
            "ci_low": [5.2, 1.9, 0.2, 0.0],
            "ci_high": [6.8, 2.9, 0.9, 0.0],
            "p": [0.00004, 0.0003, 0.2, 0.7],
        },
        index=["LEC", "Fibroblasts", "DC", "B cells"],
    )


def _augur() -> pd.DataFrame:
    return pd.DataFrame(
        {"auc": [0.76, 0.58, 0.54], "null_auc": [0.51, 0.46, 0.50]},
        index=["LEC", "Fibroblasts", "DC"],
    )


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _axis(fig, label):
    return next(ax for ax in fig.axes if ax.get_label() == label)


def _point(fig, panel, gid):
    return next(c for c in _axis(fig, panel).collections if c.get_gid() == gid)


def test_rows_run_from_the_largest_distance_down() -> None:
    fig = perturbation_prioritisation(_scdist(), _augur())
    labels = [t.get_text() for t in _axis(fig, "scdist").get_yticklabels()]
    assert labels == ["LEC", "Fibroblasts", "DC", "B cells"]


def test_a_distance_that_does_not_clear_alpha_is_an_open_marker() -> None:
    fig = perturbation_prioritisation(_scdist(), _augur(), alpha=0.05)
    fig.canvas.draw()
    sig = _point(fig, "scdist", "dist|LEC")
    ns = _point(fig, "scdist", "dist|DC")
    assert len(sig.get_facecolor()) == 1  # filled
    assert len(ns.get_facecolor()) == 0  # open: no fill at all


def test_without_augur_there_is_only_the_distance_panel() -> None:
    fig = perturbation_prioritisation(_scdist())
    assert {ax.get_label() for ax in fig.axes if ax.get_label()} == {"scdist"}


def test_a_population_augur_did_not_run_is_blank_not_an_auc_of_one_half() -> None:
    fig = perturbation_prioritisation(_scdist(), _augur())
    gids = {c.get_gid() for c in _axis(fig, "augur").collections}
    assert "auc|B cells" not in gids
    assert "auc|LEC" in gids
    notes = [t.get_text() for t in _axis(fig, "augur").texts]
    assert any("not run" in s for s in notes)


def test_the_permutation_null_is_drawn_beside_each_auc() -> None:
    fig = perturbation_prioritisation(_scdist(), _augur())
    assert {c.get_gid() for c in _axis(fig, "augur").collections} >= {"null|LEC", "null|DC"}


def test_the_highlighted_population_is_bold() -> None:
    fig = perturbation_prioritisation(_scdist(), _augur(), highlight="LEC")
    weights = {t.get_text(): t.get_fontweight() for t in _axis(fig, "scdist").get_yticklabels()}
    assert weights["LEC"] == "bold"
    assert weights["DC"] != "bold"
