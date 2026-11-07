"""The grant theme: the look of the three figures chosen for the grant, applied to every figure."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

matplotlib.use("Agg")

from cellquorum.visualization import figstyle  # noqa: E402


@pytest.fixture(autouse=True)
def _theme():
    figstyle.apply_theme("grant")
    yield
    plt.close("all")


def test_condition_colours_are_navy_for_normal_and_plum_for_lymphedema() -> None:
    assert figstyle.GRANT_NORMAL == "#14304A"
    assert figstyle.GRANT_CASE == "#9C3F71"


def test_axes_are_open_with_dark_ink_and_no_grid() -> None:
    rc = matplotlib.rcParams
    assert rc["axes.spines.top"] is False and rc["axes.spines.right"] is False
    assert not rc["axes.grid"]
    assert to_hex(rc["axes.edgecolor"]).upper() == figstyle.GRANT_INK
    assert to_hex(rc["text.color"]).upper() == figstyle.GRANT_INK


def test_titles_are_bold_navy_and_left_aligned_and_axis_labels_are_bold() -> None:
    rc = matplotlib.rcParams
    assert rc["axes.titlelocation"] == "left"
    assert rc["axes.titleweight"] == "bold"
    assert to_hex(rc["axes.titlecolor"]).upper() == figstyle.GRANT_TITLE
    assert rc["axes.labelweight"] == "bold"


def test_text_is_arial_or_its_metric_twin() -> None:
    assert matplotlib.rcParams["font.sans-serif"][0] == "Arial"
    assert "Liberation Sans" in matplotlib.rcParams["font.sans-serif"]


def test_the_sequential_scale_runs_from_pale_lilac_to_plum() -> None:
    cmap = plt.get_cmap("grant_sequential")
    assert to_hex(cmap(0.0)).upper() == "#DADFE6"
    assert to_hex(cmap(1.0)).upper() == "#9E4976"


def test_the_diverging_scale_runs_navy_to_white_to_plum() -> None:
    cmap = plt.get_cmap("grant_diverging")
    assert to_hex(cmap(0.0)).upper() == figstyle.GRANT_NORMAL
    assert to_hex(cmap(0.5)).upper() == "#FFFFFF"
    assert to_hex(cmap(1.0)).upper() == figstyle.GRANT_CASE


def test_a_figure_drawn_under_the_theme_uses_the_theme() -> None:
    fig, ax = plt.subplots()
    ax.set_title("EndoMT/ECM rises in lymphedema LECs")
    ax.set_xlabel("Normal")
    assert ax.title.get_fontweight() == "bold"
    assert ax.xaxis.label.get_fontweight() == "bold"
    assert not ax.spines["top"].get_visible()
