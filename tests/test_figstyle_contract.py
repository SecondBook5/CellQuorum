# tests/test_figstyle_contract.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cellquorum.visualization import figstyle


def test_set_style_sets_editable_vector_fonts():
    figstyle.set_style()
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42
    assert matplotlib.rcParams["svg.fonttype"] == "none"


def test_categorical_palette_has_18_hex_colors():
    assert len(figstyle.CATEGORICAL_PALETTE) == 18
    assert all(c.startswith("#") and len(c) == 7 for c in figstyle.CATEGORICAL_PALETTE)


def test_categorical_palette_first_eight_are_validated_dataviz_core():
    # Slots 1-8 are the dataviz reference categorical theme in its CVD-safe
    # order. This pins them so a future edit cannot silently swap in a palette
    # that fails the colorblind-separation gate (see module docstring).
    expected_core = [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#4a3aa7",
        "#e34948",
    ]
    assert figstyle.CATEGORICAL_PALETTE[:8] == expected_core


def test_categorical_palette_hues_are_distinct():
    # No duplicate hues anywhere in the 18 slots (a duplicate would collapse
    # two categories onto one color).
    palette = figstyle.CATEGORICAL_PALETTE
    assert len(set(palette)) == len(palette)


def test_condition_palette_maps_case_red_control_blue():
    pal = figstyle.condition_palette("LE", "Normal")
    assert pal["LE"] == figstyle.LE_RED
    assert pal["Normal"] == figstyle.NORMAL_BLUE


def test_condition_palette_extra_conditions_use_categorical_by_sorted_order():
    pal = figstyle.condition_palette("LE", "Normal", others=["Zed", "Alpha"])
    # Extra conditions fall back to the categorical palette in sorted order.
    assert pal["Alpha"] == figstyle.CATEGORICAL_PALETTE[0]
    assert pal["Zed"] == figstyle.CATEGORICAL_PALETTE[1]


def test_condition_palette_handles_missing_labels():
    pal = figstyle.condition_palette(None, None, others=["A", "B"])
    assert pal["A"] == figstyle.CATEGORICAL_PALETTE[0]
    assert pal["B"] == figstyle.CATEGORICAL_PALETTE[1]


def test_diverging_norm_centers_at_zero():
    norm = figstyle.diverging_norm(np.array([-3.0, 1.0, 2.0]))
    assert norm.vcenter == 0.0
    assert norm.vmin < 0 < norm.vmax


def test_diverging_norm_symmetric_when_vmax_given():
    norm = figstyle.diverging_norm(np.array([-1.0, 5.0]), vmax=4.0)
    assert norm.vmin == -4.0
    assert norm.vmax == 4.0


@pytest.mark.parametrize(
    "p,stars",
    [(0.0001, "***"), (0.005, "**"), (0.03, "*"), (0.2, "ns"), (float("nan"), "ns")],
)
def test_significance_stars_ladder(p, stars):
    assert figstyle.significance_stars(p) == stars


def test_save_figure_writes_png_and_pdf_and_closes(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = figstyle.save_figure(fig, tmp_path / "sub", "demo")
    assert {p.suffix for p in paths} == {".pdf", ".png"}
    assert all(p.exists() for p in paths)
    # Parent directory was created.
    assert (tmp_path / "sub").is_dir()
    # Figure was closed (no longer in the active figure registry).
    assert not plt.fignum_exists(fig.number)


def test_panel_letter_places_bold_text(tmp_path):
    fig, ax = plt.subplots()
    figstyle.panel_letter(ax, "a")
    texts = [t.get_text() for t in ax.texts]
    assert "a" in texts
