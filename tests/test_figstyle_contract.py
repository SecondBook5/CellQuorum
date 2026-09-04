# tests/test_figstyle_contract.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def test_distinct_palette_returns_n_valid_hex_colors():
    for n in (1, 5, 18, 41):
        colors = figstyle.distinct_palette(n)
        assert len(colors) == n
        assert all(c.startswith("#") and len(c) == 7 for c in colors)


def test_distinct_palette_never_repeats_even_past_fixed_list_length():
    # The generator must yield N distinct colors for any N — including counts
    # far past the 18-slot fixed CATEGORICAL_PALETTE — so a 41-subtype figure
    # never collapses two clusters onto one hue (the "coloring is atrocious"
    # fault came from cycling a short fixed list).
    colors = figstyle.distinct_palette(41)
    assert len(set(colors)) == 41


def test_distinct_palette_empty_for_nonpositive():
    assert figstyle.distinct_palette(0) == []
    assert figstyle.distinct_palette(-3) == []


# ---------------------------------------------------------------------------
# Palette assignment. The one property every categorical figure in the engine
# depends on: two categories must never be drawn in the same color. A legend
# does not recover that — the reader has no way to tell which of the two a
# point belongs to. This is a regression suite: a real 16-cluster LEC velocity
# figure came out with clusters 1 and 5 in one identical orange, because the
# group palette was built by CYCLING a fixed 20-entry list that itself
# contained "#FFB74D" twice.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 8, 12, 18, 19, 25, 41])
def test_palette_colors_never_repeats_at_any_cardinality(n):
    colors = figstyle.palette_colors(n)
    assert len(colors) == n
    assert len(set(colors)) == n, f"duplicate color at n={n}"
    assert all(c.startswith("#") and len(c) == 7 for c in colors)


def test_palette_colors_prefers_the_validated_fixed_palette_for_small_n():
    # Up to 18 groups the answer is the validated CVD-safe palette, in its
    # validated order — not the generator, whose ordering was never run through
    # the dataviz validator.
    assert figstyle.palette_colors(5) == figstyle.CATEGORICAL_PALETTE[:5]
    assert figstyle.palette_colors(18) == figstyle.CATEGORICAL_PALETTE
    # One past the fixed palette it must hand off to the generator rather than
    # wrap slot 19 back onto slot 1.
    assert figstyle.palette_colors(19)[18] not in figstyle.palette_colors(19)[:18]


def test_palette_colors_empty_for_nonpositive():
    assert figstyle.palette_colors(0) == []
    assert figstyle.palette_colors(-2) == []


@pytest.mark.parametrize("n", [2, 12, 18, 25, 41])
def test_get_group_palette_never_assigns_one_color_to_two_groups(n):
    groups = [f"cluster_{i}" for i in range(n)]
    palette = figstyle.get_group_palette(groups)
    assert set(palette) == set(groups)
    assert len(set(palette.values())) == n, f"two groups share a color at n={n}"


def test_get_group_palette_is_deterministic_and_order_independent():
    # Two figures drawn from the same groups in different orders must color
    # them identically, or the same cluster changes color between panels.
    groups = [f"c{i}" for i in range(14)]
    assert figstyle.get_group_palette(groups) == figstyle.get_group_palette(list(reversed(groups)))
    # Duplicates are one group, and must not consume two palette slots.
    assert figstyle.get_group_palette(["a", "b", "a"]) == figstyle.get_group_palette(["a", "b"])


def test_categorical_palette_fn_keeps_observed_order_and_never_repeats():
    values = ["z", "a", "m"]
    palette = figstyle.categorical_palette(values)
    # Observed order, not sorted: this one is for label sets whose order is
    # already meaningful (a category ordering carried on the object).
    assert list(palette) == values
    assert palette["z"] == figstyle.CATEGORICAL_PALETTE[0]
    # Past the fixed palette's length it must still hand out distinct colors.
    many = figstyle.categorical_palette([f"s{i}" for i in range(30)])
    assert len(set(many.values())) == 30


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


def test_save_figure_leaves_no_partial_file_when_a_format_fails(tmp_path):
    """A figure that fails to render must not leave a file that looks rendered.

    The real incident: the velocity stream figure raised inside the PDF backend
    ("Can only output finite numbers in PDF") *after* the backend had written
    part of the stream, leaving a 38 KB truncated ``velocity_stream.pdf`` sitting
    in the figures directory — a file that lists like a figure and fails to open.
    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "figs"

    real_savefig = fig.savefig

    def _fail_on_pdf(target, *args, **kwargs):
        real_savefig(target, *args, **kwargs)  # write bytes first, as the backend does
        if str(kwargs.get("format")) == "pdf":
            raise ValueError("Can only output finite numbers in PDF")

    fig.savefig = _fail_on_pdf  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="finite numbers"):
        figstyle.save_figure(fig, out, "broken", formats=("pdf",))

    # No .pdf, and no leftover temp file either.
    assert list(out.iterdir()) == []


def test_save_figure_attempts_every_format_even_when_one_raises(tmp_path):
    """One failing format must not cost the others.

    Before this, the stream figure's PDF failure meant its PNG — which renders
    fine, and is the format the report embeds — was never attempted at all.
    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "figs"

    real_savefig = fig.savefig

    def _fail_on_pdf(target, *args, **kwargs):
        if str(kwargs.get("format")) == "pdf":
            raise ValueError("Can only output finite numbers in PDF")
        real_savefig(target, *args, **kwargs)

    fig.savefig = _fail_on_pdf  # type: ignore[method-assign]
    paths = figstyle.save_figure(fig, out, "partial", formats=("pdf", "png"))

    assert [p.name for p in paths] == ["partial.png"]
    assert (out / "partial.png").stat().st_size > 0
    assert not (out / "partial.pdf").exists()
    # And the figure is still closed, so a long run does not leak figures.
    assert not plt.fignum_exists(fig.number)


def test_atomic_savefig_writes_the_format_the_path_names(tmp_path):
    """The write goes to a ``.tmp`` file, so the format cannot be left to inference.

    matplotlib picks the format from the filename it is handed. Handing it
    ``.demo.pdf.tmp`` makes it raise "Format 'tmp' is not supported" — which is
    what happened to every by-path caller (the scree plot, the QC tables, the
    reference-mapping diagnostics) the moment they were routed through here.
    """
    for suffix, magic in ((".pdf", b"%PDF"), (".png", b"\x89PNG")):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        target = tmp_path / f"demo{suffix}"
        figstyle.atomic_savefig(fig, target)
        plt.close(fig)
        assert target.read_bytes().startswith(magic), suffix
    # And nothing named .tmp survives either write.
    assert not [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_save_cellquorum_figure_writes_a_vector_twin_and_does_not_close(tmp_path):
    """A PNG-only figure is not submittable, and 19 call sites ask for PNG.

    A real run's figures directory held 60 PNGs and 57 PDFs; the three without a
    vector form were exactly the ones written by full path.
    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    returned = figstyle.save_cellquorum_figure(fig, tmp_path / "panel.png", dpi=100)

    # The return value is the format the caller asked for, since callers build
    # StageArtifacts from it.
    assert returned == tmp_path / "panel.png"
    assert (tmp_path / "panel.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "panel.pdf").read_bytes().startswith(b"%PDF")
    # This writer does not close: nineteen call sites are written against that.
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_save_cellquorum_figure_keeps_the_requested_figure_when_the_twin_fails(tmp_path):
    """The companion is additive. Losing it must not cost the caller its figure.

    The vector backend can refuse a figure the raster one accepted — that is
    exactly how the velocity stream figure failed.
    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    real_savefig = fig.savefig

    def _fail_on_pdf(target, *args, **kwargs):
        if str(kwargs.get("format")) == "pdf":
            raise ValueError("Can only output finite numbers in PDF")
        real_savefig(target, *args, **kwargs)

    fig.savefig = _fail_on_pdf  # type: ignore[method-assign]
    returned = figstyle.save_cellquorum_figure(fig, tmp_path / "panel.png", dpi=100)
    plt.close(fig)

    assert returned == tmp_path / "panel.png"
    assert (tmp_path / "panel.png").stat().st_size > 0
    assert not (tmp_path / "panel.pdf").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["panel.png"]


def test_save_publication_figure_writes_only_what_was_asked_for(tmp_path):
    """Its one caller loops over ``("png", "pdf")`` itself.

    So the companion must stay off here, or every population-identity figure
    writes its PDF twice — once as the companion of the PNG, once as the format
    the caller then requests.
    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    figstyle.save_publication_figure(fig, tmp_path / "identity.png", dpi=100)
    plt.close(fig)

    assert [p.name for p in tmp_path.iterdir()] == ["identity.png"]


def test_panel_letter_places_bold_text(tmp_path):
    fig, ax = plt.subplots()
    figstyle.panel_letter(ax, "a")
    texts = [t.get_text() for t in ax.texts]
    assert "a" in texts


# ---------------------------------------------------------------------------
# Donor-level two-group testing. These pin the unit of analysis: a figure
# p-value must never be computed over cells, because cells from one donor are
# not independent replicates.
# ---------------------------------------------------------------------------


def _cell_frame(
    *,
    n_donors_per_arm: int = 9,
    cells_per_donor: int = 300,
    shift: float = 0.4,
    paired: bool = True,
) -> pd.DataFrame:
    """Build a cell-level frame with a donor-level offset and per-cell noise.

    The arm-level effect is deliberately small relative to per-cell spread, so a
    cell-level test finds an overwhelming p-value while the donor-level test
    reports something honest.
    """

    rng = np.random.default_rng(0)
    rows = []
    for arm, offset in (("Normal", 0.0), ("LE", shift)):
        for donor_index in range(n_donors_per_arm):
            donor = f"d{donor_index}" if paired else f"{arm}_d{donor_index}"
            donor_effect = rng.normal(0.0, 0.05)
            values = rng.normal(5.0 + offset + donor_effect, 1.0, size=cells_per_donor)
            rows.append(pd.DataFrame({"metric": values, "condition": arm, "donor_id": donor}))
    return pd.concat(rows, ignore_index=True)


def test_donor_level_test_uses_donors_not_cells_as_n():
    frame = _cell_frame()
    result = figstyle.two_group_test_on_donor_medians(
        frame,
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
    )
    assert result is not None
    # n is donors (9), not the 2700 cells per arm.
    assert result.n_group1 == 9
    assert result.n_group2 == 9
    assert "donor medians" in result.label
    assert "n = 9" in result.label


def test_donor_level_test_is_paired_when_donors_appear_in_both_arms():
    result = figstyle.two_group_test_on_donor_medians(
        _cell_frame(paired=True),
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
    )
    assert result is not None
    assert result.test == "wilcoxon_signed_rank"
    assert "signed-rank" in result.label


def test_donor_level_test_is_unpaired_when_donor_sets_are_disjoint():
    result = figstyle.two_group_test_on_donor_medians(
        _cell_frame(paired=False),
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
    )
    assert result is not None
    assert result.test == "mann_whitney"


def test_donor_level_p_value_is_not_the_pseudoreplicated_cell_level_one():
    """The donor-level p-value must be orders of magnitude less extreme."""

    from scipy import stats

    frame = _cell_frame()
    cell_level = stats.mannwhitneyu(
        frame.loc[frame["condition"].eq("Normal"), "metric"],
        frame.loc[frame["condition"].eq("LE"), "metric"],
        alternative="two-sided",
    ).pvalue
    donor_level = figstyle.two_group_test_on_donor_medians(
        frame,
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
    )
    assert donor_level is not None
    # Cell-level pseudoreplication buys many orders of magnitude of fake
    # confidence; the donor-level test cannot exceed 1/2**9 for n=9 paired.
    assert cell_level < 1e-20
    assert donor_level.p_value > cell_level * 1e10


def test_donor_level_test_returns_none_when_underpowered():
    frame = _cell_frame(n_donors_per_arm=2)
    assert (
        figstyle.two_group_test_on_donor_medians(
            frame,
            value_col="metric",
            group_col="condition",
            donor_col="donor_id",
            group1="Normal",
            group2="LE",
        )
        is None
    )


def test_donor_level_test_returns_none_without_a_donor_column():
    frame = _cell_frame().drop(columns=["donor_id"])
    assert (
        figstyle.two_group_test_on_donor_medians(
            frame,
            value_col="metric",
            group_col="condition",
            donor_col="donor_id",
            group1="Normal",
            group2="LE",
        )
        is None
    )


def test_donor_level_test_ignores_non_finite_cells():
    frame = _cell_frame()
    poisoned = frame.copy()
    poisoned.loc[poisoned.index[:50], "metric"] = np.inf
    poisoned.loc[poisoned.index[50:100], "metric"] = np.nan
    clean = figstyle.two_group_test_on_donor_medians(
        frame,
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
    )
    dirty = figstyle.two_group_test_on_donor_medians(
        poisoned,
        value_col="metric",
        group_col="condition",
        donor_col="donor_id",
        group1="Normal",
        group2="LE",
    )
    assert clean is not None and dirty is not None
    # Non-finite values must not become +inf medians or drop a whole donor.
    assert dirty.n_group1 == clean.n_group1
    assert np.isfinite(dirty.p_value)
