"""The ligand -> receptor chord: ligand sectors by sender, receptor sectors by receiver.

A chord is easy to make misleading: a ribbon for a change that did not clear its test should
not look like one that did, two chords meant to be compared must share one width scale, and a
pair with no weight must not get a sector.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.communication import ligand_receptor_chord  # noqa: E402


def _edges() -> pd.DataFrame:
    rows = [
        ("Fibroblasts", "CXCL12", "LEC", "ACKR3", 0.9, True),
        ("Fibroblasts", "TGFB1", "LEC", "TGFBR2", 0.6, True),
        ("Macrophages", "VEGFC", "LEC", "FLT4", -0.4, False),
        ("Fibroblasts", "CXCL12", "DC", "CXCR4", 0.3, False),
    ]
    return pd.DataFrame(
        rows, columns=["sender", "ligand", "receiver", "receptor", "weight", "significant"]
    )


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _ribbons(fig):
    ax = fig.axes[0]
    return {p.get_gid(): p for p in ax.patches if (p.get_gid() or "").startswith("ribbon|")}


def _labels(fig, gid):
    return [t.get_text() for t in fig.axes[0].texts if t.get_gid() == gid]


def _chord(**kwargs):
    base = dict(
        sender_palette={"Fibroblasts": "#4B83C4", "Macrophages": "#D2A91A"},
        receiver_palette={"LEC": "#D7381F", "DC": "#0A9A3A"},
    )
    base.update(kwargs)
    return ligand_receptor_chord(_edges(), **base)


def test_one_ribbon_per_pair_and_one_sector_label_per_molecule() -> None:
    fig = _chord(significant_col="significant")
    assert len(_ribbons(fig)) == 4
    assert sorted(_labels(fig, "ligand_label")) == ["CXCL12", "TGFB1", "VEGFC"]
    assert sorted(_labels(fig, "receptor_label")) == ["ACKR3", "CXCR4", "FLT4", "TGFBR2"]


def test_a_ribbon_that_did_not_clear_its_test_is_fainter_than_one_that_did() -> None:
    fig = _chord(significant_col="significant")
    ribbons = _ribbons(fig)
    assert (
        ribbons["ribbon|CXCL12>ACKR3|LEC"].get_alpha()
        > ribbons["ribbon|CXCL12>CXCR4|DC"].get_alpha()
    )


def test_signed_weights_are_red_up_and_blue_down() -> None:
    fig = _chord(signed=True)
    ribbons = _ribbons(fig)
    up = matplotlib.colors.to_rgb(ribbons["ribbon|CXCL12>ACKR3|LEC"].get_facecolor())
    down = matplotlib.colors.to_rgb(ribbons["ribbon|VEGFC>FLT4|LEC"].get_facecolor())
    assert up[0] > up[2]
    assert down[2] > down[0]


def test_a_pair_with_no_weight_gets_no_ribbon_and_no_sector() -> None:
    edges = _edges()
    edges.loc[edges.ligand == "TGFB1", "weight"] = 0.0
    fig = ligand_receptor_chord(edges, sender_palette={}, receiver_palette={})
    assert "ribbon|TGFB1>TGFBR2|LEC" not in _ribbons(fig)
    assert "TGFB1" not in _labels(fig, "ligand_label")


def test_two_chords_on_one_scale_draw_the_same_ribbon_the_same_size() -> None:
    half = _edges().assign(weight=lambda d: d.weight / 2)
    full_fig = _chord(scale=4.0)
    half_fig = ligand_receptor_chord(half, sender_palette={}, receiver_palette={}, scale=4.0)

    def ligand_end_angle(fig):
        """Angle covered where the ribbon meets its ligand sector: the first eight path points."""
        xy = _ribbons(fig)["ribbon|CXCL12>ACKR3|LEC"].get_path().vertices[:8]
        first, last = np.arctan2(xy[0, 1], xy[0, 0]), np.arctan2(xy[-1, 1], xy[-1, 0])
        return float(abs(last - first))

    assert ligand_end_angle(half_fig) == pytest.approx(ligand_end_angle(full_fig) / 2, rel=1e-6)


def test_top_n_keeps_the_largest_weights() -> None:
    fig = _chord(top_n=2)
    assert set(_ribbons(fig)) == {"ribbon|CXCL12>ACKR3|LEC", "ribbon|TGFB1>TGFBR2|LEC"}


def test_an_empty_frame_gives_an_empty_figure_not_an_error() -> None:
    fig = ligand_receptor_chord(_edges().iloc[0:0], sender_palette={}, receiver_palette={})
    assert not _ribbons(fig)


def test_group_names_of_neighbouring_small_sectors_do_not_sit_on_each_other() -> None:
    rows = [(f"S{i}", f"L{i}", "LEC", f"R{i}", 0.02 if i else 5.0, True) for i in range(8)]
    edges = pd.DataFrame(
        rows, columns=["sender", "ligand", "receiver", "receptor", "weight", "significant"]
    )
    fig = ligand_receptor_chord(edges, sender_palette={}, receiver_palette={})
    pts = np.array([t.get_position() for t in fig.axes[0].texts if t.get_gid() == "group_label"])
    gaps = [float(np.hypot(*(pts[i] - pts[j]))) for i in range(len(pts)) for j in range(i)]
    assert min(gaps) > 0.12


def _overlaps(fig, first_gid, other_gids):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = fig.axes[0].texts
    firsts = [t for t in texts if t.get_gid() == first_gid]
    others = [t for t in texts if t.get_gid() in other_gids]
    return [
        (a.get_text(), b.get_text())
        for a in firsts
        for b in others
        if a is not b and a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))
    ]


def test_a_group_name_is_not_drawn_on_top_of_the_ligand_names_around_it() -> None:
    """Real defect on the lymphedema LEC chord: with several senders each carrying one short
    ligand, the sender names (a fixed distance out) ran into the radial ligand names."""
    senders = ["Fibroblast", "T cells", "Macrophages", "DC", "Mast cells", "Melanocytes", "BEC"]
    ligands = ["COL6A3", "TGFB1", "COL3A1", "TGFB1", "HMGB1", "HMGB1", "HMGB1"]
    rows = [
        (s, lig, "LEC", f"R{i}", 1.0 if i < 2 else 0.15, True)
        for i, (s, lig) in enumerate(zip(senders, ligands, strict=True))
    ]
    edges = pd.DataFrame(
        rows, columns=["sender", "ligand", "receiver", "receptor", "weight", "significant"]
    )
    fig = ligand_receptor_chord(edges, sender_palette={}, receiver_palette={})
    clashes = _overlaps(fig, "group_label", {"ligand_label", "receptor_label", "group_label"})
    assert clashes == []


def _fill(patch) -> str:
    from matplotlib.colors import to_hex

    return to_hex(patch.get_facecolor()[:3])


def test_unsigned_ribbons_take_the_sender_colour_when_asked_and_grey_otherwise() -> None:
    """Real ask: the unsigned chord was mostly grey ribbons, which do not say who is sending."""
    plain = _ribbons(_chord())
    by_sender = _ribbons(_chord(ribbon_color="sender"))
    assert _fill(by_sender["ribbon|CXCL12>ACKR3|LEC"]) == "#4b83c4"
    assert _fill(by_sender["ribbon|VEGFC>FLT4|LEC"]) == "#d2a91a"
    assert _fill(plain["ribbon|CXCL12>ACKR3|LEC"]) != "#4b83c4"


def test_highlighted_ligands_keep_their_colour_over_the_sender_colour() -> None:
    fig = _chord(ribbon_color="sender", highlight={"TGFB1": "#4B3FA8"})
    assert _fill(_ribbons(fig)["ribbon|TGFB1>TGFBR2|LEC"]) == "#4b3fa8"


def test_signed_ribbons_stay_red_and_blue_whatever_the_ribbon_colour_setting() -> None:
    fig = _chord(signed=True, ribbon_color="sender")
    ribbons = _ribbons(fig)
    assert _fill(ribbons["ribbon|CXCL12>ACKR3|LEC"]) != "#4b83c4"
    assert _fill(ribbons["ribbon|VEGFC>FLT4|LEC"]) != "#d2a91a"


def test_a_title_and_a_key_note_can_be_drawn_on_the_chord() -> None:
    """Real defect: a chord of the change in signalling said nothing of what red, blue and ribbon
    width meant or which condition it showed; the function cannot know, so the caller says."""
    fig = _chord(title="Lymphedema minus Normal", note="Red: higher in lymphedema")
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == "Lymphedema minus Normal"
    notes = [t.get_text() for t in ax.texts if t.get_gid() == "note"]
    assert notes == ["Red: higher in lymphedema"]


def test_no_title_or_note_is_drawn_unless_given() -> None:
    ax = _chord().axes[0]
    assert ax.get_title(loc="left") == ""
    assert not [t for t in ax.texts if t.get_gid() == "note"]


_LYMPHEDEMA_INTO_LEC = [
    ("Fibroblast", "VCAN", "ITGB1", 0.99),
    ("DC", "COL3A1", "ITGA2_ITGB1", 0.749),
    ("T cells", "TGFB1", "CAV1", 0.965),
    ("Macrophages", "TGFB1", "CAV1", 0.936),
    ("Fibroblast", "COL3A1", "ITGA1_ITGB1", 0.936),
    ("LEC", "VEGFA", "EPHB2", 0.726),
    ("Fibroblast", "COL6A3", "ITGA9_ITGB1", 0.93),
    ("DC", "TGFB1", "CAV1", 0.926),
    ("T cells", "TGFB1", "ITGB1", 0.908),
    ("Melanocytes", "HMGB1", "THBD", 0.645),
    ("Fibroblast", "ADM", "CALCRL", 0.899),
    ("Macrophages", "COL3A1", "DDR2", 0.894),
    ("LEC", "HMGB1", "THBD", 0.596),
    ("LEC", "TGFB1", "CAV1", 0.895),
    ("Macrophages", "COL3A1", "ITGA1_ITGB1", 0.853),
    ("BEC", "HMGB1", "THBD", 0.582),
    ("SMC", "HMGB1", "THBD", 0.591),
    ("DC", "VEGFA", "ITGB1", 0.881),
    ("Fibroblast", "HMGB1", "THBD", 0.549),
    ("Mast cells", "TGFB1", "CAV1", 0.87),
    ("Macrophages", "VEGFA", "ITGB1", 0.865),
    ("DC", "VEGFA", "NRP2", 0.705),
]


def _into_lec() -> pd.DataFrame:
    return pd.DataFrame(
        [(s, lig, "LEC", rec, w) for s, lig, rec, w in _LYMPHEDEMA_INTO_LEC],
        columns=["sender", "ligand", "receiver", "receptor", "weight"],
    )


def test_no_name_overlaps_another_on_a_chord_with_many_small_neighbouring_senders() -> None:
    """Real defect on the lymphedema chord into LEC: three adjacent small senders (Mast cells, T
    cells, BEC) printed their names over each other and over the TGFB1 labels at the top."""
    fig = ligand_receptor_chord(_into_lec(), sender_palette={}, receiver_palette={})
    names = {"group_label", "ligand_label", "receptor_label"}
    assert _overlaps(fig, "group_label", names) == []


def test_every_name_lies_inside_the_frame_so_the_title_cannot_sit_on_one() -> None:
    """Real defect: the frame was widened for the names before the widening itself shrank the
    drawing, so a long receptor name stuck out of the top and the title was drawn across it."""
    note = "Red: higher in lymphedema; blue: lower. Width: size of the mean change over the donors."
    fig = ligand_receptor_chord(
        _into_lec(), sender_palette={}, receiver_palette={}, title="Signals into LEC", note=note
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    frame = ax.get_window_extent(renderer)
    names = [
        t for t in ax.texts if t.get_gid() in {"group_label", "ligand_label", "receptor_label"}
    ]
    outside = [
        t.get_text()
        for t in names
        if t.get_window_extent(renderer).y1 > frame.y1 + 1
        or t.get_window_extent(renderer).x1 > frame.x1 + 1
        or t.get_window_extent(renderer).x0 < frame.x0 - 1
        or t.get_window_extent(renderer).y0 < frame.y0 - 1
    ]
    assert outside == []


def test_a_long_note_is_wrapped_so_it_does_not_widen_the_figure() -> None:
    note = "Red: higher in lymphedema; blue: lower. " * 6
    fig = ligand_receptor_chord(_into_lec(), sender_palette={}, receiver_palette={}, note=note)
    drawn = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == "note"][0]
    assert max(len(line) for line in drawn.splitlines()) <= 95
