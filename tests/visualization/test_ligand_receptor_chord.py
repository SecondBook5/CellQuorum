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
