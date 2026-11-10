"""State-to-state transport flow: control states on the left, case states on the right."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.transport_flow import transport_flow  # noqa: E402

_STATES_N = ["Structural", "Inflamed"]
_STATES_L = ["Structural", "Inflamed", "CV"]


def _flow() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.40, 0.10, 0.00], [0.10, 0.30, 0.10]], index=_STATES_N, columns=_STATES_L
    )


def _excess() -> pd.DataFrame:
    return pd.DataFrame([[0.0, -0.6, 0.0], [0.8, 0.0, 0.2]], index=_STATES_N, columns=_STATES_L)


def _significant() -> pd.DataFrame:
    return pd.DataFrame(
        [[False, True, False], [False, False, False]], index=_STATES_N, columns=_STATES_L
    )


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _ax(fig):
    return next(a for a in fig.axes if a.get_label() == "flow")


def _ribbons(fig):
    return {p.get_gid(): p for p in _ax(fig).patches if str(p.get_gid()).startswith("flow|")}


def test_flow_ribbon_widths_and_positions_encode_positive_transport_mass() -> None:
    fig = transport_flow(_flow())

    ribbons = _ribbons(fig)
    assert "flow|Structural|CV" not in ribbons
    assert "flow|Inflamed|CV" in ribbons
    assert len(ribbons) == 5

    def height(gid: str) -> float:
        v = ribbons[gid].get_path().vertices
        return float(abs(v[0][1] - v[7][1]))

    assert height("flow|Structural|Structural") / height("flow|Structural|Inflamed") == (
        pytest.approx(4.0)
    )

    texts = {t.get_gid(): t for t in _ax(fig).texts}
    assert texts["source|Inflamed"].get_position()[0] < texts["target|Inflamed"].get_position()[0]
    assert any("Control" in t.get_text() for t in _ax(fig).texts)


def test_flow_excess_and_significance_set_ribbon_colours_and_opacity() -> None:
    fig = transport_flow(_flow(), excess=_excess(), significant=_significant())

    ribbons = _ribbons(fig)
    assert (
        ribbons["flow|Structural|Inflamed"].get_alpha()
        > ribbons["flow|Inflamed|Structural"].get_alpha()
    )

    more = ribbons["flow|Inflamed|Structural"].get_facecolor()
    less = ribbons["flow|Structural|Inflamed"].get_facecolor()
    assert more[0] > more[2]
    assert less[2] > less[0]


def test_the_colour_bar_is_named() -> None:
    fig = transport_flow(_flow(), excess=_excess(), significant=_significant(), cbar_label="log2")
    assert any(a.get_ylabel() == "log2" for a in fig.axes)


def test_an_excess_table_that_does_not_line_up_is_refused() -> None:
    with pytest.raises(ValueError, match="excess"):
        transport_flow(_flow(), excess=_excess().iloc[:, :2])


def test_flows_that_are_not_non_negative_are_refused() -> None:
    bad = _flow()
    bad.iloc[0, 0] = -0.1
    with pytest.raises(ValueError, match="negative"):
        transport_flow(bad)


def test_the_order_given_sets_the_stacking() -> None:
    fig = transport_flow(_flow(), source_order=["Inflamed", "Structural"])
    ys = {t.get_gid(): t.get_position()[1] for t in _ax(fig).texts}
    assert ys["source|Inflamed"] > ys["source|Structural"]
    assert np.isfinite(ys["target|CV"])


def test_source_ribbon_color_follows_the_source_state():
    colors = {"Structural": "#112233", "Inflamed": "#aa5500"}
    fig = transport_flow(_flow(), state_colors=colors, ribbon_color="source")
    patches = {
        p.get_gid(): p for p in fig.axes[0].patches if (p.get_gid() or "").startswith("flow|")
    }
    from matplotlib.colors import to_rgb

    assert to_rgb(patches["flow|Structural|Inflamed"].get_facecolor()) == pytest.approx(
        to_rgb("#112233")
    )
    assert to_rgb(patches["flow|Inflamed|CV"].get_facecolor()) == pytest.approx(to_rgb("#aa5500"))


def test_excess_overrides_source_ribbon_color():
    colors = {"Structural": "#112233", "Inflamed": "#aa5500"}
    fig = transport_flow(_flow(), excess=_excess(), state_colors=colors, ribbon_color="source")
    from matplotlib.colors import to_rgb

    patch = next(p for p in fig.axes[0].patches if p.get_gid() == "flow|Inflamed|Structural")
    assert to_rgb(patch.get_facecolor()) != pytest.approx(to_rgb("#aa5500"))


def test_unknown_ribbon_color_is_refused():
    with pytest.raises(ValueError, match="ribbon_color"):
        transport_flow(_flow(), ribbon_color="rainbow")
