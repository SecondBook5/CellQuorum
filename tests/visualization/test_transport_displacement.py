"""Where one donor's Normal cells are carried to by the transport plan onto its LE cells."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.transport_distance import transport_displacement  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _inputs():
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = np.array([[4.0, 0.0], [6.0, 2.0]])
    # source 0 -> target 0 only; source 1 splits evenly; source 2 -> target 1 only
    plan = np.array([[1 / 3, 0.0], [1 / 6, 1 / 6], [0.0, 1 / 3]])
    return source, target, plan


def _axis(fig):
    return next(ax for ax in fig.axes if ax.get_label() == "displacement")


def test_each_source_cell_points_at_the_plan_weighted_mean_of_its_targets() -> None:
    source, target, plan = _inputs()
    fig = transport_displacement(source, target, plan)
    arrows = next(c for c in _axis(fig).collections if c.get_gid() == "arrows")
    ends = np.asarray(arrows.get_segments())[:, 1, :]
    np.testing.assert_allclose(ends, [[4.0, 0.0], [5.0, 1.0], [6.0, 2.0]])


def test_arrows_start_at_the_source_cells() -> None:
    source, target, plan = _inputs()
    fig = transport_displacement(source, target, plan)
    arrows = next(c for c in _axis(fig).collections if c.get_gid() == "arrows")
    starts = np.asarray(arrows.get_segments())[:, 0, :]
    np.testing.assert_allclose(starts, source)


def test_arrows_are_coloured_by_the_change_in_the_value_along_each_one() -> None:
    source, target, plan = _inputs()
    fig = transport_displacement(
        source,
        target,
        plan,
        source_value=np.array([0.0, 0.0, 0.0]),
        target_value=np.array([1.0, 3.0]),
        value_label="EndoMT score",
    )
    fig.canvas.draw()
    arrows = next(c for c in _axis(fig).collections if c.get_gid() == "arrows")
    change = np.asarray(arrows.get_array())
    np.testing.assert_allclose(change, [1.0, 2.0, 3.0])


def test_a_plan_that_does_not_match_the_cells_is_refused() -> None:
    source, target, plan = _inputs()
    with pytest.raises(ValueError, match="plan"):
        transport_displacement(source, target, plan[:, :1])


def test_a_source_cell_the_plan_never_moves_is_an_error_not_a_zero_arrow() -> None:
    source, target, plan = _inputs()
    plan[0] = 0.0
    with pytest.raises(ValueError, match="no mass"):
        transport_displacement(source, target, plan)


def test_the_two_sets_of_cells_are_drawn_beneath_the_arrows() -> None:
    source, target, plan = _inputs()
    fig = transport_displacement(source, target, plan)
    gids = {c.get_gid() for c in _axis(fig).collections}
    assert {"normal", "lymphedema"} <= gids


def test_a_background_of_all_cells_is_drawn_first_and_does_not_move_the_arrows() -> None:
    source, target, plan = _inputs()
    background = np.array([[0.0, 0.0], [3.0, 3.0], [9.0, 9.0]])
    fig = transport_displacement(source, target, plan, background_xy=background)
    collections = _axis(fig).collections
    gids = [c.get_gid() for c in collections]
    assert gids.index("background") < gids.index("normal")
    arrows = next(c for c in collections if c.get_gid() == "arrows")
    np.testing.assert_allclose(np.asarray(arrows.get_segments())[:, 0, :], source)


def test_a_key_names_the_two_dot_colours_as_normal_and_lymphedema_cells() -> None:
    """Real defect: the blue and red end-point dots had no key, so the figure did not say which
    cells were carried and which they were carried to."""
    source, target, plan = _inputs()
    legend = _axis(transport_displacement(source, target, plan)).get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["Normal cells", "Lymphedema cells"]
