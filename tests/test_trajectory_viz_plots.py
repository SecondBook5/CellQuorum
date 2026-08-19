import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure

from cellquorum.trajectory.viz import plots


def test_embedding_scatter_returns_figure():
    coords = np.random.RandomState(0).rand(30, 2)
    vals = np.linspace(0, 1, 30)
    fig = plots.embedding_scatter(coords, vals, title="t", cbar_label="pseudotime")
    assert isinstance(fig, Figure)


def test_signed_diverging_bar_handles_all_positive():
    fig = plots.signed_diverging_bar(["a", "b"], np.array([0.3, 0.7]), title="t")
    assert isinstance(fig, Figure)


def test_matrix_heatmap_shape_ok():
    m = np.random.RandomState(1).rand(3, 4)
    fig = plots.matrix_heatmap(
        m, ["r0", "r1", "r2"], ["c0", "c1", "c2", "c3"], title="t", cbar_label="corr"
    )
    assert isinstance(fig, Figure)


def test_grouped_violin_sorted_groups():
    groups = {"B": np.random.RandomState(2).rand(20), "A": np.random.RandomState(3).rand(20)}
    fig = plots.grouped_violin(groups, title="t", ylabel="fate prob")
    assert isinstance(fig, Figure)
