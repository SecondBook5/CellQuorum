import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure

from cellquorum.stages.trajectory.viz import plots


def test_embedding_scatter_returns_figure():
    coords = np.random.RandomState(0).rand(30, 2)
    vals = np.linspace(0, 1, 30)
    fig = plots.embedding_scatter(coords, vals, title="t", cbar_label="pseudotime")
    assert isinstance(fig, Figure)


def test_embedding_scatter_handles_non_finite_values_without_crashing(tmp_path):
    """matplotlib's own autoscale already masks non-finite values (verified:
    scatter(c=[...finite..., inf]).get_clim() returns the finite max, not
    inf) -- so an inf in the array was never the thing crushing this figure's
    color scale. It must still not crash export, though (PDF export failing
    on a non-finite value reaching the backend is a real failure class fixed
    elsewhere in this codebase for the velocity streamplot).
    """
    coords = np.random.RandomState(0).rand(10, 2)
    vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, np.inf])
    fig = plots.embedding_scatter(coords, vals, title="t", cbar_label="pt")
    fig.savefig(tmp_path / "out.pdf", format="pdf")


def test_embedding_scatter_clips_color_scale_to_be_robust_to_outliers():
    """Real bug found on the fibroblast cohort's dpt_pseudotime figure: 99.84%
    of 61,601 cells sat in [0, 0.03] of the raw value range, while a handful
    of legitimate finite outlier cells reached up to 1.0. embedding_scatter's
    color scale used raw min/max, so it spanned the full [0, 1] range and the
    99.84% majority collapsed into one indistinguishable dark shade at the
    bottom of the colormap -- a figure that LOOKED broken (near-uniform
    color) even though every underlying value was finite and real. The scale
    must be robust to a few outliers, the same skew-robustness principle
    already applied to the pseudotime heatmap's rank-based binning.
    """
    coords = np.random.RandomState(0).rand(1000, 2)
    vals = np.concatenate([np.linspace(0.0, 0.03, 995), np.linspace(0.1, 1.0, 5)])
    fig = plots.embedding_scatter(coords, vals, title="t", cbar_label="pt")
    ax = fig.axes[0]
    mappables = [c for c in ax.collections if c.get_array() is not None]
    assert mappables, "no colored scatter collection found"
    vmin, vmax = mappables[0].get_clim()
    # A handful of outliers must not stretch the scale to the raw max; it
    # should track where the bulk of the real data actually is.
    assert vmax < 0.1, f"vmax={vmax} -- outliers still dominate the color scale"


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


def test_grouped_violin_does_not_draw_misleading_extrema_bars():
    """Real complaint: matplotlib's default violinplot(showextrema=True) draws
    a thin bar from the true min to true max of each group. For a 0-1-bounded
    value like a fate probability, that bar visually dominates over the much
    narrower actual density shape, reading as a plotting bug rather than the
    real (concentrated) distribution it's supposed to show.
    """
    groups = {"A": np.random.RandomState(0).rand(30), "B": np.random.RandomState(1).rand(30)}
    fig = plots.grouped_violin(groups, title="t", ylabel="fate prob")
    ax = fig.axes[0]
    # showextrema=True would add THREE extra LineCollections (cmins, cmaxes,
    # cbars) alongside the one legitimate cmedians line; only cmedians (kept
    # via showmedians=True) should remain.
    from matplotlib.collections import LineCollection

    line_collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(line_collections) == 1, f"expected only cmedians, found {len(line_collections)}"


def test_grouped_violin_bodies_are_not_the_bare_matplotlib_default_color():
    """Real complaint: unstyled matplotlib default fill reads as generic /
    off-house-style next to every other figure in this package.
    """
    from matplotlib.collections import PolyCollection

    groups = {"A": np.random.RandomState(0).rand(30), "B": np.random.RandomState(1).rand(30)}
    fig = plots.grouped_violin(groups, title="t", ylabel="fate prob")
    ax = fig.axes[0]
    bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert bodies
    default_blue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)
    for body in bodies:
        face = tuple(body.get_facecolor()[0])
        assert (
            face[:3] != default_blue[:3]
        ), "violin body still uses matplotlib's bare default color"
