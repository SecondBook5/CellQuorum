import matplotlib

matplotlib.use("Agg")
import numpy as np

from cellquorum.stages.trajectory.viz import heatmap


def test_bin_masks_sorts_and_bins():
    pt = np.array([0.9, 0.1, 0.5, 0.3])
    order, binid = heatmap.bin_masks(pt, n_bins=4)
    # order sorts pseudotime ascending
    assert list(pt[order]) == sorted(pt)
    assert binid.min() >= 0 and binid.max() <= 3


def test_binned_profile_shape():
    rng = np.random.default_rng(0)
    pt = np.linspace(0, 1, 50)
    mat = rng.random((50, 6))
    prof = heatmap.binned_profile(pt, mat, n_bins=10)
    assert prof.shape == (10, 6)


def test_peak_bin_order_sorts_by_argmax():
    # gene 0 peaks at bin 2, gene 1 at bin 0, gene 2 at bin 1
    combined = np.zeros((3, 3))
    combined[2, 0] = 1
    combined[0, 1] = 1
    combined[1, 2] = 1
    order = heatmap.peak_bin_order(combined)
    assert list(order) == [1, 2, 0]


def test_condition_split_heatmap_builds_figure():
    n_bins, n_genes = 20, 5
    prof = np.tile(np.linspace(0, 1, n_bins)[:, None], (1, n_genes))
    profiles = {"Normal": prof, "LE": prof[::-1]}
    score = np.linspace(0, 1, n_bins)
    state = np.zeros(n_bins, dtype=int)
    tracks = {"Normal": (score, state), "LE": (score, state)}
    order = np.arange(n_genes)
    fig = heatmap.condition_split_heatmap(
        profiles,
        tracks,
        [f"G{i}" for i in range(n_genes)],
        order,
        condition_order=["Normal", "LE"],
        state_cats=["Basal"],
        state_colors=["#4C72B0"],
        present_state_codes=[0],
    )
    # 2 conditions x 4 rows (pt, score, state, expr) = 8 primary axes (+ colorbars)
    assert len(fig.axes) >= 8


def test_condition_split_heatmap_single_panel_when_one_condition():
    n_bins, n_genes = 10, 3
    prof = np.tile(np.linspace(0, 1, n_bins)[:, None], (1, n_genes))
    profiles = {"all": prof}
    tracks = {"all": (np.full(n_bins, np.nan), np.full(n_bins, -1))}
    fig = heatmap.condition_split_heatmap(
        profiles,
        tracks,
        [f"G{i}" for i in range(n_genes)],
        np.arange(n_genes),
        condition_order=["all"],
        state_cats=[],
        state_colors=[],
        present_state_codes=[],
    )
    assert len(fig.axes) >= 2  # pseudotime gradient row + expression row at least
