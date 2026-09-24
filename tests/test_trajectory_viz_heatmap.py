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


def test_bin_masks_uses_rank_not_raw_value_so_skew_does_not_empty_most_bins():
    """Real bug found on the fibroblast cohort: dpt_pseudotime had 99.84% of
    61,601 cells packed into the first 5% of its raw value range, with a
    handful of straggler cells stretching the max out to 1.0. Binning by raw
    VALUE (the old np.linspace(pt.min(), pt.max(), n_bins+1) edges) left
    83/100 bins completely empty; the heatmap's forward-fill then painted
    those empty bins with whichever straggler landed last, producing large,
    confident-looking blocks of color backed by a handful of outlier cells
    rather than real trend. Binning by RANK instead of raw value must keep
    every bin populated regardless of how skewed the raw values are.
    """
    n_dense, n_stray = 995, 5
    pt = np.concatenate([np.linspace(0.0, 0.03, n_dense), np.linspace(0.1, 1.0, n_stray)])
    _, binid = heatmap.bin_masks(pt, n_bins=100)
    counts = np.bincount(binid, minlength=100)
    assert (counts > 0).all(), f"{(counts == 0).sum()} of 100 bins are empty"


def test_binned_profile_shape():
    rng = np.random.default_rng(0)
    pt = np.linspace(0, 1, 50)
    mat = rng.random((50, 6))
    prof = heatmap.binned_profile(pt, mat, n_bins=10)
    assert prof.shape == (10, 6)


def test_binned_score_track_forward_fills_empty_bins():
    pt = np.array([0.0, 0.0, 1.0, 1.0])  # everything lands in bin 0 or bin 3 of 4
    score = np.array([2.0, 2.0, 4.0, 4.0])
    track = heatmap.binned_score_track(pt, score, n_bins=4)
    assert track[0] == 2.0
    assert track[-1] == 4.0
    assert not np.isnan(track).any()  # empty middle bins forward-filled


def test_binned_categorical_track_takes_the_modal_code_per_bin():
    pt = np.linspace(0, 1, 6)
    codes = np.array([0, 0, 0, 1, 1, 1])
    track = heatmap.binned_categorical_track(pt, codes, n_bins=2)
    assert list(track) == [0, 1]


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
    score_tracks = {"Normal": np.linspace(0, 1, n_bins), "LE": np.linspace(0, 1, n_bins)}
    state = np.zeros(n_bins, dtype=int)
    categorical_tracks = [
        {
            "label": "State",
            "cats": ["Basal"],
            "colors": ["#4C72B0"],
            "tracks": {"Normal": state, "LE": state},
            "present_codes": [0],
        }
    ]
    order = np.arange(n_genes)
    fig = heatmap.condition_split_heatmap(
        profiles,
        score_tracks,
        categorical_tracks,
        [f"G{i}" for i in range(n_genes)],
        order,
        condition_order=["Normal", "LE"],
    )
    # 2 conditions x 4 rows (pt, score, state, expr) = 8 primary axes (+ colorbars)
    assert len(fig.axes) >= 8


def test_condition_split_heatmap_two_categorical_tracks():
    """The real ask: subcluster AND cell-cycle-phase stacked as two separate
    categorical rows, alongside pseudotime -- not just the single legacy 'state'
    row the API used to hard-cap at.
    """
    n_bins, n_genes = 15, 4
    prof = np.tile(np.linspace(0, 1, n_bins)[:, None], (1, n_genes))
    profiles = {"all": prof}
    subcluster = np.array([0, 0, 1, 1, 2] * 3)
    phase = np.array([0, 1, 2, 0, 1] * 3)
    categorical_tracks = [
        {
            "label": "Subcluster",
            "cats": ["Perivascular", "Mechanosensory", "Reticular"],
            "colors": ["#4C72B0", "#DD8452", "#55A868"],
            "tracks": {"all": subcluster},
            "present_codes": [0, 1, 2],
        },
        {
            "label": "Cell cycle",
            "cats": ["G1", "S", "G2M"],
            "colors": ["#999999", "#C44E52", "#8172B2"],
            "tracks": {"all": phase},
            "present_codes": [0, 1, 2],
        },
    ]
    fig = heatmap.condition_split_heatmap(
        profiles,
        {},
        categorical_tracks,
        [f"G{i}" for i in range(n_genes)],
        np.arange(n_genes),
        condition_order=["all"],
    )
    # pt + subcluster + cell-cycle + expr = 4 rows, 1 condition column
    assert len(fig.axes) >= 4
    # x-axis of the expression panel must carry real tick labels now, not the
    # blank axis the old figure shipped with. fig.axes[-1] would be the
    # colorbar (added after the main loop), so find the real panel by its label.
    expr_ax = next(ax for ax in fig.axes if ax.get_xlabel().startswith("pseudotime"))
    assert [t.get_text() for t in expr_ax.get_xticklabels()] == [
        "0%",
        "25%",
        "50%",
        "75%",
        "100%",
    ]


def test_condition_split_heatmap_single_panel_when_one_condition():
    n_bins, n_genes = 10, 3
    prof = np.tile(np.linspace(0, 1, n_bins)[:, None], (1, n_genes))
    profiles = {"all": prof}
    fig = heatmap.condition_split_heatmap(
        profiles,
        {},
        [],
        [f"G{i}" for i in range(n_genes)],
        np.arange(n_genes),
        condition_order=["all"],
    )
    assert len(fig.axes) >= 2  # pseudotime gradient row + expression row at least
