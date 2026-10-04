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


def test_normalize_profiles_pooled_preserves_cross_condition_magnitude():
    """Real bug: normalizing each condition panel to its OWN 0-1 range hides a
    true magnitude difference between conditions -- a gene uniformly higher
    in one condition still gets stretched to look identical to every other
    condition. Pooling min/max across all conditions before normalizing must
    keep that real difference visible: a gene at a constant LOW value in one
    condition and a constant HIGH value in another must not both end up at a
    flat 1.0 (or a flat 0.0) once normalized.
    """
    # One gene, two bins, two conditions. Condition "lo" sits at 1-2,
    # condition "hi" sits at 8-9 -- a real magnitude gap.
    profiles = {
        "lo": np.array([[1.0], [2.0]]),
        "hi": np.array([[8.0], [9.0]]),
    }
    normed = heatmap.normalize_profiles_pooled(profiles)
    assert normed["lo"].max() < normed["hi"].min(), "lo condition must stay visibly lower than hi"
    # Global min/max (1 and 9) anchor the 0-1 scale, not each condition's own.
    assert np.isclose(normed["lo"].min(), 0.0)
    assert np.isclose(normed["hi"].max(), 1.0)


def test_normalize_profiles_pooled_handles_a_constant_gene():
    profiles = {"a": np.array([[5.0], [5.0]]), "b": np.array([[5.0], [5.0]])}
    normed = heatmap.normalize_profiles_pooled(profiles)
    assert np.isfinite(normed["a"]).all()
    assert np.isfinite(normed["b"]).all()


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


def test_condition_split_heatmap_labels_genes_with_leader_lines_on_the_last_panel():
    """Reference-figure ask: gene names for a dense heatmap (dozens of rows)
    read better as evenly-spaced text connected to its real row by a thin
    leader line than as a plain, easily-cramped y-tick list -- and they
    belong on the RIGHTMOST condition panel (matching the reference, which
    puts one shared gene-label margin after the last condition column), not
    the first.
    """
    n_bins, n_genes = 20, 5
    prof = np.tile(np.linspace(0, 1, n_bins)[:, None], (1, n_genes))
    profiles = {"Normal": prof, "LE": prof[::-1]}
    gene_labels = [f"G{i}" for i in range(n_genes)]
    fig = heatmap.condition_split_heatmap(
        profiles, {}, [], gene_labels, np.arange(n_genes), condition_order=["Normal", "LE"]
    )
    expr_axes = [ax for ax in fig.axes if ax.get_xlabel().startswith("pseudotime")]
    assert len(expr_axes) == 2
    first_ax, last_ax = expr_axes[0], expr_axes[1]

    # No cramped y-tick list on either panel any more.
    assert first_ax.get_yticklabels() == [] or all(
        t.get_text() == "" for t in first_ax.get_yticklabels()
    )
    # The leader-line labels themselves live on the LAST panel, one per gene.
    last_texts = {t.get_text() for t in last_ax.texts}
    assert set(gene_labels) <= last_texts
    first_texts = {t.get_text() for t in first_ax.texts}
    assert not (set(gene_labels) & first_texts), "gene labels leaked onto the first panel"
    # A thin leader line connects each label back to its row.
    assert len(last_ax.lines) >= n_genes


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


def test_condition_split_heatmap_last_panel_is_as_wide_as_the_tracks_above_it():
    """Real bug reported on the fibroblast figure: in the last condition panel the expression image
    ended well before the pseudotime/subcluster tracks above it. The gene labels sit outside the
    axes, and drawing them there widened the x-limits, so the fixed-extent image no longer filled
    its axes. Every column must span the same bins as the tracks stacked over it."""
    n_bins, n_genes = 40, 6
    prof = np.tile(np.linspace(0, 1, n_bins)[:, None], (1, n_genes))
    profiles = {"Normal": prof, "LE": prof[::-1]}
    cat = {
        "label": "Subcluster",
        "cats": ["a", "b"],
        "colors": ["#1b4f8a", "#c41e3a"],
        "tracks": {c: (np.arange(n_bins) >= n_bins // 2).astype(int) for c in profiles},
        "present_codes": [0, 1],
    }
    fig = heatmap.condition_split_heatmap(
        profiles,
        {},
        [cat],
        [f"G{i}" for i in range(n_genes)],
        np.arange(n_genes),
        condition_order=["Normal", "LE"],
    )
    fig.canvas.draw()
    expr_axes = [ax for ax in fig.axes if ax.get_xlabel().startswith("pseudotime")]
    assert len(expr_axes) == 2
    for ax in expr_axes:
        assert ax.get_xlim() == (-0.5, n_bins - 0.5), ax.get_xlim()


def _profile_peaking_at(peaks, n_bins=20):
    """One column per gene, each peaking at the given bin."""
    prof = np.zeros((n_bins, len(peaks)))
    for j, p in enumerate(peaks):
        prof[p, j] = 1.0
    return prof


def test_grouped_peak_order_keeps_groups_together_and_in_declared_order():
    """Genes are grouped by what they are, in the order the figure's story needs; within a group
    they run by where they peak along pseudotime. Peak order alone interleaves the biology."""
    labels = ["A1", "A2", "B1", "B2", "C1"]
    combined = _profile_peaking_at([15, 3, 10, 1, 7])
    groups = {"Late": ["A1", "B1"], "Early": ["A2", "B2"], "Mid": ["C1"]}
    order, row_groups = heatmap.grouped_peak_order(combined, labels, groups)
    assert [labels[i] for i in order] == ["B1", "A1", "B2", "A2", "C1"]
    assert row_groups == ["Late", "Late", "Early", "Early", "Mid"]


def test_grouped_peak_order_ignores_group_members_that_are_not_columns():
    labels = ["A", "B"]
    combined = _profile_peaking_at([4, 2])
    order, row_groups = heatmap.grouped_peak_order(combined, labels, {"G": ["B", "MISSING", "A"]})
    assert [labels[i] for i in order] == ["B", "A"]
    assert row_groups == ["G", "G"]


def test_condition_split_heatmap_labels_each_gene_group_once_and_separates_them():
    n_bins, n_genes = 30, 6
    prof = np.tile(np.linspace(0, 1, n_bins)[:, None], (1, n_genes))
    profiles = {"Normal": prof, "LE": prof[::-1]}
    groups = ["Identity", "Identity", "TGF-beta", "TGF-beta", "TGF-beta", "Matrix"]
    fig = heatmap.condition_split_heatmap(
        profiles,
        {},
        [],
        [f"G{i}" for i in range(n_genes)],
        np.arange(n_genes),
        condition_order=["Normal", "LE"],
        gene_group_labels=groups,
    )
    fig.canvas.draw()
    expr_axes = [ax for ax in fig.axes if ax.get_xlabel().startswith("pseudotime")]
    drawn = [t.get_text() for t in expr_axes[0].texts]
    for name in ("Identity", "TGF-beta", "Matrix"):
        assert drawn.count(name) == 1, (name, drawn)
    # Grouping must not undo the width alignment with the tracks above.
    for ax in expr_axes:
        assert ax.get_xlim() == (-0.5, n_bins - 0.5)
    # A divider between consecutive groups, drawn on every panel: 2 boundaries x 2 panels.
    dividers = [
        ln
        for ax in expr_axes
        for ln in ax.lines
        if len(set(ln.get_ydata())) == 1 and ln.get_xdata()[0] <= -0.5 + 1e-9
    ]
    assert len(dividers) == 4


def test_condition_split_heatmap_rejects_group_labels_of_the_wrong_length():
    import pytest

    prof = np.tile(np.linspace(0, 1, 10)[:, None], (1, 3))
    with pytest.raises(ValueError, match="gene_group_labels"):
        heatmap.condition_split_heatmap(
            {"Normal": prof},
            {},
            [],
            ["a", "b", "c"],
            np.arange(3),
            condition_order=["Normal"],
            gene_group_labels=["only", "two"],
        )
