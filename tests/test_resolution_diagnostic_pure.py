"""Scratch RED-phase test for the pure Jaccard-matching helpers, before wiring into the
real module. Copied into tests/ once the module exists."""

from __future__ import annotations

import pandas as pd
import pytest

from cellquorum.stages.clustering.resolution_diagnostic import (
    _best_match_jaccard,
    _cluster_membership,
)


def test_cluster_membership_groups_by_value():
    labels = pd.Series({"c1": "0", "c2": "0", "c3": "1"})
    got = _cluster_membership(labels)
    assert got == {"0": {"c1", "c2"}, "1": {"c3"}}


def test_best_match_jaccard_picks_the_highest_overlap():
    reference = {"a", "b", "c"}
    subsample_clusters = {
        "x": {"a", "b"},  # jaccard = 2/3
        "y": {"a", "b", "c", "d"},  # jaccard = 3/4
    }
    assert _best_match_jaccard(reference, subsample_clusters) == 3 / 4


def test_best_match_jaccard_perfect_match_is_one():
    reference = {"a", "b"}
    subsample_clusters = {"x": {"a", "b"}}
    assert _best_match_jaccard(reference, subsample_clusters) == 1.0


def test_best_match_jaccard_no_overlap_is_zero():
    reference = {"a", "b"}
    subsample_clusters = {"x": {"c", "d"}}
    assert _best_match_jaccard(reference, subsample_clusters) == 0.0


def test_best_match_jaccard_empty_reference_is_none_not_zero():
    """An empty reference (every member resampled out) is 'not evaluable', not 'destroyed'."""
    assert _best_match_jaccard(set(), {"x": {"a"}}) is None


def test_summarize_resolution_stability_computes_median_and_iqr():
    from cellquorum.stages.clustering.resolution_diagnostic import summarize_resolution_stability

    long_df = pd.DataFrame(
        {
            "resolution": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            "cluster": ["0", "0", "1", "0", "1", "1"],
            "bootstrap": [0, 1, 0, 0, 0, 1],
            "jaccard": [1.0, 0.8, 0.6, 0.5, 0.4, 0.6],
        }
    )
    n_clusters = {0.5: 2, 1.0: 3}

    summary = summarize_resolution_stability(long_df, n_clusters)

    assert list(summary["resolution"]) == [0.5, 1.0]
    assert list(summary["n_clusters"]) == [2, 3]
    row_05 = summary[summary["resolution"] == 0.5].iloc[0]
    assert row_05["median_jaccard"] == pytest.approx(0.8)
    assert row_05["n_bootstrap_observations"] == 3


def test_summarize_resolution_stability_handles_an_empty_long_df():
    from cellquorum.stages.clustering.resolution_diagnostic import summarize_resolution_stability

    summary = summarize_resolution_stability(
        pd.DataFrame(columns=["resolution", "cluster", "bootstrap", "jaccard"]), {0.5: 2, 1.0: 3}
    )

    assert list(summary["resolution"]) == [0.5, 1.0]
    assert list(summary["n_bootstrap_observations"]) == [0, 0]
    assert summary["median_jaccard"].isna().all()
