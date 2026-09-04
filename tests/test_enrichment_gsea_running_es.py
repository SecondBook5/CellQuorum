"""Tests for the GSEA running-ES walk helper and its persisted CSV."""

import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.comparative.enrichment.gsea_method import (
    leading_edge,
    running_es_walk,
)


def test_running_es_walk_endpoint_returns_near_zero():
    # Metric ranked descending; a mid-list gene set.
    metric = pd.Series({f"g{i}": 10.0 - i for i in range(10)})
    members = {"g2", "g3", "g4"}
    walk = running_es_walk(metric, members)
    assert list(walk.columns) == ["rank", "running_es", "hit", "metric"]
    assert len(walk) == 10
    # Walk ends at ~0 (P_hit and P_miss both reach 1).
    assert abs(float(walk["running_es"].iloc[-1])) < 1e-9
    # Hit count matches membership present in the ranking.
    assert int(walk["hit"].sum()) == 3
    # Ranks are 1..N ascending.
    assert list(walk["rank"]) == list(range(1, 11))


def test_running_es_walk_hits_accumulate_monotonically():
    metric = pd.Series({f"g{i}": 10.0 - i for i in range(10)})
    members = {"g0", "g1", "g2"}  # top of the list → positive peak early
    walk = running_es_walk(metric, members)
    peak = float(walk["running_es"].max())
    assert peak > 0
    # cumulative hit column is nondecreasing
    assert (np.diff(walk["hit"].cumsum().to_numpy()) >= 0).all()


def test_running_es_walk_degenerate_returns_none():
    metric = pd.Series({f"g{i}": 10.0 - i for i in range(5)})
    assert running_es_walk(metric, set()) is None  # no hits
    assert running_es_walk(metric, {f"g{i}" for i in range(5)}) is None  # all hits


# --------------------------------------------------------------------------- #
# the leading edge: which genes carry the enrichment
# --------------------------------------------------------------------------- #


def test_the_leading_edge_and_the_plotted_walk_agree_on_the_score():
    """The anti-drift guard for the whole feature.

    `leading_edge` computes the score in closed form over the hit positions because it runs
    for every source in a collection; `running_es_walk` walks every gene because it is drawn.
    Two implementations of one quantity drift, and if they do, the table's score and the
    figure's peak stop describing the same thing. So they are pinned against each other over
    many random sets, including sets at the top, the bottom, and the middle of the list.
    """
    rng = np.random.default_rng(0)
    metric = pd.Series({f"g{i}": float(v) for i, v in enumerate(rng.normal(size=120))})
    for _ in range(40):
        size = int(rng.integers(2, 40))
        members = set(rng.choice(metric.index.to_numpy(), size, replace=False))
        found = leading_edge(metric, members)
        walk = running_es_walk(metric, members)
        assert (found is None) == (walk is None)
        if found is None:
            continue
        score, _ = found
        running = walk["running_es"].to_numpy(dtype=float)
        # The score is the running value furthest from zero, in whichever direction.
        expected = max(running.max(), 0.0)
        if -min(running.min(), 0.0) > expected:
            expected = min(running.min(), 0.0)
        assert score == pytest.approx(expected, abs=1e-9)


def test_a_set_at_the_top_of_the_list_scores_positive_and_leads_with_those_genes():
    metric = pd.Series({f"g{i}": 10.0 - i for i in range(20)})
    score, genes = leading_edge(metric, {"g0", "g1", "g2"})
    assert score > 0
    # The peak is at the last of the three hits, so all three carry the enrichment.
    assert genes == ["g0", "g1", "g2"]


def test_a_set_at_the_bottom_of_the_list_scores_negative_and_leads_from_the_trough():
    metric = pd.Series({f"g{i}": 10.0 - i for i in range(20)})
    score, genes = leading_edge(metric, {"g17", "g18", "g19"})
    assert score < 0
    assert genes == ["g17", "g18", "g19"]


def test_only_the_genes_before_the_peak_are_in_the_leading_edge():
    """A set split between the top and the bottom of the list peaks at the top group, and
    the trailing members are not what carried it."""
    metric = pd.Series({f"g{i}": 100.0 - i for i in range(60)})
    score, genes = leading_edge(metric, {"g0", "g1", "g2", "g3", "g50", "g55"})
    assert score > 0
    assert genes == ["g0", "g1", "g2", "g3"]


def test_the_leading_edge_is_ranked_not_alphabetical():
    """A reader takes the first n genes as the strongest contributors, so the order is the
    ranking's, not the set's or the sort's."""
    metric = pd.Series({"zeta": 5.0, "alpha": 4.0, "mu": 3.0, "beta": -1.0, "kappa": -2.0})
    _, genes = leading_edge(metric, {"alpha", "zeta", "mu"})
    assert genes == ["zeta", "alpha", "mu"]


def test_a_degenerate_set_has_no_leading_edge():
    metric = pd.Series({f"g{i}": 10.0 - i for i in range(5)})
    assert leading_edge(metric, set()) is None
    assert leading_edge(metric, {f"g{i}" for i in range(5)}) is None


def test_a_set_carrying_no_weight_has_no_leading_edge():
    """All-zero metric values on the hits: there is nothing for the walk to accumulate, and
    dividing by the total weight would be a divide by zero."""
    metric = pd.Series({"a": 0.0, "b": 0.0, "c": 3.0, "d": -3.0})
    assert leading_edge(metric, {"a", "b"}) is None
