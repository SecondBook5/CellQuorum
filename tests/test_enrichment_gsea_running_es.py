"""Tests for the GSEA running-ES walk helper and its persisted CSV."""

import numpy as np
import pandas as pd

from cellquorum.enrichment.gsea_method import running_es_walk


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
