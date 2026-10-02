"""TDD tests for cellquorum.stages.comparative.differential_state.compute."""

from __future__ import annotations

import pandas as pd

from cellquorum.stages.comparative.differential_state.compute import (
    drop_underpowered_donor_groups,
    resolve_score_columns,
)


def test_drop_underpowered_donor_groups_keeps_well_powered_rows():
    frame = pd.DataFrame(
        {
            "donor": ["d1"] * 5 + ["d2"] * 5,
            "condition": ["Normal"] * 5 + ["LE"] * 5,
            "value": range(10),
        }
    )
    out = drop_underpowered_donor_groups(
        frame, donor_col="donor", condition_col="condition", min_cells=5
    )
    assert len(out) == 10  # both groups have exactly 5, the floor


def test_drop_underpowered_donor_groups_drops_a_thin_group():
    frame = pd.DataFrame(
        {
            "donor": ["d1"] * 5 + ["d2"] * 3,
            "condition": ["Normal"] * 5 + ["LE"] * 3,
            "value": range(8),
        }
    )
    out = drop_underpowered_donor_groups(
        frame, donor_col="donor", condition_col="condition", min_cells=5
    )
    assert len(out) == 5
    assert set(out["donor"]) == {"d1"}


def test_drop_underpowered_donor_groups_evaluates_per_donor_condition_pair():
    """The SAME donor can pass in one condition and fail in the other."""
    frame = pd.DataFrame(
        {
            "donor": ["d1"] * 5 + ["d1"] * 2,
            "condition": ["Normal"] * 5 + ["LE"] * 2,
            "value": range(7),
        }
    )
    out = drop_underpowered_donor_groups(
        frame, donor_col="donor", condition_col="condition", min_cells=5
    )
    assert len(out) == 5
    assert set(out["condition"]) == {"Normal"}


def test_resolve_score_columns_returns_explicit_list_when_given():
    obs = pd.DataFrame({"state_a": [1.0], "state_b": [2.0], "other": [3.0]})
    assert resolve_score_columns(obs, explicit=["other"], prefix="state_") == ["other"]


def test_resolve_score_columns_auto_detects_prefixed_numeric_columns():
    obs = pd.DataFrame(
        {
            "state_a": [1.0, 2.0],
            "state_b": [3.0, 4.0],
            "state_label": ["x", "y"],  # non-numeric, must be excluded
            "unrelated": [5.0, 6.0],
        }
    )
    assert resolve_score_columns(obs, explicit=None, prefix="state_") == ["state_a", "state_b"]


def test_resolve_score_columns_empty_when_nothing_matches():
    obs = pd.DataFrame({"unrelated": [1.0, 2.0]})
    assert resolve_score_columns(obs, explicit=None, prefix="state_") == []
