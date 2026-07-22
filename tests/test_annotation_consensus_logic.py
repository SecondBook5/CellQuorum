"""Unit tests for consensus reconciliation pure functions."""

from __future__ import annotations

from cellquorum.annotation_consensus.consensus import normalize_label, reconcile_votes


def test_normalize_label_applies_alias():
    aliases = {"T cell": "T/NK", "keratinocyte": "Keratinocytes"}
    assert normalize_label("T cell", aliases) == "T/NK"
    assert normalize_label("Keratinocytes", aliases) == "Keratinocytes"  # passthrough
    assert normalize_label(None, aliases) is None


def test_all_agree_is_high():
    label, tier, needs = reconcile_votes(
        ["T/NK", "T/NK", "T/NK"], min_agree_fraction=0.5, high_confidence_all=True
    )
    assert label == "T/NK"
    assert tier == "high"
    assert needs is False


def test_two_of_three_is_medium():
    label, tier, needs = reconcile_votes(
        ["T/NK", "T/NK", "Fibroblasts"], min_agree_fraction=0.5, high_confidence_all=True
    )
    assert label == "T/NK"
    assert tier == "medium"
    assert needs is False


def test_three_way_split_is_low():
    label, tier, needs = reconcile_votes(
        ["T/NK", "Fibroblasts", "DC"], min_agree_fraction=0.5, high_confidence_all=True
    )
    assert tier == "low"
    assert needs is True


def test_missing_votes_tolerated():
    # One method skipped (None). Two remaining agree -> high.
    label, tier, needs = reconcile_votes(
        ["Mast", "Mast", None], min_agree_fraction=0.5, high_confidence_all=True
    )
    assert label == "Mast"
    assert tier == "high"


def test_single_vote_is_low_confidence():
    # Only one non-missing vote -> not enough to be confident.
    label, tier, needs = reconcile_votes(
        [None, None, "B cells"], min_agree_fraction=0.5, high_confidence_all=True
    )
    assert label == "B cells"
    assert tier == "low"
    assert needs is True


def test_no_votes_returns_none():
    label, tier, needs = reconcile_votes(
        [None, None, None], min_agree_fraction=0.5, high_confidence_all=True
    )
    assert label is None
    assert tier == "low"
    assert needs is True
