"""Tests for the metacells stage dispatch + method registration."""

from __future__ import annotations

from cellquorum.stages.metacells.stage import MetacellsStage


def test_stage_identity() -> None:
    s = MetacellsStage()
    assert s.name == "metacells"
    assert s.stage_category == "metacells"


def test_selects_metacells_by_default() -> None:
    s = MetacellsStage()
    assert s._select_method_name({}) == "metacells"
    assert s._select_method_name({"method": "metacells"}) == "metacells"
