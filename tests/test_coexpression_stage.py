"""Tests for CoexpressionStage."""

from __future__ import annotations

from types import SimpleNamespace

from cellquorum.stages.gene_regulation.coexpression.stage import CoexpressionStage


def test_stage_identity() -> None:
    s = CoexpressionStage()
    assert s.name == "coexpression"
    assert s.stage_category == "coexpression"


def test_default_method_name() -> None:
    s = CoexpressionStage()
    assert s._select_method_name({}) == "hdwgcna"
    assert s._select_method_name({"method": "other"}) == "other"


def test_augment_fills_condition_from_design() -> None:
    s = CoexpressionStage()
    design = SimpleNamespace(condition_col="grp")
    ctx = SimpleNamespace(config=SimpleNamespace(design=design))
    out = s._augment_config(ctx, {})
    assert out["condition_col"] == "grp"


def test_augment_noop_without_design() -> None:
    s = CoexpressionStage()
    ctx = SimpleNamespace(config=SimpleNamespace(design=None))
    assert s._augment_config(ctx, {"x": 1}) == {"x": 1}
