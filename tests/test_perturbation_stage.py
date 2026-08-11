"""Tests for the perturbation stage + method registration."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.perturbation.stage import PerturbationStage


def test_stage_identity() -> None:
    s = PerturbationStage()
    assert s.name == "perturbation"
    assert s.stage_category == "perturbation"


def test_selects_celloracle_by_default() -> None:
    s = PerturbationStage()
    assert s._select_method_name({}) == "celloracle"
    assert s._select_method_name({"method": "celloracle"}) == "celloracle"


def test_method_is_registered_on_import() -> None:
    import cellquorum.perturbation  # noqa: F401

    assert METHOD_REGISTRY.has("perturbation", "celloracle")
