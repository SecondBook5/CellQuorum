"""Tests for the GRN stage dispatch + method registration."""

from __future__ import annotations

from cellquorum.gene_regulation.grn.stage import GrnStage
from cellquorum.methods.registry import METHOD_REGISTRY


def test_stage_identity() -> None:
    s = GrnStage()
    assert s.name == "grn"
    assert s.stage_category == "grn"


def test_selects_pyscenic_by_default() -> None:
    s = GrnStage()
    assert s._select_method_name({}) == "pyscenic"
    assert s._select_method_name({"method": "pyscenic"}) == "pyscenic"


def test_method_is_registered() -> None:
    import cellquorum.gene_regulation.grn  # noqa: F401

    assert METHOD_REGISTRY.has("grn", "pyscenic")
