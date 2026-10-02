"""Tests for the analysis-method registry."""

from __future__ import annotations

import pytest

from cellquorum.core.contracts import CellQuorumContractError, DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.methods.registry import MethodRegistry


class _MethodA(AnalysisMethod):
    name = "soupx"
    stage_category = "ambient_correction"
    backend = "r"

    def input_contract(self, config):
        return DataContract()

    def _run(self, adata, config, context):
        return StageResult(adata=adata)


def test_register_and_get():
    reg = MethodRegistry()
    reg.register(_MethodA)
    assert reg.get("ambient_correction", "soupx") is _MethodA


def test_names_for_category():
    reg = MethodRegistry()
    reg.register(_MethodA)
    assert reg.names("ambient_correction") == ["soupx"]


def test_get_unknown_raises():
    reg = MethodRegistry()
    with pytest.raises(CellQuorumContractError, match="decontx"):
        reg.get("ambient_correction", "decontx")


def test_duplicate_registration_raises():
    reg = MethodRegistry()
    reg.register(_MethodA)
    with pytest.raises(CellQuorumContractError, match="already registered"):
        reg.register(_MethodA)


def test_has_reports_registration():
    reg = MethodRegistry()
    reg.register(_MethodA)
    assert reg.has("ambient_correction", "soupx") is True
    assert reg.has("ambient_correction", "decontx") is False
