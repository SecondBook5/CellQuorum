"""Tests for the analysis-method registry."""

from __future__ import annotations

import csv
from importlib.util import find_spec

import pytest
from repo_paths import REPO_ROOT

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


def test_public_study_methods_resolve_through_the_stage_registry():
    from cellquorum.core.stages import all_stage_specs
    from cellquorum.methods.registry import METHOD_REGISTRY

    specs = {s.name: s for s in all_stage_specs()}
    with (REPO_ROOT / "docs/benchmarks/study-coverage.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert {r["stage"] for r in rows if r["method"] == "(stage)"} == set(specs)
    documented = {(r["stage"], r["method"]) for r in rows if r["method"] != "(stage)"}
    # SoupX runs directly in its stage. scArches registers when the scvi extra is installed.
    documented.remove(("ambient_correction", "soupx"))
    if find_spec("scvi") is None:
        documented.remove(("reference_mapping", "scarches"))
    shipping = {
        (stage, name)
        for stage, spec in specs.items()
        for name in METHOD_REGISTRY.names(spec.category or stage)
        if METHOD_REGISTRY.get(spec.category or stage, name).__module__.startswith("cellquorum.")
    }
    assert shipping == documented
    for stage, name in sorted(documented):
        method = METHOD_REGISTRY.get(specs[stage].category, name)
        assert (method.stage_category, method.name) == (stage, name)
