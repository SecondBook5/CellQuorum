"""Tests for the AnalysisMethod strategy base class."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class _DummyMethod(AnalysisMethod):
    name = "dummy"
    stage_category = "demo"
    backend = "python"

    def input_contract(self, config):
        return DataContract(required_obs=["condition"])

    def min_donors(self):
        return 3

    def _run(self, adata, config, context):
        return StageResult(adata=adata, notes=["ran dummy"])


def _adata(donors):
    obs = pd.DataFrame(
        {"condition": ["Normal"] * len(donors), "patient_id": donors},
        index=[f"c{i}" for i in range(len(donors))],
    )
    return ad.AnnData(X=np.zeros((len(donors), 2), dtype=np.float32), obs=obs)


def test_cannot_instantiate_incomplete_subclass():
    # Missing _run / input_contract => abstract => TypeError at instantiation.
    class Incomplete(AnalysisMethod):
        name = "x"
        stage_category = "demo"
        backend = "python"

    with pytest.raises(TypeError):
        Incomplete()


def test_run_executes_when_guards_pass():
    m = _DummyMethod()
    a = _adata(["P1", "P2", "P3"])
    result = m.run(a, config={}, context=None, donor_col="patient_id")
    assert isinstance(result, StageResult)
    assert "ran dummy" in result.notes


def test_run_skips_below_min_donors():
    m = _DummyMethod()
    a = _adata(["P1", "P2"])  # 2 < min_donors 3
    result = m.run(a, config={}, context=None, donor_col="patient_id")
    assert isinstance(result, MethodSkip)
    assert "min_donors" in result.reason


def test_run_raises_on_contract_violation():
    # Use a method with no min_donors guard so we reach contract validation.
    class _NoGuardMethod(AnalysisMethod):
        name = "no_guard"
        stage_category = "demo"
        backend = "python"

        def input_contract(self, config):
            return DataContract(required_obs=["condition"])

        def _run(self, adata, config, context):
            return StageResult(adata=adata)

    m = _NoGuardMethod()
    a = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))  # no 'condition' obs
    from cellquorum.contracts import CellQuorumContractError

    with pytest.raises(CellQuorumContractError):
        m.run(a, config={}, context=None, donor_col=None)
