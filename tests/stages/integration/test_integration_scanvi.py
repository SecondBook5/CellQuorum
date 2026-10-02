"""Tests for the scANVI integration method (registration + fail-loud gating).

scANVI training itself needs a GPU + scvi-tools, so these tests cover the parts
reachable without a GPU: registry wiring, the config surface, and the fail-loud
paths (no GPU backend, missing label_key).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.exceptions import CellQuorumStageError
from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.integration.scanvi_methods import ScANVIMethod


def _counts_adata(n=60, g=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.negative_binomial(2, 0.3, size=(n, g)).astype(np.float32)
    a = ad.AnnData(X=x)
    a.layers["counts"] = x.copy()
    a.obs["patient_id"] = pd.Categorical(["A", "B"] * (n // 2))
    a.obs["cell_type"] = pd.Categorical((["T"] * (n // 2)) + (["B"] * (n // 2)))
    return a


class _Registry:
    """Minimal registry stub controlling GPU availability."""

    def __init__(self, gpu: bool):
        self._gpu = gpu

    def available(self, name: str) -> bool:
        return self._gpu if name == "gpu" else False


class _Ctx:
    def __init__(self, adata, *, gpu: bool):
        self._adata = adata
        self.backend_registry = _Registry(gpu)
        self.config = {}

    def require_adata(self):
        return self._adata


def test_scanvi_is_registered():
    """scANVI must be resolvable from the method registry (import triggers it)."""
    import cellquorum.stages.integration  # noqa: F401  (registration side effect)

    method_cls = METHOD_REGISTRY.get("integration", "scanvi")
    assert method_cls is ScANVIMethod


def test_scanvi_fails_loud_without_gpu():
    """No GPU backend -> a clear, actionable error (not a raw import failure)."""
    a = _counts_adata()
    ctx = _Ctx(a, gpu=False)

    with pytest.raises(CellQuorumStageError, match="GPU"):
        ScANVIMethod()._run(a, {"batch_key": "patient_id", "label_key": "cell_type"}, ctx)


def test_scanvi_requires_label_key():
    """scANVI is semi-supervised: a missing label_key fails loud, even with GPU."""
    a = _counts_adata()
    ctx = _Ctx(a, gpu=True)

    with pytest.raises(CellQuorumStageError, match="label_key"):
        ScANVIMethod()._run(a, {"batch_key": "patient_id"}, ctx)


def test_scanvi_input_contract_requires_counts_and_labels():
    """The contract requires the counts layer plus batch and label obs columns."""
    contract = ScANVIMethod().input_contract({"batch_key": "patient_id", "label_key": "cell_type"})
    assert "counts" in contract.required_layers
    assert "patient_id" in contract.required_obs
    assert "cell_type" in contract.required_obs
