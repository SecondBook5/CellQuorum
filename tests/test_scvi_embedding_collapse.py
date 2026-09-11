"""scVI's degenerate-embedding checks must actually reach the user.

The fatal collapse (< 2 variable dimensions) raises, which is loud by construction. The
partial-collapse case used Python's warnings.warn instead of StageResult.warnings -- which
this pipeline's own reporting never captures, so the signal a low-effective-dimensionality
embedding exists could go completely unseen in a batch run, a notebook that doesn't display
warnings, or simply Python's own default "once per location" warning dedup.

Uses a stub scvi.model.SCVI rather than real GPU training, so the collapse-detection logic
itself is tested in isolation, deterministically, without needing a CUDA device.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.exceptions import CellQuorumStageError
from cellquorum.stages.integration.scvi_methods import ScVIMethod


class _Registry:
    def available(self, name: str) -> bool:
        return name == "gpu"


class _Context:
    backend_registry = _Registry()


def _cohort(n: int = 60, n_genes: int = 40, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=(n, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=counts.copy())
    adata.layers["counts"] = counts
    adata.obs["patient_id"] = ["a"] * (n // 2) + ["b"] * (n // 2)
    return adata


class _StubSCVIModel:
    """Replaces scvi.model.SCVI: no real training, a controlled latent output."""

    _next_latent: np.ndarray | None = None

    def __init__(self, adata, n_latent, **kwargs):
        self._adata = adata
        self.n_latent = n_latent

    @staticmethod
    def setup_anndata(adata, batch_key=None, **kwargs):
        return None

    def train(self, max_epochs=None, **kwargs):
        return None

    def get_latent_representation(self, adata=None):
        return _StubSCVIModel._next_latent


def _run_scvi_with_stub_latent(monkeypatch, latent: np.ndarray):
    import scvi

    monkeypatch.setattr(scvi.model, "SCVI", _StubSCVIModel)
    _StubSCVIModel._next_latent = latent

    adata = _cohort(n=latent.shape[0])
    return ScVIMethod()._run(
        adata,
        {"batch_key": "patient_id", "n_latent": latent.shape[1], "use_highly_variable": False},
        context=_Context(),
    )


def test_fatal_collapse_raises(monkeypatch):
    n, n_latent = 60, 10
    latent = np.zeros((n, n_latent), dtype=np.float32)  # every dimension constant

    with pytest.raises(CellQuorumStageError, match="collapsed"):
        _run_scvi_with_stub_latent(monkeypatch, latent)


def test_partial_collapse_is_a_stage_result_warning_not_a_python_warning(monkeypatch):
    n, n_latent = 60, 10
    rng = np.random.default_rng(0)
    latent = np.zeros((n, n_latent), dtype=np.float32)
    latent[:, :3] = rng.normal(size=(n, 3))  # only 3 of 10 dims carry variance

    result = _run_scvi_with_stub_latent(monkeypatch, latent)

    assert any(
        "low effective dimensionality" in w for w in result.warnings
    ), f"partial collapse must surface in StageResult.warnings, got: {result.warnings}"


def test_a_healthy_embedding_produces_no_collapse_warning(monkeypatch):
    n, n_latent = 60, 10
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(n, n_latent)).astype(np.float32)

    result = _run_scvi_with_stub_latent(monkeypatch, latent)

    assert not any("dimensionality" in w or "collapsed" in w for w in result.warnings)
