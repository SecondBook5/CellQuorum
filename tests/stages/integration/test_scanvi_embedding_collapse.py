"""scANVI needs the same degenerate-embedding check scVI has -- it trains an extra
model stage (scVI base + a semi-supervised scANVI head) and previously had NO
collapse check at all, unlike its sibling.

Uses stub scvi.model.SCVI/SCANVI classes rather than real GPU training, so the
collapse-detection logic is tested in isolation, deterministically, without a CUDA device.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.exceptions import CellQuorumStageError
from cellquorum.stages.integration.scanvi_methods import ScANVIMethod


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
    adata.obs["cell_type"] = ["t1"] * (n // 2) + ["t2"] * (n // 2)
    return adata


class _StubSCVI:
    def __init__(self, adata, n_latent, **kwargs):
        pass

    @staticmethod
    def setup_anndata(adata, batch_key=None, **kwargs):
        return None

    def train(self, max_epochs=None, **kwargs):
        return None


class _StubSCANVI:
    _next_latent: np.ndarray | None = None

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_scvi_model(cls, vae, unlabeled_category=None, labels_key=None, **kwargs):
        return cls()

    def train(self, max_epochs=None, **kwargs):
        return None

    def get_latent_representation(self, adata=None):
        return _StubSCANVI._next_latent


def _run_scanvi_with_stub_latent(monkeypatch, latent: np.ndarray):
    import scvi

    monkeypatch.setattr(scvi.model, "SCVI", _StubSCVI)
    monkeypatch.setattr(scvi.model, "SCANVI", _StubSCANVI)
    _StubSCANVI._next_latent = latent

    adata = _cohort(n=latent.shape[0])
    return ScANVIMethod()._run(
        adata,
        {
            "batch_key": "patient_id",
            "label_key": "cell_type",
            "n_latent": latent.shape[1],
        },
        context=_Context(),
    )


def test_scanvi_fatal_collapse_raises(monkeypatch):
    n, n_latent = 60, 10
    latent = np.zeros((n, n_latent), dtype=np.float32)

    with pytest.raises(CellQuorumStageError, match="collapsed"):
        _run_scanvi_with_stub_latent(monkeypatch, latent)


def test_scanvi_partial_collapse_is_a_stage_result_warning(monkeypatch):
    n, n_latent = 60, 10
    rng = np.random.default_rng(0)
    latent = np.zeros((n, n_latent), dtype=np.float32)
    latent[:, :3] = rng.normal(size=(n, 3))

    result = _run_scanvi_with_stub_latent(monkeypatch, latent)

    assert any(
        "low effective dimensionality" in w for w in result.warnings
    ), f"got: {result.warnings}"


def test_scanvi_healthy_embedding_produces_no_collapse_warning(monkeypatch):
    n, n_latent = 60, 10
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(n, n_latent)).astype(np.float32)

    result = _run_scanvi_with_stub_latent(monkeypatch, latent)

    assert not any("dimensionality" in w or "collapsed" in w for w in result.warnings)
