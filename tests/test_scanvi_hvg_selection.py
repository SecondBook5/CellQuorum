"""scANVI must restrict training to highly variable genes the same way scVI does.

Before this fix scANVI had no HVG-restriction logic at all: it built the training object
straight from `adata.layers["counts"]` and every gene in `var`, ignoring both
`var['highly_variable']` and `integration.use_highly_variable`. This is the same bug class
scVI's own code comments describe as already fixed there -- unaddressed on its sibling.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.exceptions import CellQuorumStageError
from cellquorum.stages.integration.scanvi_methods import ScANVIMethod


class _Registry:
    def available(self, name: str) -> bool:
        return name == "gpu"


class _Context:
    backend_registry = _Registry()


def _cohort(n: int = 60, n_genes: int = 600, n_hvg: int = 520, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=(n, n_genes)).astype(np.float32)
    genes = [f"G{i}" for i in range(n_genes)]
    adata = ad.AnnData(X=counts.copy(), var=pd.DataFrame(index=genes))
    adata.layers["counts"] = counts
    adata.obs["patient_id"] = ["a"] * (n // 2) + ["b"] * (n // 2)
    adata.obs["cell_type"] = ["t1"] * (n // 2) + ["t2"] * (n // 2)
    if n_hvg is not None:
        mask = np.zeros(n_genes, dtype=bool)
        mask[:n_hvg] = True
        adata.var["highly_variable"] = mask
    return adata


class _CapturingSCVI:
    captured_n_vars: list[int] = []

    def __init__(self, adata, n_latent, **kwargs):
        _CapturingSCVI.captured_n_vars.append(adata.n_vars)

    @staticmethod
    def setup_anndata(adata, batch_key=None, **kwargs):
        _CapturingSCVI.captured_n_vars.append(adata.n_vars)
        return None

    def train(self, max_epochs=None, **kwargs):
        return None


class _StubSCANVI:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_scvi_model(cls, vae, unlabeled_category=None, labels_key=None, **kwargs):
        return cls()

    def train(self, max_epochs=None, **kwargs):
        return None

    def get_latent_representation(self, adata=None):
        n = adata.n_obs if adata is not None else 60
        rng = np.random.default_rng(1)
        return rng.normal(size=(n, 10)).astype(np.float32)


def _run_scanvi(monkeypatch, adata, config: dict | None = None):
    import scvi

    _CapturingSCVI.captured_n_vars = []
    monkeypatch.setattr(scvi.model, "SCVI", _CapturingSCVI)
    monkeypatch.setattr(scvi.model, "SCANVI", _StubSCANVI)

    merged = {"batch_key": "patient_id", "label_key": "cell_type", "n_latent": 10}
    merged.update(config or {})
    result = ScANVIMethod()._run(adata, merged, context=_Context())
    return result, _CapturingSCVI.captured_n_vars


def test_scanvi_restricts_training_to_highly_variable_genes(monkeypatch):
    adata = _cohort(n_genes=600, n_hvg=520)

    result, captured = _run_scanvi(monkeypatch, adata)

    assert captured, "scVI stub was never called"
    assert all(n == 520 for n in captured), captured
    assert any("520" in note for note in result.notes)


def test_scanvi_falls_back_to_all_genes_when_too_few_hvg_flagged(monkeypatch):
    adata = _cohort(n_genes=600, n_hvg=10)

    result, captured = _run_scanvi(monkeypatch, adata)

    assert all(n == 600 for n in captured), captured


def test_scanvi_raises_when_use_highly_variable_requested_but_flag_absent(monkeypatch):
    adata = _cohort(n_genes=50, n_hvg=None)

    with pytest.raises(CellQuorumStageError, match="use_highly_variable"):
        _run_scanvi(monkeypatch, adata, {"use_highly_variable": True})
