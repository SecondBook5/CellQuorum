"""Test the ScArchesMethod (scVI→scANVI→surgery with multi-seed consensus)."""

from __future__ import annotations

import types
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from cellquorum.contracts import CellQuorumContractError, set_layer_tag
from cellquorum.methods.base import MethodSkip
from cellquorum.reference_mapping.scarches import ScArchesMethod


def _synth(n: int, seed: int, labels: bool = True) -> ad.AnnData:
    """Build a tiny synthetic anndata for CPU-fast training."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n, 50)).astype("float32")
    a = ad.AnnData(X=X)
    a.layers["counts"] = X.copy()
    set_layer_tag(a, "counts", kind="counts", recipe=None)
    a.var_names = [f"g{i}" for i in range(50)]
    a.obs_names = [f"{seed}_c{i}" for i in range(n)]
    a.obs["batch"] = rng.choice(["b0", "b1"], n)
    if labels:
        a.obs["cell_type"] = rng.choice(["A", "B", "C"], n)
    return a


class _Ctx:
    """Minimal context for method testing."""

    def __init__(self, tmp: Path) -> None:
        self.paths = types.SimpleNamespace(objects=tmp, figures=tmp, results=tmp, scratch=tmp)
        self.backend_registry = None
        self.config = None


def _cfg(atlas_path: Path | str, **over: object) -> dict:
    """Build a minimal config with CPU-fast training settings."""
    base = {
        "method": "scarches",
        "atlas_h5ad": str(atlas_path),
        "label_key": "cell_type",
        "atlas_batch_key": "batch",
        "counts_layer": "counts",
        "n_top_genes": 30,
        "force_genes": [],
        "n_latent": 10,
        "n_layers": 1,
        "max_epochs_scvi": 2,
        "max_epochs_scanvi": 2,
        "max_epochs_query": 2,
        "seeds": [0],
        "knn_k": 5,
        "key_added": "ref_state",
        "write_loss_curves": False,
        "unlabeled_category": "Unknown",
        "query_batch_value": "query",
        "hvg_flavor": "seurat_v3",
        "dropout_rate": 0.2,
        "gene_likelihood": "zinb",
        "early_stopping": True,
        "query_early_stopping_patience": 10,
        "query_early_stopping_monitor": "reconstruction_loss_train",
        "compute_backend": "cpu",
    }
    base.update(over)
    return base


def test_scarches_transfers_labels_and_uncertainty(tmp_path: Path) -> None:
    """ScArchesMethod should transfer labels + write kNN entropy/agreement."""
    atlas = _synth(400, 0)
    atlas.write_h5ad(tmp_path / "atlas.h5ad")
    query = _synth(120, 1, labels=False)
    res = ScArchesMethod().run(query, _cfg(tmp_path / "atlas.h5ad"), context=_Ctx(tmp_path))
    assert not isinstance(res, MethodSkip)
    obs = res.adata.obs
    assert "ref_state" in obs
    assert "ref_state_knn_entropy" in obs
    assert "ref_state_knn_agreement" in obs
    assert "X_scANVI" in res.adata.obsm
    assert "knn_accuracy" in res.metrics


def test_scarches_skips_when_atlas_missing(tmp_path: Path) -> None:
    """ScArchesMethod should return MethodSkip if atlas is missing."""
    query = _synth(50, 1, labels=False)
    res = ScArchesMethod().run(query, _cfg(tmp_path / "nope.h5ad"), context=_Ctx(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "atlas" in res.reason.lower() or "missing" in res.reason.lower()


def test_scarches_requires_counts_layer(tmp_path: Path) -> None:
    """ScArchesMethod should fail-loud if counts layer is absent."""
    atlas = _synth(200, 0)
    atlas.write_h5ad(tmp_path / "atlas.h5ad")
    query = _synth(60, 1, labels=False)
    del query.layers["counts"]
    with pytest.raises(CellQuorumContractError):
        ScArchesMethod().run(query, _cfg(tmp_path / "atlas.h5ad"), context=_Ctx(tmp_path))


def test_scarches_multiseed_consensus(tmp_path: Path) -> None:
    """ScArchesMethod should run multi-seed consensus + write per-seed cols."""
    atlas = _synth(400, 0)
    atlas.write_h5ad(tmp_path / "atlas.h5ad")
    query = _synth(120, 1, labels=False)
    res = ScArchesMethod().run(
        query, _cfg(tmp_path / "atlas.h5ad", seeds=[0, 1]), context=_Ctx(tmp_path)
    )
    assert not isinstance(res, MethodSkip)
    assert "ref_state_consensus_frac" in res.adata.obs
    assert "ref_state_seed0" in res.adata.obs
    assert "ref_state_seed1" in res.adata.obs
