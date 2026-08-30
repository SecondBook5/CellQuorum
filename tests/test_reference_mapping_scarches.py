"""Test the ScArchesMethod (scVI→scANVI→surgery with multi-seed consensus)."""

from __future__ import annotations

import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.annotation.reference_mapping.scarches import (
    ScArchesMethod,
    _load_seed_checkpoint,
    _mean_soft_probabilities,
    _save_seed_checkpoint,
    _scvi_gpu_available,
)
from cellquorum.core.contracts import CellQuorumContractError, set_layer_tag
from cellquorum.methods.base import MethodSkip

# scvi is an optional GPU backend, not installed in the core test tier. Every test
# here exercises the scVI→scANVI surgery path, so skip the whole module cleanly when
# scvi is absent (these run in the cellquorum-gpu env). Placed after the imports so
# ruff's E402 (module-import-not-at-top) stays satisfied.
pytest.importorskip("scvi")


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
        results_dir = tmp / "results"
        results_dir.mkdir(exist_ok=True)
        self.paths = types.SimpleNamespace(
            root=tmp, objects=tmp, figures=tmp, results=results_dir, scratch=tmp
        )
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
    assert "ref_state_probabilities" in res.adata.obsm
    assert res.adata.obsm["ref_state_probabilities"].shape[0] == res.adata.n_obs
    assert res.adata.uns["reference_mapping"]["probability_obsm"] == "ref_state_probabilities"
    assert "knn_accuracy" in res.metrics
    # CRITICAL: returned adata must preserve full gene space (not HVG subset).
    assert res.adata.n_vars == query.n_vars


def test_scarches_skips_when_atlas_missing(tmp_path: Path) -> None:
    """ScArchesMethod should return MethodSkip if atlas is missing."""
    query = _synth(50, 1, labels=False)
    res = ScArchesMethod().run(query, _cfg(tmp_path / "nope.h5ad"), context=_Ctx(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "atlas" in res.reason.lower() or "missing" in res.reason.lower()


def test_scvi_gpu_probe_returns_bool() -> None:
    """The scVI accelerator probe should always return a boolean."""

    assert isinstance(_scvi_gpu_available(), bool)


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
    assert res.adata.n_vars == query.n_vars
    # Mean-posterior consensus: a confidence column consistent with the label,
    # and probabilities documented as a cross-seed consensus.
    assert "ref_state_confidence" in res.adata.obs
    assert res.adata.uns["reference_mapping"]["probability_consensus"] == (
        "mean_posterior_across_seeds"
    )
    # The consensus label must be the argmax of the consensus probability matrix.
    probs = res.adata.obsm["ref_state_probabilities"]
    cols = res.adata.uns["reference_mapping"]["probability_columns"]
    argmax_labels = [cols[i] for i in probs.argmax(axis=1)]
    assert list(res.adata.obs["ref_state"].astype(str)) == argmax_labels


def test_mean_soft_probabilities_averages_and_normalizes():
    """The consensus helper averages seed posteriors and renormalizes rows to 1."""
    seed_predictions = {
        0: {"soft": pd.DataFrame({"A": [0.8, 0.2], "B": [0.2, 0.8]}, index=["c0", "c1"])},
        1: {"soft": pd.DataFrame({"A": [0.6, 0.1], "B": [0.4, 0.9]}, index=["c0", "c1"])},
    }
    mean = _mean_soft_probabilities(seed_predictions, [0, 1])
    # Mean of A for c0 = (0.8 + 0.6) / 2 = 0.7.
    assert mean.loc["c0", "A"] == pytest.approx(0.7)
    # Every row is a valid distribution.
    np.testing.assert_allclose(mean.sum(axis=1).to_numpy(), [1.0, 1.0])


def test_mean_soft_probabilities_aligns_mismatched_label_columns():
    """Seeds with different/partial label columns align to the union set."""
    seed_predictions = {
        0: {"soft": pd.DataFrame({"A": [1.0], "B": [0.0]}, index=["c0"])},
        1: {"soft": pd.DataFrame({"B": [0.5], "C": [0.5]}, index=["c0"])},
    }
    mean = _mean_soft_probabilities(seed_predictions, [0, 1])
    assert set(mean.columns) == {"A", "B", "C"}
    assert mean.sum(axis=1).to_numpy()[0] == pytest.approx(1.0)


def test_scarches_skips_on_no_shared_genes(tmp_path: Path) -> None:
    """ScArchesMethod should MethodSkip if atlas and query have no shared genes."""
    atlas = _synth(200, 0)
    atlas.var_names = [f"atlas_g{i}" for i in range(50)]
    atlas.write_h5ad(tmp_path / "atlas.h5ad")
    query = _synth(60, 1, labels=False)
    query.var_names = [f"query_g{i}" for i in range(50)]
    res = ScArchesMethod().run(query, _cfg(tmp_path / "atlas.h5ad"), context=_Ctx(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "no shared genes" in res.reason.lower()


def test_scarches_skips_on_too_few_atlas_cells(tmp_path: Path) -> None:
    """ScArchesMethod should MethodSkip if atlas has < knn_k cells."""
    atlas = _synth(3, 0)
    atlas.write_h5ad(tmp_path / "atlas.h5ad")
    query = _synth(60, 1, labels=False)
    res = ScArchesMethod().run(
        query, _cfg(tmp_path / "atlas.h5ad", knn_k=5), context=_Ctx(tmp_path)
    )
    assert isinstance(res, MethodSkip)
    assert "filtered atlas" in res.reason.lower()
    assert "knn_k" in res.reason.lower()


def test_seed_checkpoint_roundtrip_and_stale_rejection(tmp_path: Path) -> None:
    """Per-seed checkpoints should roundtrip and reject incompatible metadata."""

    meta = {
        "version": 1,
        "atlas_h5ad": "/atlas.h5ad",
        "label_key": "cell_type",
        "counts_layer": "counts",
        "key_added": "ref_state",
        "n_query_cells": 3,
        "n_ref_cells": 4,
        "n_hvg": 5,
        "n_shared": 6,
        "knn_k": 2,
        "n_latent": 2,
        "query_obs_digest": "query",
        "atlas_obs_digest": "atlas",
        "hvg_digest": "hvg",
    }
    prediction = {
        "hard": np.array(["A", "B", "A"]),
        "soft": pd.DataFrame({"A": [0.8, 0.2, 0.7], "B": [0.2, 0.8, 0.3]}),
        "knn_entropy": np.array([0.1, 0.2, 0.3]),
        "knn_agreement": np.array([1.0, 0.5, 1.0]),
    }
    latent = {
        "query": np.ones((3, 2)),
        "ref": np.ones((4, 2)),
    }
    loss_history = {"scvi": {"train_loss": [1.0]}, "scanvi": {}, "query_surgery": {}}

    _save_seed_checkpoint(
        objects_path=tmp_path,
        key_added="ref_state",
        seed=7,
        meta=meta,
        prediction=prediction,
        latent=latent,
        loss_history=loss_history,
    )

    loaded = _load_seed_checkpoint(
        objects_path=tmp_path,
        key_added="ref_state",
        seed=7,
        expected_meta=meta,
    )
    assert loaded is not None
    np.testing.assert_array_equal(loaded["prediction"]["hard"], prediction["hard"])
    np.testing.assert_allclose(
        loaded["prediction"]["soft"].to_numpy(),
        prediction["soft"].to_numpy(),
    )
    np.testing.assert_allclose(loaded["latent"]["query"], latent["query"])
    assert loaded["loss_history"] == loss_history

    stale_meta = {**meta, "n_query_cells": 4}
    assert (
        _load_seed_checkpoint(
            objects_path=tmp_path,
            key_added="ref_state",
            seed=7,
            expected_meta=stale_meta,
        )
        is None
    )
