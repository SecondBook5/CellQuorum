"""Tests for doublet detection + consensus."""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.qc.config import QCDoubletConfig
from cellquorum.qc.doublets import detect_doublets


def _counts_adata(seed=0, n=200, g=500):
    rng = np.random.default_rng(seed)
    x = rng.poisson(1.0, size=(n, g)).astype(np.float32)
    a = ad.AnnData(X=x)
    a.layers["counts"] = x.copy()
    return a


def test_scrublet_consensus_writes_calls():
    a = _counts_adata()
    cfg = QCDoubletConfig(
        enabled=True,
        methods=["scrublet"],
        consensus="any",
        remove=False,
        expected_doublet_rate=0.06,
    )
    metrics = detect_doublets(a, cfg, backend=None)
    assert "doublet_score" in a.obs
    assert "predicted_doublet" in a.obs
    assert "scrublet" in metrics["methods_run"]
    # No cells removed here (flag-only).
    assert a.n_obs == 200


def test_per_sample_detection_runs_per_library():
    """With per_sample + a sample_key, detection runs per library, not pooled."""
    import pandas as pd

    a = _counts_adata(n=120, g=400)
    # Two libraries; per-sample detection should score each independently.
    a.obs["sample_id"] = pd.Categorical(["libA"] * 60 + ["libB"] * 60)
    cfg = QCDoubletConfig(
        enabled=True,
        methods=["scrublet"],
        consensus="any",
        per_sample=True,
        expected_doublet_rate=0.06,
    )

    metrics = detect_doublets(a, cfg, backend=None, sample_key="sample_id")

    assert metrics["scored_scope"] == "per_sample"
    assert metrics["sample_key"] == "sample_id"
    assert "doublet_score" in a.obs
    # Every cell was scored by some library-local run.
    assert a.obs["doublet_score"].notna().all()


def test_per_sample_falls_back_to_pooled_without_sample_key():
    """per_sample=True but no sample_key resolves to pooled detection."""
    a = _counts_adata(n=100, g=300)
    cfg = QCDoubletConfig(enabled=True, methods=["scrublet"], consensus="any", per_sample=True)

    metrics = detect_doublets(a, cfg, backend=None, sample_key=None)

    assert metrics["scored_scope"] == "pooled"
    assert metrics["sample_key"] is None


def test_consensus_any_vs_all_semantics():
    # Two synthetic per-method call columns combined by rule.
    import pandas as pd

    from cellquorum.qc.doublets import combine_consensus

    calls = pd.DataFrame({"m1": [True, True, False], "m2": [True, False, False]})
    assert list(combine_consensus(calls, "any")) == [True, True, False]
    assert list(combine_consensus(calls, "all")) == [True, False, False]
    assert list(combine_consensus(calls, "majority")) == [True, False, False]
