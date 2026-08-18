"""Tests for doublet detection + consensus."""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.qc import doublets as dbl
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


def test_score_threshold_flags_at_ceiling(monkeypatch):
    """A score AT the default 0.5 threshold must flag (regression for `> 0.5`).

    The historical bug used ``scores > 0.5`` while observed scores ceiling at
    exactly 0.5, so no cell was ever flagged. The fix uses ``>=``.
    """

    a = _counts_adata(n=5)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    # No native call → the score-threshold fallback path is exercised.
    monkeypatch.setattr(
        dbl, "run_scrublet", lambda adata, *, expected_rate, random_state: (scores, None)
    )

    cfg = QCDoubletConfig(enabled=True, methods=["scrublet"], consensus="any", per_sample=False)
    metrics = detect_doublets(a, cfg, backend=None)

    assert a.obs["predicted_doublet"].to_numpy().tolist() == [False, False, False, False, True]
    assert metrics["used_native_calls"] == {"scrublet": False}
    assert int(metrics["n_predicted_doublets"]) == 1


def test_native_calls_take_precedence(monkeypatch):
    """The detector's own call is used, not a re-threshold of the score."""

    a = _counts_adata(n=4)
    # Every cell scores "high", but the detector's native call flags only 0 and 2.
    scores = np.array([0.9, 0.9, 0.1, 0.1], dtype=float)
    native = np.array([True, False, True, False])
    monkeypatch.setattr(
        dbl, "run_scrublet", lambda adata, *, expected_rate, random_state: (scores, native)
    )

    cfg = QCDoubletConfig(enabled=True, methods=["scrublet"], consensus="any", per_sample=False)
    metrics = detect_doublets(a, cfg, backend=None)

    assert a.obs["predicted_doublet"].to_numpy().tolist() == [True, False, True, False]
    assert metrics["used_native_calls"] == {"scrublet": True}


def test_zero_flagged_warns_loudly(monkeypatch, caplog):
    """A detector that scored cells but flagged none must warn (no-silent-decisions)."""

    a = _counts_adata(n=4)
    # All below the 0.5 threshold, no native call → zero flagged.
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    monkeypatch.setattr(
        dbl, "run_scrublet", lambda adata, *, expected_rate, random_state: (scores, None)
    )

    cfg = QCDoubletConfig(enabled=True, methods=["scrublet"], consensus="any", per_sample=False)
    with caplog.at_level("WARNING"):
        metrics = detect_doublets(a, cfg, backend=None)

    assert int(metrics["n_predicted_doublets"]) == 0
    assert any("flagged 0 doublets" in record.message for record in caplog.records)


def test_consensus_any_vs_all_semantics():
    # Two synthetic per-method call columns combined by rule.
    import pandas as pd

    from cellquorum.qc.doublets import combine_consensus

    calls = pd.DataFrame({"m1": [True, True, False], "m2": [True, False, False]})
    assert list(combine_consensus(calls, "any")) == [True, True, False]
    assert list(combine_consensus(calls, "all")) == [True, False, False]
    assert list(combine_consensus(calls, "majority")) == [True, False, False]
