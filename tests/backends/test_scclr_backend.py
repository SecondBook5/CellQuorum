"""Tests for the scclr subprocess backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from cellquorum.backends.scclr_backend import PFLOG_HELPER, ScclrBackend, build_scclr_backend


def test_backend_reports_unavailable_for_missing_env():
    """A nonexistent env name makes the backend report unavailable, not crash."""
    backend = build_scclr_backend(env_name="definitely-not-an-env-xyz")
    status = backend.status()
    # Either the launcher is missing or scclr is missing — never available here.
    assert status.available is False
    assert status.missing


def test_backend_missing_launcher_reports_unavailable():
    """An absent launcher executable makes the backend unavailable."""
    backend = ScclrBackend(launcher="no-such-launcher-xyz")
    status = backend.status()
    assert status.available is False
    assert "no-such-launcher-xyz" in status.missing


def _backend_or_skip() -> ScclrBackend:
    backend = build_scclr_backend()
    if not backend.status().available:
        pytest.skip("scclr environment unavailable (isolated micromamba env not built)")
    return backend


def test_helper_round_trips_normalize(tmp_path: Path):
    """The normalize helper round-trips counts -> sparse PFlog + row_center + meta."""
    backend = _backend_or_skip()

    rng = np.random.default_rng(0)
    counts = sp.csr_matrix(rng.negative_binomial(2, 0.15, size=(60, 25)).astype(np.float32))
    counts_path = tmp_path / "counts.npz"
    matrix_out = tmp_path / "pflog.npz"
    meta_out = tmp_path / "meta.json"
    sp.save_npz(counts_path, counts)

    result = backend.run_helper(
        PFLOG_HELPER,
        ["normalize", str(counts_path), str(matrix_out), str(meta_out), "--target", "auto"],
    )
    assert result.returncode == 0, result.stderr

    pflog = sp.load_npz(matrix_out)
    meta = json.loads(meta_out.read_text())
    assert sp.issparse(pflog)
    assert pflog.shape == (60, 25)
    assert len(meta["row_center"]) == 60
    assert meta["alpha"] is not None


def test_helper_pca_round_trips(tmp_path: Path):
    """The pca helper round-trips a sparse+center pair into PCA scores."""
    backend = _backend_or_skip()

    # First normalize to get a valid sparse+center pair.
    rng = np.random.default_rng(0)
    counts = sp.csr_matrix(rng.negative_binomial(2, 0.15, size=(60, 25)).astype(np.float32))
    counts_path = tmp_path / "counts.npz"
    pflog_path = tmp_path / "pflog.npz"
    meta_path = tmp_path / "meta.json"
    sp.save_npz(counts_path, counts)
    r1 = backend.run_helper(
        PFLOG_HELPER,
        ["normalize", str(counts_path), str(pflog_path), str(meta_path), "--target", "auto"],
    )
    assert r1.returncode == 0, r1.stderr

    center_path = tmp_path / "center.npy"
    np.save(center_path, np.asarray(json.loads(meta_path.read_text())["row_center"], dtype=float))
    pca_out = tmp_path / "pca.npz"
    pca_meta = tmp_path / "pca_meta.json"
    r2 = backend.run_helper(
        PFLOG_HELPER,
        [
            "pca",
            str(pflog_path),
            str(center_path),
            str(pca_out),
            str(pca_meta),
            "--n-components",
            "10",
        ],
    )
    assert r2.returncode == 0, r2.stderr
    with np.load(pca_out) as data:
        assert data["scores"].shape == (60, 10)
        assert data["explained_variance_ratio"].shape[0] == 10
