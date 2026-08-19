"""Tests for deterministic stage input fingerprints."""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.core.fingerprint import compute_input_fingerprint


def _adata() -> ad.AnnData:
    """Build a small deterministic AnnData for fingerprint tests."""

    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.normal(size=(6, 4)))
    a.var_names = [f"gene_{i}" for i in range(4)]
    a.layers["counts"] = a.X.copy()
    return a


def test_fingerprint_is_deterministic() -> None:
    """Identical inputs must yield an identical fingerprint."""

    a = _adata()
    fp1 = compute_input_fingerprint(
        stage_name="qc", stage_config={"mode": "flag_no_drop"}, adata=a, random_seed=1337
    )
    fp2 = compute_input_fingerprint(
        stage_name="qc", stage_config={"mode": "flag_no_drop"}, adata=a, random_seed=1337
    )
    assert fp1 == fp2
    assert isinstance(fp1, str) and len(fp1) == 64


def test_fingerprint_changes_with_config() -> None:
    """A config change must flip the fingerprint."""

    a = _adata()
    fp_report = compute_input_fingerprint(
        stage_name="qc", stage_config={"mode": "flag_no_drop"}, adata=a, random_seed=1337
    )
    fp_filter = compute_input_fingerprint(
        stage_name="qc", stage_config={"mode": "filter"}, adata=a, random_seed=1337
    )
    assert fp_report != fp_filter


def test_fingerprint_changes_with_seed_and_shape() -> None:
    """Seed and adata shape/var-space changes must flip the fingerprint."""

    a = _adata()
    base = compute_input_fingerprint(stage_name="qc", stage_config={}, adata=a, random_seed=1337)

    # Different seed.
    seeded = compute_input_fingerprint(stage_name="qc", stage_config={}, adata=a, random_seed=7)
    assert base != seeded

    # Different gene space (var names) flips the fingerprint.
    b = a[:, :3].copy()
    trimmed = compute_input_fingerprint(stage_name="qc", stage_config={}, adata=b, random_seed=1337)
    assert base != trimmed


def test_fingerprint_handles_missing_adata() -> None:
    """Fingerprinting before ingestion (no adata) must still produce a string."""

    fp = compute_input_fingerprint(stage_name="qc", stage_config={}, adata=None, random_seed=1337)
    assert isinstance(fp, str) and len(fp) == 64
