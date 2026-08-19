"""SoupX resume: reuse already-corrected libraries instead of re-running R.

Regression/feature: a re-run after a later-stage failure should not redo the
multi-minute-per-library SoupX correction when a complete corrected output
(+ rho sidecar) already exists.
"""

from __future__ import annotations

import gzip

from cellquorum.qc.ambient.soupx import (
    corrected_output_exists,
    read_rho_sidecar,
    write_rho_sidecar,
)


def _write_corrected_dir(d, *, with_rho=True):
    d.mkdir(parents=True, exist_ok=True)
    # Minimal 10x-style trio; contents don't matter for existence checks.
    for name in ("matrix.mtx.gz", "features.tsv.gz", "barcodes.tsv.gz"):
        with gzip.open(d / name, "wt") as fh:
            fh.write("x\n")
    if with_rho:
        write_rho_sidecar(d, 0.037)


def test_corrected_output_exists_true_when_complete(tmp_path):
    d = tmp_path / "P1"
    _write_corrected_dir(d)
    assert corrected_output_exists(d) is True


def test_corrected_output_exists_false_when_missing_file(tmp_path):
    d = tmp_path / "P2"
    _write_corrected_dir(d)
    (d / "barcodes.tsv.gz").unlink()
    assert corrected_output_exists(d) is False


def test_rho_sidecar_roundtrip(tmp_path):
    d = tmp_path / "P3"
    d.mkdir()
    write_rho_sidecar(d, 0.042)
    assert read_rho_sidecar(d) == 0.042


def test_read_rho_sidecar_absent_returns_none(tmp_path):
    d = tmp_path / "P4"
    d.mkdir()
    assert read_rho_sidecar(d) is None


def test_read_rho_sidecar_garbage_returns_none(tmp_path):
    d = tmp_path / "P5"
    d.mkdir()
    (d / "rho.txt").write_text("not-a-number\n")
    assert read_rho_sidecar(d) is None
