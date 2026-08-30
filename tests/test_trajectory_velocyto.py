# tests/test_trajectory_velocyto.py
from __future__ import annotations

import pytest

from cellquorum.stages.trajectory import _velocyto
from cellquorum.stages.trajectory.config import VelocityGenerationConfig


def test_ensure_loom_skips_when_disabled(tmp_path):
    gen = VelocityGenerationConfig(generate_missing=False)
    loom, reason = _velocyto.ensure_loom("s1", tmp_path, gen)
    assert loom is None
    assert "disabled" in reason.lower()


def test_ensure_loom_idempotent_returns_existing(tmp_path, monkeypatch):
    # An existing loom under velocyto/ must be returned WITHOUT running anything.
    velo_dir = tmp_path / "velocyto"
    velo_dir.mkdir()
    existing = velo_dir / "s1.loom"
    existing.write_bytes(b"loom")
    gen = VelocityGenerationConfig(generate_missing=True, gtf_path=tmp_path / "genes.gtf")
    (tmp_path / "genes.gtf").write_text("gtf")

    called = {"run": False}
    monkeypatch.setattr(_velocyto, "_run", lambda *a, **k: called.__setitem__("run", True))
    monkeypatch.setattr(_velocyto, "_binary_available", lambda name: True)

    loom, reason = _velocyto.ensure_loom("s1", tmp_path, gen)
    assert loom == existing
    assert called["run"] is False
    assert "exists" in reason.lower()


def test_ensure_loom_skips_when_binary_missing(tmp_path):
    gen = VelocityGenerationConfig(generate_missing=True, gtf_path=tmp_path / "g.gtf")
    (tmp_path / "g.gtf").write_text("gtf")
    # No monkeypatch → real which() almost certainly lacks velocyto in CI.
    import shutil

    if shutil.which("velocyto") and shutil.which("samtools"):
        pytest.skip("velocyto/samtools actually installed")
    loom, reason = _velocyto.ensure_loom("s1", tmp_path, gen)
    assert loom is None
    assert "velocyto" in reason.lower() or "samtools" in reason.lower()


def test_ensure_loom_skips_when_gtf_missing(tmp_path, monkeypatch):
    gen = VelocityGenerationConfig(generate_missing=True, gtf_path=tmp_path / "absent.gtf")
    monkeypatch.setattr(_velocyto, "_binary_available", lambda name: True)
    loom, reason = _velocyto.ensure_loom("s1", tmp_path, gen)
    assert loom is None
    assert "gtf" in reason.lower()


def test_ensure_loom_runs_cellsort_and_velocyto_when_all_present(tmp_path, monkeypatch):
    gtf = tmp_path / "genes.gtf"
    gtf.write_text("gtf")
    outs = tmp_path / "outs"
    outs.mkdir()
    (outs / "possorted_genome_bam.bam").write_bytes(b"bam")
    gen = VelocityGenerationConfig(generate_missing=True, gtf_path=gtf, threads=4)

    monkeypatch.setattr(_velocyto, "_binary_available", lambda name: True)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        # simulate velocyto producing the loom on its run10x call
        if cmd[0] == "velocyto":
            velo_dir = tmp_path / "velocyto"
            velo_dir.mkdir(exist_ok=True)
            (velo_dir / "s1.loom").write_bytes(b"loom")

    monkeypatch.setattr(_velocyto, "_run", fake_run)

    loom, reason = _velocyto.ensure_loom("s1", tmp_path, gen)
    assert loom is not None and loom.exists()
    # CB-sort (samtools) invoked because no cellsorted BAM existed, then velocyto.
    assert any(c[0] == "samtools" for c in calls)
    assert any(c[0] == "velocyto" for c in calls)


def test_ensure_loom_skips_cellsort_when_cellsorted_present(tmp_path, monkeypatch):
    gtf = tmp_path / "genes.gtf"
    gtf.write_text("gtf")
    outs = tmp_path / "outs"
    outs.mkdir()
    (outs / "cellsorted_possorted_genome_bam.bam").write_bytes(b"bam")
    gen = VelocityGenerationConfig(generate_missing=True, gtf_path=gtf)

    monkeypatch.setattr(_velocyto, "_binary_available", lambda name: True)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "velocyto":
            velo_dir = tmp_path / "velocyto"
            velo_dir.mkdir(exist_ok=True)
            (velo_dir / "s1.loom").write_bytes(b"loom")

    monkeypatch.setattr(_velocyto, "_run", fake_run)
    loom, reason = _velocyto.ensure_loom("s1", tmp_path, gen)
    assert loom is not None
    assert not any(c[0] == "samtools" for c in calls)  # cellsort skipped
