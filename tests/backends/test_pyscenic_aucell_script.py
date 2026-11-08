"""Tests for the in-env pyscenic_aucell.py graceful-skip contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from repo_paths import REPO_ROOT

SCRIPT = REPO_ROOT / "src/cellquorum/backends/pyscenic_scripts/pyscenic_aucell.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_skips_when_loom_missing(tmp_path: Path) -> None:
    out = tmp_path / "auc.parquet"
    res = _run(
        [
            "--loom",
            str(tmp_path / "nope.loom"),
            "--regulons",
            str(tmp_path / "nope.csv"),
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0, res.stderr
    assert (out.parent / f"{out.stem}_SKIPPED.txt").exists()


def test_skips_when_regulons_empty(tmp_path: Path) -> None:
    loom = tmp_path / "in.loom"
    loom.write_bytes(b"not-really-a-loom-but-nonempty")
    regs = tmp_path / "regs.csv"
    regs.write_text("")  # empty -> skip
    out = tmp_path / "auc.parquet"
    res = _run(["--loom", str(loom), "--regulons", str(regs), "--out", str(out)])
    assert res.returncode == 0, res.stderr
    assert (out.parent / f"{out.stem}_SKIPPED.txt").exists()


def test_script_takes_a_seed_for_the_ranking_step() -> None:
    """Without a seed, ``pyscenic aucell`` picks a random one and the AUC matrix changes per run.

    Measured on real data: two identical runs gave per-regulon rank correlations down to 0.46
    between their AUC matrices, though every earlier step was seeded and identical.
    """
    res = _run(["--help"])
    assert res.returncode == 0, res.stderr
    assert "--seed" in res.stdout
