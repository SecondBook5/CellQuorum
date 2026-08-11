"""Tests for the in-env pyscenic_aucell.py graceful-skip contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/cellquorum/backends/pyscenic_scripts/pyscenic_aucell.py"
)


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
