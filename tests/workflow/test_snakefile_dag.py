from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dry_run_expands_expected_targets(tmp_path: Path) -> None:
    pytest.importorskip("snakemake", minversion="8")
    fixture = Path(__file__).parent / "fixtures" / "hypotheses_fixture.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "-n",
            "--snakefile",
            str(REPO_ROOT / "workflow" / "Snakefile"),
            "--config",
            f"manifest={fixture}",
            f"workdir={tmp_path}",
            "--cores",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for target in ("il33_axis", "emt_krt", "matrix_status"):
        assert target in out
