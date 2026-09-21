"""Tests for the in-env run_metacell2.py graceful-skip contract.

CI has no metacells_env / metacells package, so these exercise the
pure-Python skip paths only: the script must write a SKIPPED sentinel + an
empty assignments CSV and exit 0 when the metacells package is not
importable, never crash the caller.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/cellquorum/backends/metacells_scripts/run_metacell2.py"
)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_skips_when_metacells_package_not_importable(tmp_path: Path) -> None:
    out = tmp_path / "out"
    # This test's own interpreter (pytest's env) has no `metacells` package
    # installed, so the script's own import-gate fires exactly as it would
    # in CI -- no monkeypatching needed.
    res = _run(
        [
            str(tmp_path / "does_not_matter.h5ad"),
            str(out),
            "X_pca_harmony",
            "counts",
            "cell_type",
            "96",
            "5000",
            "0",
        ]
    )
    assert res.returncode == 0, res.stderr
    assert (out / "metacells_SKIPPED.txt").exists()
    assert (
        (out / "metacell_assignments.csv")
        .read_text()
        .startswith("cell_id,metacell_id,group,n_umis")
    )


def test_wrong_argument_count_exits_nonzero() -> None:
    res = _run(["only_one_arg"])
    assert res.returncode != 0


def test_no_study_specific_biology_in_source() -> None:
    text = SCRIPT.read_text()
    for forbidden in ("fibroblast", "lymphedema", "keratinocyte", "AAH", "AIS"):
        assert forbidden.lower() not in text.lower()
