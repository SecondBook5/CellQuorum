"""Tests for the in-env pyscenic_grn.py graceful-skip contract.

CI has no pyscenic_env / cisTarget DBs, so these exercise the pure-Python skip
paths only: the script must write the expected empty schemas + a SKIPPED marker
and exit 0 when its inputs are unconfigured, never crash the caller.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "src/cellquorum/backends/pyscenic_scripts/pyscenic_grn.py"
)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_skips_when_rankings_missing(tmp_path: Path) -> None:
    # An h5ad that does not exist is fine: the DB gate fires before it is read,
    # OR the loom/HDF5 import gate fires first. Either way -> SKIPPED + exit 0.
    out = tmp_path / "out"
    res = _run(
        [
            "--h5ad",
            str(tmp_path / "nope.h5ad"),
            "--tfs",
            "",
            "--motifs",
            "",
            "--rankings",
            "",
            "--out-dir",
            str(out),
            "--tag",
            "T",
        ]
    )
    assert res.returncode == 0, res.stderr
    assert (out / "grn_SKIPPED_T.txt").exists()
    assert (out / "scenic_adjacencies_T.tsv").read_text().startswith("TF\ttarget\timportance")
    assert (out / "scenic_regulons_T.csv").read_text().startswith("TF,MotifID,AUC,NES,TargetGenes")


def test_no_epithelial_subset_symbol_in_source() -> None:
    # Guard the generalization: the study-specific epithelial subset must be gone.
    text = SCRIPT.read_text()
    assert "is_epithelial" not in text
    assert "epithelial" not in text.lower()
