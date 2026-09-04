"""Tests for the scripts/smoke_real_h5ad.py entry point.

The script is invoked as a subprocess (that is the thing under test -- a CLI),
but always with `sys.executable`, i.e. the SAME interpreter running the tests.
An earlier version shelled out to `mamba run -n cellquorum-dev python`, which
tested whatever happened to be installed in a hard-coded developer environment
rather than the editable install under test -- so it passed or failed for
reasons unrelated to the code being changed, and paid mamba's solver overhead
on every one of the four cases.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_real_h5ad.py"


def make_tiny_h5ad(path: Path) -> None:
    """Create a tiny synthetic h5ad for testing."""
    matrix = np.array(
        [
            [5, 5, 0, 0, 1, 2],
            [9, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 3],
            [2, 3, 1, 0, 5, 1],
        ],
        dtype=np.float32,
    )

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(4)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(6)])

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.layers["counts"] = matrix.copy()

    adata.write_h5ad(path)


def run_smoke(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the smoke script with the current interpreter and the given flags."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def parse_summary(result: subprocess.CompletedProcess[str]) -> dict:
    """Extract the JSON summary the script prints as its last stdout object."""
    stdout = result.stdout.strip()
    json_start = stdout.rfind("{")
    assert json_start >= 0, f"No JSON output found in: {stdout}"
    return json.loads(stdout[json_start:])


def test_smoke_script_basic_execution(tmp_path: Path) -> None:
    """Test that smoke script executes successfully on synthetic data."""
    input_h5ad = tmp_path / "input.h5ad"
    make_tiny_h5ad(input_h5ad)

    result = run_smoke(
        "--input-h5ad",
        str(input_h5ad),
        "--output-dir",
        str(tmp_path / "output"),
        "--n-cells",
        "4",
        "--n-genes",
        "6",
        "--counts-layer",
        "counts",
        "--recipe",
        "cellquorum_log1p_cp10k_v1",
        "--overwrite-output",
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    summary = parse_summary(result)
    assert summary["status"] == "success"
    assert summary["subset_shape"] == [4, 6]
    assert "qc" in summary["successful_stages"]
    assert "preprocessing" in summary["successful_stages"]
    assert summary["failed_stages"] == []
    assert summary["has_normalized_layer"] is True
    assert summary["has_counts_layer"] is True
    assert summary["preprocessing_summary_exists"] is True
    assert summary["stage_records_exists"] is True


def test_smoke_script_subset_limiting(tmp_path: Path) -> None:
    """Test that smoke script respects n-cells and n-genes limits."""
    input_h5ad = tmp_path / "input.h5ad"
    make_tiny_h5ad(input_h5ad)

    result = run_smoke(
        "--input-h5ad",
        str(input_h5ad),
        "--output-dir",
        str(tmp_path / "output"),
        "--n-cells",
        "2",
        "--n-genes",
        "3",
        "--overwrite-output",
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    summary = parse_summary(result)
    assert summary["subset_shape"] == [2, 3]
    assert summary["status"] == "success"


def test_smoke_script_missing_counts_layer(tmp_path: Path) -> None:
    """Test that smoke script fails gracefully on missing counts layer."""
    # No counts layer on this object, so --counts-layer names something absent.
    adata = ad.AnnData(X=np.array([[1, 2], [3, 4]], dtype=np.float32))
    input_h5ad = tmp_path / "input.h5ad"
    adata.write_h5ad(input_h5ad)

    result = run_smoke(
        "--input-h5ad",
        str(input_h5ad),
        "--output-dir",
        str(tmp_path / "output"),
        "--counts-layer",
        "counts",
        "--overwrite-output",
    )

    assert result.returncode != 0
    assert "not found" in result.stderr or "not found" in result.stdout


def test_smoke_script_without_counts_layer_flag(tmp_path: Path) -> None:
    """Test that smoke script works without specifying counts-layer."""
    input_h5ad = tmp_path / "input.h5ad"
    make_tiny_h5ad(input_h5ad)

    result = run_smoke(
        "--input-h5ad",
        str(input_h5ad),
        "--output-dir",
        str(tmp_path / "output"),
        "--n-cells",
        "4",
        "--n-genes",
        "6",
        "--overwrite-output",
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert parse_summary(result)["status"] == "success"
