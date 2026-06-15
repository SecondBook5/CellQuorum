"""Tests for smoke_real_h5ad.py script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


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


def test_smoke_script_basic_execution(tmp_path: Path) -> None:
    """Test that smoke script executes successfully on synthetic data."""
    # Create input h5ad.
    input_h5ad = tmp_path / "input.h5ad"
    make_tiny_h5ad(input_h5ad)

    # Create output directory.
    output_dir = tmp_path / "output"

    # Run smoke script.
    script_path = Path(__file__).parent.parent / "scripts" / "smoke_real_h5ad.py"

    result = subprocess.run(
        [
            "mamba",
            "run",
            "-n",
            "cellquorum-dev",
            "python",
            str(script_path),
            "--input-h5ad",
            str(input_h5ad),
            "--output-dir",
            str(output_dir),
            "--n-cells",
            "4",
            "--n-genes",
            "6",
            "--counts-layer",
            "counts",
            "--recipe",
            "cellquorum_log1p_cp10k_v1",
            "--overwrite-output",
        ],
        capture_output=True,
        text=True,
    )

    # Check exit code.
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Parse JSON output (find the JSON object in stdout).
    stdout = result.stdout.strip()
    # JSON output starts with { and ends with }
    json_start = stdout.rfind("{")
    assert json_start >= 0, f"No JSON output found in: {stdout}"
    json_str = stdout[json_start:]
    summary = json.loads(json_str)

    # Validate summary.
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
    # Create input h5ad.
    input_h5ad = tmp_path / "input.h5ad"
    make_tiny_h5ad(input_h5ad)

    # Create output directory.
    output_dir = tmp_path / "output"

    # Run smoke script with smaller subset.
    script_path = Path(__file__).parent.parent / "scripts" / "smoke_real_h5ad.py"

    result = subprocess.run(
        [
            "mamba",
            "run",
            "-n",
            "cellquorum-dev",
            "python",
            str(script_path),
            "--input-h5ad",
            str(input_h5ad),
            "--output-dir",
            str(output_dir),
            "--n-cells",
            "2",
            "--n-genes",
            "3",
            "--overwrite-output",
        ],
        capture_output=True,
        text=True,
    )

    # Check exit code.
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Parse JSON output (find the JSON object in stdout).
    stdout = result.stdout.strip()
    # JSON output starts with { and ends with }
    json_start = stdout.rfind("{")
    assert json_start >= 0, f"No JSON output found in: {stdout}"
    json_str = stdout[json_start:]
    summary = json.loads(json_str)

    # Validate subset shape.
    assert summary["subset_shape"] == [2, 3]
    assert summary["status"] == "success"


def test_smoke_script_missing_counts_layer(tmp_path: Path) -> None:
    """Test that smoke script fails gracefully on missing counts layer."""
    # Create input h5ad without counts layer.
    matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
    adata = ad.AnnData(X=matrix)

    input_h5ad = tmp_path / "input.h5ad"
    adata.write_h5ad(input_h5ad)

    # Create output directory.
    output_dir = tmp_path / "output"

    # Run smoke script.
    script_path = Path(__file__).parent.parent / "scripts" / "smoke_real_h5ad.py"

    result = subprocess.run(
        [
            "mamba",
            "run",
            "-n",
            "cellquorum-dev",
            "python",
            str(script_path),
            "--input-h5ad",
            str(input_h5ad),
            "--output-dir",
            str(output_dir),
            "--counts-layer",
            "counts",
            "--overwrite-output",
        ],
        capture_output=True,
        text=True,
    )

    # Check that it failed.
    assert result.returncode != 0
    assert "not found" in result.stderr or "not found" in result.stdout


def test_smoke_script_without_counts_layer_flag(tmp_path: Path) -> None:
    """Test that smoke script works without specifying counts-layer."""
    # Create input h5ad.
    input_h5ad = tmp_path / "input.h5ad"
    make_tiny_h5ad(input_h5ad)

    # Create output directory.
    output_dir = tmp_path / "output"

    # Run smoke script without --counts-layer.
    script_path = Path(__file__).parent.parent / "scripts" / "smoke_real_h5ad.py"

    result = subprocess.run(
        [
            "mamba",
            "run",
            "-n",
            "cellquorum-dev",
            "python",
            str(script_path),
            "--input-h5ad",
            str(input_h5ad),
            "--output-dir",
            str(output_dir),
            "--n-cells",
            "4",
            "--n-genes",
            "6",
            "--overwrite-output",
        ],
        capture_output=True,
        text=True,
    )

    # Check exit code.
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Parse JSON output (find the JSON object in stdout).
    stdout = result.stdout.strip()
    # JSON output starts with { and ends with }
    json_start = stdout.rfind("{")
    assert json_start >= 0, f"No JSON output found in: {stdout}"
    json_str = stdout[json_start:]
    summary = json.loads(json_str)

    # Validate.
    assert summary["status"] == "success"
