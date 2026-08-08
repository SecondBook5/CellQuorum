# tests/test_da_milo_script.py
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MILO_R = Path("src/cellquorum/backends/r_scripts/milo.R")


def _milor_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status=ifelse(requireNamespace('miloR', quietly=TRUE),0,1))",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


@pytest.mark.skipif(not _milor_available(), reason="Rscript+miloR not available")
def test_milo_script_runs_and_writes_da_table(tmp_path):
    # Two well-separated 2D Gaussian blobs, 3 ctrl + 3 case donors.
    # Blob-B enriched in case (ctrl 40:20, case 20:40), ~300+ cells, k=15, prop=0.2.
    # Expected: min SpatialFDR ~0.078 per verified recipe.
    rng = np.random.default_rng(42)

    # Blob A: centered at (-5, -5) with std 0.5
    # Blob B: centered at (5, 5) with std 0.5
    # Pad to 5 dims with tiny noise for realism
    blob_a_center = np.array([-5.0, -5.0])
    blob_b_center = np.array([5.0, 5.0])
    std = 0.5

    cells_meta = []
    cells_emb = []
    cell_id_counter = 0

    # Control donors: 40 from blob A, 20 from blob B per donor
    for donor_id in ["ctrl1", "ctrl2", "ctrl3"]:
        for blob_label, n_cells, center in [
            ("TypeA", 40, blob_a_center),
            ("TypeB", 20, blob_b_center),
        ]:
            for _ in range(n_cells):
                xy = rng.normal(center, std)
                # Pad to 5 dims with small noise
                emb = np.concatenate([xy, rng.normal(0, 0.01, 3)])
                cells_emb.append(emb)
                cells_meta.append(
                    {
                        "cell": f"cell_{cell_id_counter}",
                        "donor": donor_id,
                        "condition": "Control",
                        "cell_type": blob_label,
                    }
                )
                cell_id_counter += 1

    # Case donors: 20 from blob A, 40 from blob B per donor
    for donor_id in ["case1", "case2", "case3"]:
        for blob_label, n_cells, center in [
            ("TypeA", 20, blob_a_center),
            ("TypeB", 40, blob_b_center),
        ]:
            for _ in range(n_cells):
                xy = rng.normal(center, std)
                emb = np.concatenate([xy, rng.normal(0, 0.01, 3)])
                cells_emb.append(emb)
                cells_meta.append(
                    {
                        "cell": f"cell_{cell_id_counter}",
                        "donor": donor_id,
                        "condition": "Case",
                        "cell_type": blob_label,
                    }
                )
                cell_id_counter += 1

    # Build DataFrames with cell as the first column
    emb_df = pd.DataFrame(cells_emb, columns=[f"PC{i+1}" for i in range(5)])
    emb_df.insert(0, "cell", [m["cell"] for m in cells_meta])
    meta_df = pd.DataFrame(cells_meta).set_index("cell")

    rep_csv = tmp_path / "rep.csv"
    meta_csv = tmp_path / "meta.csv"
    out_csv = tmp_path / "da.csv"
    emb_df.to_csv(rep_csv, index=False)
    meta_df.to_csv(meta_csv)

    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            str(_MILO_R),
            str(rep_csv),
            str(meta_csv),
            str(out_csv),
            "condition",
            "Case",
            "Control",
            "donor",
            "15",  # k
            "0.2",  # prop
            "cell_type",  # celltype column
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    da = pd.read_csv(out_csv)

    # Assert contract columns
    expected_cols = [
        "nhood",
        "logFC",
        "PValue",
        "SpatialFDR",
        "nhood_size",
        "majority_celltype",
        "celltype_fraction",
    ]
    assert set(expected_cols).issubset(
        da.columns
    ), f"Missing columns: {set(expected_cols) - set(da.columns)}"

    # Assert at least one neighborhood with SpatialFDR < 0.2 (verified: min ~0.078)
    assert (
        da["SpatialFDR"] < 0.2
    ).any(), f"No significant nhoods; min SpatialFDR = {da['SpatialFDR'].min()}"

    # Assert majority_celltype and celltype_fraction are populated (not all NA)
    assert da["majority_celltype"].notna().any()
    assert da["celltype_fraction"].notna().any()


@pytest.mark.skipif(not _milor_available(), reason="Rscript+miloR not available")
def test_milo_script_handles_missing_celltype(tmp_path):
    # Same fixture as above but without the celltype argument.
    # Should emit NA for majority_celltype and celltype_fraction.
    rng = np.random.default_rng(42)

    blob_a_center = np.array([-5.0, -5.0])
    blob_b_center = np.array([5.0, 5.0])
    std = 0.5

    cells_meta = []
    cells_emb = []
    cell_id_counter = 0

    for donor_id in ["ctrl1", "ctrl2", "ctrl3"]:
        for n_cells, center in [(40, blob_a_center), (20, blob_b_center)]:
            for _ in range(n_cells):
                xy = rng.normal(center, std)
                emb = np.concatenate([xy, rng.normal(0, 0.01, 3)])
                cells_emb.append(emb)
                cells_meta.append(
                    {
                        "cell": f"cell_{cell_id_counter}",
                        "donor": donor_id,
                        "condition": "Control",
                    }
                )
                cell_id_counter += 1

    for donor_id in ["case1", "case2", "case3"]:
        for n_cells, center in [(20, blob_a_center), (40, blob_b_center)]:
            for _ in range(n_cells):
                xy = rng.normal(center, std)
                emb = np.concatenate([xy, rng.normal(0, 0.01, 3)])
                cells_emb.append(emb)
                cells_meta.append(
                    {
                        "cell": f"cell_{cell_id_counter}",
                        "donor": donor_id,
                        "condition": "Case",
                    }
                )
                cell_id_counter += 1

    emb_df = pd.DataFrame(cells_emb, columns=[f"PC{i+1}" for i in range(5)])
    emb_df.insert(0, "cell", [m["cell"] for m in cells_meta])
    meta_df = pd.DataFrame(cells_meta).set_index("cell")

    rep_csv = tmp_path / "rep.csv"
    meta_csv = tmp_path / "meta.csv"
    out_csv = tmp_path / "da.csv"
    emb_df.to_csv(rep_csv, index=False)
    meta_df.to_csv(meta_csv)

    # Omit the celltype argument
    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            str(_MILO_R),
            str(rep_csv),
            str(meta_csv),
            str(out_csv),
            "condition",
            "Case",
            "Control",
            "donor",
            "15",
            "0.2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    da = pd.read_csv(out_csv)

    # Assert contract columns present
    expected_cols = [
        "nhood",
        "logFC",
        "PValue",
        "SpatialFDR",
        "nhood_size",
        "majority_celltype",
        "celltype_fraction",
    ]
    assert set(expected_cols).issubset(da.columns)

    # Assert annotation columns are all NA when celltype argument is missing
    assert da["majority_celltype"].isna().all()
    assert da["celltype_fraction"].isna().all()

    # Assert at least one significant nhood
    assert (da["SpatialFDR"] < 0.2).any()
