# tests/test_da_propeller_script.py
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

_PROPELLER_R = Path("src/cellquorum/backends/r_scripts/propeller.R")


def _speckle_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status=ifelse(requireNamespace('speckle', quietly=TRUE),0,1))",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


@pytest.mark.skipif(not _speckle_available(), reason="Rscript+speckle not available")
def test_propeller_script_runs_and_writes_da_table(tmp_path):
    # 6 pseudo-samples: 3 replicates x 2 conditions; CellTypeB is strongly shifted (~10% → ~50%).
    # CellTypeA stays constant at 500 counts (proportion adjusts but absolute stable).
    cell_types = ["CellTypeA", "CellTypeB", "CellTypeC"]
    rows, meta = [], []
    for rep in ["rep1", "rep2", "rep3"]:
        for cond in ["Control", "Case"]:
            # CellTypeA: constant at 500 (proportion ~50% → ~42%)
            # CellTypeB: shifts 100 → 600 (proportion ~10% → ~50%)
            # CellTypeC: compensates 400 → 100 (proportion ~40% → ~8%)
            if cond == "Control":
                counts = [500, 100, 400]  # Total=1000, A=50%, B=10%, C=40%
            else:
                counts = [500, 600, 100]  # Total=1200, A=42%, B=50%, C=8%
            rows.append(counts)
            meta.append({"sample": f"{rep}__{cond}", "condition": cond})
    counts_df = pd.DataFrame(rows, columns=cell_types)
    counts_df.insert(0, "sample", [m["sample"] for m in meta])
    meta_df = pd.DataFrame(meta).set_index("sample")

    counts_csv = tmp_path / "counts.csv"
    meta_csv = tmp_path / "meta.csv"
    out_csv = tmp_path / "da.csv"
    counts_df.to_csv(counts_csv, index=False)
    meta_df.to_csv(meta_csv)

    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            str(_PROPELLER_R),
            str(counts_csv),
            str(meta_csv),
            str(out_csv),
            "condition",
            "Case",
            "Control",
            "asin",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    da = pd.read_csv(out_csv)
    assert set(["cell_type", "PropRatio", "Tstatistic", "PValue", "FDR"]).issubset(da.columns)
    # CellTypeB should be significantly different (strongest signal).
    celltype_b = da.loc[da["cell_type"] == "CellTypeB"].iloc[0]
    assert celltype_b["PValue"] < 0.05
    # All three cell types change proportions in this fixture, but B has the largest shift.
    # Just verify we get valid numeric results for all.
    assert len(da) == 3
    assert da["PValue"].notna().all()
