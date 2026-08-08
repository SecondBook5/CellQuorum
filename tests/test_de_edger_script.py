# tests/test_de_edger_script.py
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_EDGER_R = Path("src/cellquorum/backends/r_scripts/edger.R")


def _edger_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status=ifelse(requireNamespace('edgeR', quietly=TRUE),0,1))",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


@pytest.mark.skipif(not _edger_available(), reason="Rscript+edgeR not available")
def test_edger_script_runs_and_writes_de_table(tmp_path):
    # 6 pseudo-samples: 3 donors x 2 conditions; GeneB is strongly up in LE.
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(20)]
    rows, meta = [], []
    for donor in ["d1", "d2", "d3"]:
        for cond in ["Normal", "LE"]:
            base = rng.poisson(50, size=20).astype(int)
            if cond == "LE":
                base[1] += 400  # GeneB up in LE
            rows.append(base)
            meta.append({"sample": f"{donor}__{cond}", "donor": donor, "condition": cond})
    counts = pd.DataFrame(rows, columns=genes)
    counts.insert(0, "sample", [m["sample"] for m in meta])
    meta_df = pd.DataFrame(meta).set_index("sample")

    counts_csv = tmp_path / "counts.csv"
    meta_csv = tmp_path / "meta.csv"
    out_csv = tmp_path / "de.csv"
    counts.to_csv(counts_csv, index=False)
    meta_df.to_csv(meta_csv)

    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            str(_EDGER_R),
            str(counts_csv),
            str(meta_csv),
            str(out_csv),
            "condition",
            "LE",
            "Normal",
            "donor + condition",
            "10",
            "15",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    de = pd.read_csv(out_csv)
    assert set(["gene", "logFC", "logCPM", "F", "PValue", "FDR"]).issubset(de.columns)
    # GeneB should be significantly up in LE.
    gene_b = de.loc[de["gene"] == "G1"].iloc[0]
    assert gene_b["logFC"] > 1
    assert gene_b["FDR"] < 0.05
    # Pin contrast direction: a null gene should be non-significant.
    gene_null = de.loc[de["gene"] == "G0"].iloc[0]
    assert gene_null["FDR"] > 0.05


@pytest.mark.skipif(not _edger_available(), reason="Rscript+edgeR not available")
def test_edger_script_handles_continuous_and_categorical_covariates(tmp_path):
    # 6 pseudo-samples: 3 donors x 2 conditions; GeneB strongly up in LE.
    # Add continuous (age) and categorical (sex) covariates.
    # Design is ~ age + sex + condition (no donor to avoid df exhaustion).
    # This test discriminates: age as numeric = 1 col (well-conditioned, 4 total design cols);
    # age factorized = 2 dummies (5 design cols, near rank-deficient with 6 samples).
    rng = np.random.default_rng(42)
    genes = [f"G{i}" for i in range(20)]
    donor_ages = {"d1": 40, "d2": 55, "d3": 70}
    donor_sexes = {"d1": "M", "d2": "F", "d3": "M"}
    rows, meta = [], []
    for donor in ["d1", "d2", "d3"]:
        for cond in ["Normal", "LE"]:
            base = rng.poisson(50, size=20).astype(int)
            if cond == "LE":
                base[1] += 400  # GeneB up in LE
            rows.append(base)
            meta.append(
                {
                    "sample": f"{donor}__{cond}",
                    "condition": cond,
                    "age": donor_ages[donor],
                    "sex": donor_sexes[donor],
                }
            )
    counts = pd.DataFrame(rows, columns=genes)
    counts.insert(0, "sample", [m["sample"] for m in meta])
    meta_df = pd.DataFrame(meta).set_index("sample")

    counts_csv = tmp_path / "counts.csv"
    meta_csv = tmp_path / "meta.csv"
    out_csv = tmp_path / "de.csv"
    counts.to_csv(counts_csv, index=False)
    meta_df.to_csv(meta_csv)

    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            str(_EDGER_R),
            str(counts_csv),
            str(meta_csv),
            str(out_csv),
            "condition",
            "LE",
            "Normal",
            "age + sex + condition",
            "10",
            "15",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    de = pd.read_csv(out_csv)
    assert set(["gene", "logFC", "logCPM", "F", "PValue", "FDR"]).issubset(de.columns)
    # GeneB should still be significantly up in LE with covariates adjusted.
    gene_b = de.loc[de["gene"] == "G1"].iloc[0]
    assert gene_b["logFC"] > 1
    assert gene_b["FDR"] < 0.05
