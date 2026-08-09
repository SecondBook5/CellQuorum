# tests/test_ccc_multinichenet_script.py
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp

_SCRIPT = (
    Path(__file__).parent.parent
    / "src"
    / "cellquorum"
    / "backends"
    / "r_scripts"
    / "multinichenet.R"
)


def _rscript_pkg(pkg: str) -> bool:
    if shutil.which("Rscript") is None:
        return False
    import subprocess

    r = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            f"quit(status=ifelse(requireNamespace('{pkg}', quietly=TRUE),0,1))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.returncode == 0


def test_multinichenet_script_file_exists():
    assert _SCRIPT.is_file()


@pytest.mark.skipif(not _rscript_pkg("multinichenetr"), reason="multinichenetr unavailable")
def test_multinichenet_script_runs(tmp_path):
    # Minimal SCE inputs; if the biology is too small multinichenet may error —
    # we assert the script either writes the out CSV or exits non-zero cleanly
    # (never hangs). This is a smoke test for arg wiring, not a biology test.
    import subprocess

    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 40
    X = sp.csr_matrix(rng.poisson(1.0, size=(n_genes, n_cells)).astype(float))
    scipy.io.mmwrite(tmp_path / "counts.mtx", X.tocoo())
    pd.DataFrame({"gene": [f"G{i}" for i in range(n_genes)]}).to_csv(
        tmp_path / "genes.csv", index=False
    )
    bcs = [f"c{i}" for i in range(n_cells)]
    pd.DataFrame({"barcode": bcs}).to_csv(tmp_path / "barcodes.csv", index=False)
    pd.DataFrame(
        {
            "barcode": bcs,
            "cell_type": (["A", "B"] * 30),
            "sample_id": ([f"s{i%4}" for i in range(n_cells)]),
            "condition": (["case", "ctrl"] * 30),
        }
    ).to_csv(tmp_path / "obs.csv", index=False)
    out = tmp_path / "out.csv"
    # No prior RDS in CI -> pass nonexistent paths; script must exit non-zero, not hang.
    proc = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            str(_SCRIPT),
            str(tmp_path / "counts.mtx"),
            str(tmp_path / "genes.csv"),
            str(tmp_path / "barcodes.csv"),
            str(tmp_path / "obs.csv"),
            str(out),
            "cell_type",
            "sample_id",
            "condition",
            "case",
            "ctrl",
            str(tmp_path / "nope_lt.rds"),
            str(tmp_path / "nope_lr.rds"),
            "0.05",
            "0.5",
            "0.5",
            "0.05",
            "FALSE",
            "250",
            "regular",
            "1",
            "42",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    # Missing priors -> non-zero exit (clean failure the Python wrapper turns into MethodSkip).
    assert proc.returncode != 0
