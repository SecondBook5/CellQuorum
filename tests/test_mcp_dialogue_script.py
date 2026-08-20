"""Direct smoke test of the bundled dialogue.R against fixture files.

DIALOGUE cannot converge on pure noise: DIALOGUE1's ANOVA filter keeps only
X features (PCs) that vary across samples (>=5 must pass), and the sparse-CCA /
gene-correlation steps only recover programs when the per-sample structure is
shared across cell types. So this fixture is not merely "large" random data --
it injects a shared per-sample latent signal into both the PCs and a block of
genes. Minimum that converges to MCP1 here: 8 samples, ~30 cells/sample,
8 PCs (all sample-structured so >=5 clear the ANOVA filter), 100 genes with two
20-gene latent-loaded blocks. Smaller/unstructured fixtures return "No programs".
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp

_DIALOGUE_R = Path("src/cellquorum/backends/r_scripts/dialogue.R").resolve()

_N_SAMPLES = 8
_N_PCS = 8


def _dialogue_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    p = subprocess.run(
        [
            "Rscript",
            "-e",
            'quit(status = ifelse(requireNamespace("DIALOGUE", quietly=TRUE), 0, 1))',
        ],
    )
    return p.returncode == 0


def _write_celltype(
    root: Path,
    label: str,
    n_cells: int,
    n_genes: int,
    latent: np.ndarray,
    seed: int,
) -> str:
    """Write one cell type's fixture files, driven by a *shared* per-sample latent.

    ``latent`` is (n_samples, n_pcs), identical across cell types, so the top
    latent factors show up as cross-cell-type correlated PCs (what CCA recovers)
    and as correlated gene blocks (what becomes the MCP gene program).
    """
    rng = np.random.default_rng(seed)
    stripped = label.replace("_", "")
    d = root / stripped
    d.mkdir(parents=True, exist_ok=True)
    cells = [f"{stripped}.c{i}" for i in range(n_cells)]
    genes = [f"G{i}" for i in range(n_genes)]
    samp_idx = np.array([i % _N_SAMPLES for i in range(n_cells)])

    # PCs: strong shared per-sample latent + noise -> pass ANOVA + shared signal.
    pcs = rng.normal(size=(n_cells, _N_PCS)) + 3.0 * latent[samp_idx, :]

    # Expression: baseline counts + two latent-loaded gene blocks (factors 0, 1).
    base = rng.poisson(3, size=(n_genes, n_cells)).astype(float)
    load1 = np.zeros(n_genes)
    load1[:20] = 3.0
    load2 = np.zeros(n_genes)
    load2[20:40] = 3.0
    signal = np.outer(load1, latent[samp_idx, 0]) + np.outer(load2, latent[samp_idx, 1])
    expr = base + np.maximum(np.round(signal), 0.0)  # genes x cells

    scipy.io.mmwrite(d / "expr.mtx", sp.csr_matrix(expr))
    (d / "genes.txt").write_text("\n".join(genes) + "\n")
    (d / "cells.txt").write_text("\n".join(cells) + "\n")

    pc_cols = [f"PC{i}" for i in range(_N_PCS)]
    pd.DataFrame(pcs, columns=pc_cols).assign(cell=cells)[["cell", *pc_cols]].to_csv(
        d / "X.csv", index=False
    )
    samples = [f"s{s}" for s in samp_idx]
    pd.DataFrame(
        {
            "cell": cells,
            "sample": samples,
            "cellQ": rng.normal(size=n_cells),
            "pheno": [("A" if int(s[1:]) % 2 == 0 else "B") for s in samples],
        }
    ).to_csv(d / "meta.csv", index=False)
    return stripped


@pytest.mark.skipif(not _dialogue_available(), reason="Rscript+DIALOGUE not available")
def test_dialogue_script_smoke(tmp_path):
    scratch = tmp_path / "scratch"
    out = tmp_path / "out"
    scratch.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # Shared latent (same for every cell type) so programs are multicellular.
    latent = np.random.default_rng(0).normal(size=(_N_SAMPLES, _N_PCS))

    ct_map = {}
    for label, seed in (("Type_A", 1), ("TypeB", 2)):
        stripped = _write_celltype(
            scratch, label, n_cells=240, n_genes=100, latent=latent, seed=seed
        )
        ct_map[stripped] = {"label": label, "dir": stripped}
    (scratch / "celltypes.json").write_text(json.dumps(ct_map))

    proc = subprocess.run(
        ["Rscript", str(_DIALOGUE_R), str(scratch), str(out), "2", "30", "1", "pheno", "5"],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    prog = pd.read_csv(out / "mcp_gene_programs.csv")
    assert list(prog.columns) == ["program", "cell_type", "gene", "loading", "direction"]
    # Original labels (with underscore) are preserved via the stripped->original map.
    assert set(prog["cell_type"]) <= {"Type_A", "TypeB"}
    assert not prog.empty, "fixture should converge to at least MCP1"
    assert set(prog["direction"]) <= {"up", "down"}
    assert "Type_A" in set(prog["cell_type"])  # underscore label round-trips

    scores = pd.read_csv(out / "mcp_scores.csv")
    assert list(scores.columns) == ["cell_id", "sample", "cell_type", "program", "score"]
    assert set(scores["cell_type"]) <= {"Type_A", "TypeB"}

    assoc = pd.read_csv(out / "mcp_associations.csv")
    assert list(assoc.columns) == ["program", "statistic", "pvalue", "padj", "direction"]

    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["k"] == 2
    assert meta["seed"] == 1
    assert set(meta["cell_counts"]) == {"TypeA", "TypeB"}
