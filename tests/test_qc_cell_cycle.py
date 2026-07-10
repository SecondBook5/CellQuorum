"""Tests for cell-cycle scoring."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.qc.cell_cycle import (
    TIROSH_G2M_GENES,
    TIROSH_S_GENES,
    score_cell_cycle,
)
from cellquorum.qc.config import QCCellCycleConfig


def _adata_with_cc_genes(seed=0):
    rng = np.random.default_rng(seed)
    # Include some real S and G2M gene symbols so scoring has signal.
    s = TIROSH_S_GENES[:5]
    g2m = TIROSH_G2M_GENES[:5]
    other = [f"G{i}" for i in range(100)]
    genes = s + g2m + other
    n = 60
    x = rng.random((n, len(genes))).astype(np.float32)
    # Make first third S-high, second third G2M-high.
    x[: n // 3, : len(s)] += 3.0
    x[n // 3 : 2 * n // 3, len(s) : len(s) + len(g2m)] += 3.0
    a = ad.AnnData(X=x, var=pd.DataFrame(index=genes))
    a.layers["cellquorum_normalized"] = x.copy()
    return a


def test_score_cell_cycle_writes_obs():
    a = _adata_with_cc_genes()
    cfg = QCCellCycleConfig(
        enabled=True, s_genes=TIROSH_S_GENES[:5], g2m_genes=TIROSH_G2M_GENES[:5]
    )
    metrics = score_cell_cycle(a, cfg)
    assert "S_score" in a.obs
    assert "G2M_score" in a.obs
    assert "phase" in a.obs
    assert metrics["n_s_genes_used"] == 5
    assert set(a.obs["phase"].unique()) <= {"S", "G2M", "G1"}


def test_tirosh_lists_nonempty():
    assert len(TIROSH_S_GENES) > 10
    assert len(TIROSH_G2M_GENES) > 10
