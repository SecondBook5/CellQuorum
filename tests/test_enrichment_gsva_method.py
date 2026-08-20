# tests/test_enrichment_gsva_method.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.enrichment.gsva_method import GsvaMethod
from cellquorum.methods.base import MethodSkip

# Exercises the REAL dc.mt.gsva. One pseudobulk sample has zero library size, so
# without the fix it would be CPM-normalized into an all-zero row that decoupler
# drops (empty=True) — desynchronizing the condition mask (IndexError). Only the
# network fetch is stubbed via get_net.
dc = pytest.importorskip("decoupler")


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)


def _adata(zero_library_donor: str | None = None):
    """6 donors (3 Normal, 3 Disease), 12 genes. If `zero_library_donor` is set,
    every cell of that donor is all-zero, so its pseudobulk sample has zero
    library size and (pre-fix) would collapse to an all-zero CPM row."""
    rng = np.random.default_rng(0)
    donors = ["d1", "d2", "d3", "d4", "d5", "d6"]
    rows, blocks = [], []
    for i, d in enumerate(donors):
        cond = "Normal" if i < 3 else "Disease"
        for _ in range(20):
            if d == zero_library_donor:
                blocks.append(np.zeros(12, dtype=float))
            else:
                blocks.append(rng.poisson(5, size=12).astype(float) + 1.0)
            rows.append({"patient_id": d, "condition": cond, "cell_type": "T0"})
    X = np.vstack(blocks)
    a = ad.AnnData(X=X, obs=pd.DataFrame(rows))
    a.var_names = [f"G{i}" for i in range(12)]
    a.layers["counts"] = X.copy()
    return a


def _net():
    return pd.DataFrame(
        {
            "source": ["P0"] * 6 + ["P1"] * 6,
            "target": [f"G{i}" for i in range(6)] + [f"G{i}" for i in range(6, 12)],
        }
    )


def _patch_get_net(monkeypatch, net):
    monkeypatch.setattr(
        "cellquorum.enrichment.gsva_method.get_net",
        lambda collection, **kw: net.copy(),
    )


def _cfg():
    return {
        "gene_set_collections": ["hallmark"],
        "condition_col": "condition",
        "donor_col": "patient_id",
        "case": "Disease",
        "control": "Normal",
        "paired": False,
        "min_size": 3,
        "counts_layer": "counts",
    }


def test_gsva_skips_when_no_case_control(tmp_path):
    out = GsvaMethod()._run(_adata(), {"gene_set_collections": ["hallmark"]}, _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)


def test_gsva_runs_and_writes_csv(tmp_path, monkeypatch):
    _patch_get_net(monkeypatch, _net())
    out = GsvaMethod()._run(_adata(), _cfg(), _Ctx(tmp_path))
    assert not isinstance(out, MethodSkip)
    scores = tmp_path / "results" / "enrichment_gsva_scores_hallmark.csv"
    contrast = tmp_path / "results" / "enrichment_gsva_contrast_hallmark.csv"
    assert scores.exists() and contrast.exists()
    cdf = pd.read_csv(contrast)
    assert list(cdf.columns) == [
        "source",
        "case_mean",
        "control_mean",
        "statistic",
        "pvalue",
        "padj",
        "significant",
        "collection",
    ]


def test_gsva_survives_dropped_zero_library_sample(tmp_path, monkeypatch):
    """C3 guard: donor d4 (Disease) has zero library size. Pre-fix it became an
    all-zero CPM row that decoupler drops, so the positional condition mask no
    longer matched es → IndexError, aborting the stage. With the fix the zero-
    library sample is removed up front and the condition vector is aligned to the
    surviving es.index, so the t-test runs on the survivors and never raises.
    d5 and d6 keep the Disease arm at 2 samples, so a contrast is still produced.
    """
    _patch_get_net(monkeypatch, _net())
    out = GsvaMethod()._run(_adata(zero_library_donor="d4"), _cfg(), _Ctx(tmp_path))
    assert not isinstance(out, MethodSkip)
    cdf = pd.read_csv(tmp_path / "results" / "enrichment_gsva_contrast_hallmark.csv")
    # Both pathway sources present, p-values computed on the aligned survivors.
    assert set(cdf["source"]) == {"P0", "P1"}
    assert cdf["pvalue"].notna().all()


def test_gsva_skips_when_arm_emptied_by_drop(tmp_path, monkeypatch):
    """C3 guard (skip branch): drop two of three Disease donors' libraries so the
    case arm falls below 2 surviving samples → recorded MethodSkip, never a
    crash and never a stage abort."""
    _patch_get_net(monkeypatch, _net())
    a = _adata(zero_library_donor="d4")
    # Also zero-out d5 so only d6 remains in the Disease arm.
    d5_mask = (a.obs["patient_id"] == "d5").to_numpy()
    a.layers["counts"][d5_mask, :] = 0.0
    out = GsvaMethod()._run(a, _cfg(), _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)
