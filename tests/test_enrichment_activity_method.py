# tests/test_enrichment_activity_method.py
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.comparative.enrichment.activity_method import ActivityMethod
from cellquorum.methods.base import MethodSkip

# Exercises the REAL dc.mt.ulm, which drops all-zero observations (empty=True).
# Only the network fetch is stubbed via get_net so the drop path (C2) runs for
# real — a constant stub that never shortens the frame would not guard it.
dc = pytest.importorskip("decoupler")


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)


def _adata_with_zero_cell():
    """20 cells x 8 genes, two cell types; one cell is all-zero over the net's
    targets so real decoupler will drop it from the returned frame."""
    rng = np.random.default_rng(0)
    n = 20
    X = rng.random(size=(n, 8)) + 0.1  # strictly positive so only forced zeros drop
    X[5, :] = 0.0  # this cell (a T0) is all-zero → decoupler drops it
    obs = pd.DataFrame({"cell_type": (["T0"] * 10) + (["T1"] * 10)})
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = [f"G{i}" for i in range(8)]
    a.layers["cellquorum_normalized"] = X.copy()
    return a


def _weighted_net():
    return pd.DataFrame(
        {
            "source": ["TF0"] * 4 + ["TF1"] * 4,
            "target": [f"G{i}" for i in range(4)] + [f"G{i}" for i in range(4, 8)],
            "weight": [1.0] * 8,
        }
    )


def _patch_get_net(monkeypatch, net):
    monkeypatch.setattr(
        "cellquorum.stages.comparative.enrichment.activity_method.get_net",
        lambda collection, **kw: net.copy(),
    )


def test_activity_survives_dropped_empty_cell(tmp_path, monkeypatch):
    """C2 guard: decoupler drops the all-zero cell → the returned es frame is
    shorter than the original obs. The old code did `es[col] = full_labels`,
    raising ValueError (length mismatch) OUTSIDE the guard, aborting the stage.
    With the fix, labels are realigned to es.index and the method returns a valid
    per-cell-type result without raising.
    """
    a = _adata_with_zero_cell()
    _patch_get_net(monkeypatch, _weighted_net())
    cfg = {
        "activity_resources": ["collectri"],
        "cell_type_col": "cell_type",
        "layer": "cellquorum_normalized",
        "min_size": 3,
    }
    out = ActivityMethod()._run(a, cfg, _Ctx(tmp_path))
    assert not isinstance(out, MethodSkip)
    df = pd.read_csv(tmp_path / "results" / "enrichment_activity_collectri.csv")
    assert list(df.columns) == ["cell_type", "source", "mean_score"]
    # Both cell types still represented despite the dropped cell.
    assert set(df["cell_type"]) == {"T0", "T1"}
    assert df["mean_score"].notna().all()


def test_activity_skips_when_decoupler_absent(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "decoupler", None)
    cfg = {
        "activity_resources": ["collectri"],
        "cell_type_col": "cell_type",
        "layer": "cellquorum_normalized",
    }
    out = ActivityMethod()._run(_adata_with_zero_cell(), cfg, _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)
