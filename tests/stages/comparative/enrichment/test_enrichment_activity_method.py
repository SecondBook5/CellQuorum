# tests/stages/comparative/enrichment/test_enrichment_activity_method.py
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.comparative.enrichment.activity_method import ActivityMethod

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


def _sparse_adata_with_zero_cell():
    """Sparse variant of _adata_with_zero_cell, matching a real cohort's layer dtype."""
    import scipy.sparse as sp

    a = _adata_with_zero_cell()
    sparse = sp.csr_matrix(np.asarray(a.layers["cellquorum_normalized"]))
    a.X = sparse.copy()
    a.layers["cellquorum_normalized"] = sparse.copy()
    return a


def test_activity_does_not_densify_the_full_expression_matrix(tmp_path, monkeypatch):
    """Activity (ulm) must not materialize a dense copy of the WHOLE expression
    matrix -- the same "12 GB in a minute, VM died" failure mode already fixed
    once in state_scoring/aucell_method.py, just for decoupler's ulm instead of
    aucell. decoupler's AnnData-input path handles a sparse layer natively, so
    nothing here should ever call .toarray()/np.asarray() on the full sparse
    layer -- a real cohort's full matrix (201,871 x 33,417) densified is ~54 GB.
    """
    import scipy.sparse as sp

    a = _sparse_adata_with_zero_cell()
    _patch_get_net(monkeypatch, _weighted_net())
    cfg = {
        "activity_resources": ["collectri"],
        "cell_type_col": "cell_type",
        "layer": "cellquorum_normalized",
        "min_size": 3,
    }

    original_toarray = sp.csr_matrix.toarray

    def _guarded_toarray(self, *args, **kwargs):
        if self.shape == a.layers["cellquorum_normalized"].shape:
            raise AssertionError(
                "Activity densified the FULL expression matrix instead of letting "
                "decoupler handle the sparse layer natively."
            )
        return original_toarray(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", _guarded_toarray)

    out = ActivityMethod()._run(a, cfg, _Ctx(tmp_path))
    assert not isinstance(out, MethodSkip)
    df = pd.read_csv(tmp_path / "results" / "enrichment_activity_collectri.csv")
    assert set(df["cell_type"]) == {"T0", "T1"}
    assert df["mean_score"].notna().all()


def _adata_with_condition():
    """20 cells x 8 genes, two cell types x two conditions, no dropped cells --

    isolates the condition-splitting behavior from the empty-cell guard
    already covered above.
    """
    rng = np.random.default_rng(1)
    n = 20
    X = rng.random(size=(n, 8)) + 0.1
    obs = pd.DataFrame(
        {
            "cell_type": (["T0"] * 10) + (["T1"] * 10),
            "condition": (["Normal"] * 5 + ["Lymphedema"] * 5) * 2,
        }
    )
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = [f"G{i}" for i in range(8)]
    a.layers["cellquorum_normalized"] = X.copy()
    return a


def test_activity_splits_by_condition_when_condition_col_configured(tmp_path, monkeypatch):
    """condition_col is bridged into every enrichment method's config from

    config.design (see EnrichmentStage._augment_config) -- when present in
    obs, activity must aggregate per (cell_type, condition) instead of just
    cell_type, so the resulting figure can compare disease states rather
    than only re-confirming cluster identity.
    """
    a = _adata_with_condition()
    _patch_get_net(monkeypatch, _weighted_net())
    cfg = {
        "activity_resources": ["collectri"],
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "layer": "cellquorum_normalized",
        "min_size": 3,
    }
    out = ActivityMethod()._run(a, cfg, _Ctx(tmp_path))
    assert not isinstance(out, MethodSkip)
    df = pd.read_csv(tmp_path / "results" / "enrichment_activity_collectri.csv")
    assert list(df.columns) == ["cell_type", "condition", "source", "mean_score"]
    assert set(df["condition"]) == {"Normal", "Lymphedema"}
    # Both cell types x both conditions represented (2 x 2 x 2 sources = 8 rows).
    assert len(df) == 8
    assert df["mean_score"].notna().all()


def test_activity_omits_condition_column_when_not_configured(tmp_path, monkeypatch):
    """Backward compatible: no condition_col in config -> unchanged output,

    same as before this feature existed.
    """
    a = _adata_with_condition()
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


def test_activity_skips_when_decoupler_absent(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "decoupler", None)
    cfg = {
        "activity_resources": ["collectri"],
        "cell_type_col": "cell_type",
        "layer": "cellquorum_normalized",
    }
    out = ActivityMethod()._run(_adata_with_zero_cell(), cfg, _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)
