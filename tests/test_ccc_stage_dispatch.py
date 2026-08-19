# tests/test_ccc_stage_dispatch.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.cell_cell_communication.stage import CellCellCommunicationStage
from cellquorum.core.contracts.layer_tags import set_layer_tag

pytest.importorskip("liana")


class _Paths:
    def __init__(self, tmp):
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Cfg:
    def __init__(self):
        self.cell_cell_communication = {
            "enabled": True,
            "cell_type_col": "cell_type",
            "sample_col": "sample_id",
            "layer": "cellquorum_normalized",
            "seed": 42,
            "min_cells": 2,
            "n_perms": 10,
            "min_samples": 3,
            "rank": 2,
            "tf_optimization": "regular",
        }
        self.cohort = None


class _Ctx:
    def __init__(self, tmp, adata):
        self.config = _Cfg()
        self.paths = _Paths(tmp)
        self._adata = adata

    def require_adata(self):
        return self._adata


def _adata():
    genes = ["LGALS9", "PTPRC", "MET", "CD44"]
    rng = np.random.default_rng(0)
    cts, samples = [], []
    for s in ("s1", "s2", "s3", "s4"):
        for ct in ("A", "B"):
            cts += [ct] * 6
            samples += [s] * 6
    n = len(cts)
    X = rng.random((n, len(genes))) + 0.5
    a = ad.AnnData(X=X, obs=pd.DataFrame({"cell_type": cts, "sample_id": samples}))
    a.var_names = genes
    a.layers["cellquorum_normalized"] = X.copy()
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm")
    return a


def test_stage_injects_default_methods_liana_first():
    stage = CellCellCommunicationStage()
    augmented = stage._augment_config(_Ctx.__new__(_Ctx), {"enabled": True})
    names = [m["method"] for m in augmented["methods"]]
    assert names == ["liana", "tensor_c2c"]


def test_stage_forces_liana_before_tensor():
    stage = CellCellCommunicationStage()
    cfg = {"methods": [{"method": "tensor_c2c"}, {"method": "liana"}]}
    augmented = stage._augment_config(_Ctx.__new__(_Ctx), cfg)
    names = [m["method"] for m in augmented["methods"]]
    assert names.index("liana") < names.index("tensor_c2c")


def test_stage_runs_both_methods(tmp_path):
    a = _adata()
    result = CellCellCommunicationStage().run(_Ctx(tmp_path, a))
    # Both methods must be dispatched, liana before tensor. On this deliberately
    # minimal fixture (4 genes) liana cannot clear its resource-coverage check, so
    # both skip cleanly — the point here is that the stage runs both and does NOT
    # crash (previously a partial liana result crashed tensor_c2c on ``.columns``).
    assert result.metrics["n_methods"] == 2
    methods = [m["method"] for m in result.metrics["per_method"]]
    assert methods == ["liana", "tensor_c2c"]
    # liana must never leave a non-DataFrame artifact behind for tensor to trip on.
    assert "liana_res" not in result.adata.uns or isinstance(
        result.adata.uns["liana_res"], pd.DataFrame
    )
