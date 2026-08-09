from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.cell_cell_communication.stage import CellCellCommunicationStage
from cellquorum.contracts.layer_tags import set_layer_tag

pytest.importorskip("liana")
pytest.importorskip("cell2cell")


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
    X = rng.random((len(cts), len(genes))) + 0.5
    a = ad.AnnData(X=X, obs=pd.DataFrame({"cell_type": cts, "sample_id": samples}))
    a.var_names = genes
    a.layers["cellquorum_normalized"] = X.copy()
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm")
    return a


def test_liana_output_threads_into_tensor(tmp_path):
    result = CellCellCommunicationStage().run(_Ctx(tmp_path, _adata()))
    # Both methods ran; liana_res present; tensor consumed it and wrote loadings.
    assert result.metrics["n_methods"] == 2
    assert "liana_res" in result.adata.uns
    per_method = result.metrics["per_method"]
    tensor_entry = next(m for m in per_method if m.get("method") == "tensor_c2c")
    # tensor either produced loadings or recorded an honest skip — never crashed.
    assert "tensor_c2c" in result.adata.uns or tensor_entry.get("skipped")
