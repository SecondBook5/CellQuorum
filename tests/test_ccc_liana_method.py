from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.cell_cell_communication.liana_method import LianaMethod
from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.methods.base import MethodSkip

li = pytest.importorskip("liana")


class _Paths:
    def __init__(self, tmp):
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)
        self.config = None


def _config():
    return {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "layer": "cellquorum_normalized",
        "seed": 42,
        "resource_name": "consensus",
        "expr_prop": 0.1,
        "min_cells": 2,
        "n_perms": 10,
    }


def _adata(n_per_group=8):
    """Two cell types across two samples, expressing real consensus LR genes so
    rank_aggregate returns non-empty results."""
    # Use sufficient genes from the liana consensus resource to pass feature checks
    genes = [
        "LGALS9",
        "PTPRC",
        "MET",
        "CD44",
        "CD47",
        "LRP1",
        "CD4",
        "CD8A",
        "CD3D",
        "ICAM1",
        "VCAM1",
        "IL7",
        "IL7R",
        "CD11B",
        "CD11C",
        "CD14",
        "CD16",
        "CD19",
        "CD20",
        "CD27",
        "CD28",
        "CD38",
        "CD40",
        "CD45",
        "CD45RA",
        "CD45RO",
        "CD56",
        "CD57",
        "CD62L",
        "CD68",
        "CD69",
        "CD70",
        "CD74",
        "CD79A",
        "CD80",
        "CD86",
        "CD95",
        "CD127",
        "CD152",
        "CD155",
        "CD226",
        "CD244",
        "CD247",
        "CD274",
        "CD278",
        "PDCD1",
        "LAG3",
        "HAVCR2",
        "ENTPD1",
        "ADORA2A",
        "TIGIT",
        "IL2",
        "IL4",
        "IL10",
        "TNF",
        "IFNG",
        "GZMA",
        "GZMB",
        "EOMES",
        "TBX21",
        "CCL2",
        "CCL3",
        "CCL4",
        "CCL5",
        "CCL20",
        "CCL21",
        "CXCL8",
        "CXCL10",
        "CXCL12",
    ]
    # Remove duplicates and ensure we have enough
    genes = list(dict.fromkeys(genes))[:60]  # keep ~60 unique genes

    rng = np.random.default_rng(0)
    cell_types, samples = [], []
    for s in ("s1", "s2"):
        for ct in ("A", "B"):
            cell_types += [ct] * n_per_group
            samples += [s] * n_per_group
    n = len(cell_types)
    X = rng.random(size=(n, len(genes))) + 0.5  # positive lognorm-like
    a = ad.AnnData(X=X, obs=pd.DataFrame({"cell_type": cell_types, "sample_id": samples}))
    a.var_names = genes
    a.layers["cellquorum_normalized"] = X.copy()
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm")
    return a


def test_liana_writes_per_sample_result(tmp_path):
    a = _adata()
    result = LianaMethod().run(a, _config(), _Ctx(tmp_path))
    assert not isinstance(result, MethodSkip)
    assert "liana_res" in result.adata.uns
    df = result.adata.uns["liana_res"]
    assert "sample" in df.columns
    assert len(df) > 0
    assert (tmp_path / "results" / "cell_cell_communication" / "liana_ranks.csv").exists()


def test_liana_skips_single_cell_type(tmp_path):
    a = _adata()
    a.obs["cell_type"] = "A"  # only one type
    result = LianaMethod().run(a, _config(), _Ctx(tmp_path))
    assert isinstance(result, MethodSkip)
    assert "cell type" in result.reason.lower()


def test_liana_skips_when_layer_untagged(tmp_path):
    """Contract requires an explicit lognorm tag; an untagged layer must raise a
    contract error, which the stage layer converts to a skip. Here we call run()
    directly and assert it raises the contract error (stage-level skip is Task 6)."""
    from cellquorum.core.contracts import CellQuorumContractError

    a = _adata()
    del a.uns["cellquorum"]["layer_tags"]["cellquorum_normalized"]
    with pytest.raises(CellQuorumContractError):
        LianaMethod().run(a, _config(), _Ctx(tmp_path))


def test_liana_skips_when_dependency_absent(tmp_path, monkeypatch):
    """Simulate liana being unimportable → MethodSkip, not crash."""
    import builtins

    a = _adata()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "liana" or name.startswith("liana."):
            raise ImportError("simulated: no liana")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = LianaMethod().run(a, _config(), _Ctx(tmp_path))
    assert isinstance(result, MethodSkip)
    assert "unavailable" in result.reason.lower()
