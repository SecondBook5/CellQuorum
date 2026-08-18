# tests/test_ccc_tensor_method.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.cell_cell_communication.tensor_c2c_method import TensorCell2CellMethod
from cellquorum.contracts.layer_tags import set_layer_tag
from cellquorum.methods.base import MethodSkip

pytest.importorskip("liana")
pytest.importorskip("cell2cell")


class _Paths:
    def __init__(self, tmp):
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)
        self.config = None


def _config(rank=2):
    return {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "layer": "cellquorum_normalized",
        "seed": 42,
        "rank": rank,
        "tf_optimization": "regular",
        "min_samples": 3,
        "tensor_how": "outer",
        "outer_fraction": 1.0 / 3.0,
    }


def _adata_with_liana_res(n_samples=4):
    """Minimal AnnData carrying a synthetic per-sample liana_res long frame with
    magnitude_rank, source/target, ligand/receptor across several samples."""
    genes = ["LGALS9", "PTPRC", "MET", "CD44"]
    obs = pd.DataFrame(
        {"cell_type": ["A", "B"] * 4, "sample_id": [f"s{i}" for i in range(4) for _ in range(2)]}
    )
    a = ad.AnnData(X=np.abs(np.random.default_rng(0).random((8, len(genes)))) + 0.5, obs=obs)
    a.var_names = genes
    a.layers["cellquorum_normalized"] = a.X.copy()
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm")

    rows = []
    rng = np.random.default_rng(1)
    pairs = [("LGALS9", "PTPRC"), ("LGALS9", "MET"), ("LGALS9", "CD44")]
    for s in range(n_samples):
        for src in ("A", "B"):
            for tgt in ("A", "B"):
                for lig, rec in pairs:
                    rows.append(
                        {
                            "sample": f"s{s}",
                            "source": src,
                            "target": tgt,
                            "ligand_complex": lig,
                            "receptor_complex": rec,
                            "magnitude_rank": float(rng.random()),
                        }
                    )
    a.uns["liana_res"] = pd.DataFrame(rows)
    return a


def test_tensor_decomposes_and_writes_four_tables(tmp_path):
    a = _adata_with_liana_res()
    result = TensorCell2CellMethod().run(a, _config(rank=2), _Ctx(tmp_path))
    assert not isinstance(result, MethodSkip), getattr(result, "reason", "")
    d = tmp_path / "results" / "cell_cell_communication"
    for name in ("contexts", "lr_pairs", "senders", "receivers"):
        assert (d / f"tensor_factors_{name}.csv").exists()
    assert "tensor_c2c" in result.adata.uns
    assert set(result.adata.uns["tensor_c2c"]) >= {"contexts", "lr_pairs", "senders", "receivers"}


def test_tensor_skips_without_liana_res(tmp_path):
    a = _adata_with_liana_res()
    del a.uns["liana_res"]
    result = TensorCell2CellMethod().run(a, _config(), _Ctx(tmp_path))
    assert isinstance(result, MethodSkip)
    assert "liana_res" in result.reason


def test_tensor_skips_on_partial_dict_liana_res(tmp_path):
    # Regression: liana's by_sample aborts mid-loop on a sparse sample and can
    # leave a partial ``{sample: DataFrame}`` dict in uns['liana_res']. That dict
    # has len>0, so it passes the presence guard; tensor_c2c must NOT then crash
    # on ``res.columns`` (AttributeError) — it must skip cleanly.
    a = _adata_with_liana_res()
    a.uns["liana_res"] = {"s0": a.uns["liana_res"]}  # simulate the partial dict
    result = TensorCell2CellMethod().run(a, _config(), _Ctx(tmp_path))
    assert isinstance(result, MethodSkip)
    assert "tabular" in result.reason.lower() or "liana_res" in result.reason


def test_tensor_skips_below_min_samples(tmp_path):
    a = _adata_with_liana_res(n_samples=2)
    result = TensorCell2CellMethod().run(a, _config(), _Ctx(tmp_path))
    assert isinstance(result, MethodSkip)
    assert "sample" in result.reason.lower()
