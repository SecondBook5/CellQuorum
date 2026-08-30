from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.cell_cell_communication.tensor_c2c_method import TensorCell2CellMethod
from cellquorum.core.contracts.layer_tags import set_layer_tag

pytest.importorskip("liana")
pytest.importorskip("cell2cell")


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)
        self.config = None


def _adata_with_liana_res():
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
    for s in range(4):
        for src in ("A", "B"):
            for tgt in ("A", "B"):
                for lig, rec in (("LGALS9", "PTPRC"), ("LGALS9", "MET"), ("LGALS9", "CD44")):
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


def _cfg():
    return {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "layer": "cellquorum_normalized",
        "seed": 42,
        "rank": 2,
        "tf_optimization": "regular",
        "min_samples": 3,
        "tensor_how": "outer",
        "outer_fraction": 1.0 / 3.0,
    }


def test_same_seed_same_loadings(tmp_path):
    r1 = TensorCell2CellMethod().run(_adata_with_liana_res(), _cfg(), _Ctx(tmp_path / "a"))
    r2 = TensorCell2CellMethod().run(_adata_with_liana_res(), _cfg(), _Ctx(tmp_path / "b"))
    for key in ("contexts", "lr_pairs", "senders", "receivers"):
        v1 = r1.adata.uns["tensor_c2c"][key].to_numpy()
        v2 = r2.adata.uns["tensor_c2c"][key].to_numpy()
        assert np.allclose(v1, v2), f"non-deterministic loadings for {key}"


def test_elbow_rank_selection_deterministic(tmp_path):
    cfg = dict(_cfg())
    cfg["rank"] = None  # force the seeded elbow_rank_selection path
    r1 = TensorCell2CellMethod().run(_adata_with_liana_res(), cfg, _Ctx(tmp_path / "a"))
    r2 = TensorCell2CellMethod().run(_adata_with_liana_res(), cfg, _Ctx(tmp_path / "b"))
    # If elbow skipped for a benign reason, both must skip identically — never a crash.
    from cellquorum.methods.base import MethodSkip

    if isinstance(r1, MethodSkip) or isinstance(r2, MethodSkip):
        assert isinstance(r1, MethodSkip) and isinstance(r2, MethodSkip)
        return
    assert r1.metrics["rank"] == r2.metrics["rank"], "elbow selected different ranks"
    for key in ("contexts", "lr_pairs", "senders", "receivers"):
        v1 = r1.adata.uns["tensor_c2c"][key].to_numpy()
        v2 = r2.adata.uns["tensor_c2c"][key].to_numpy()
        assert np.allclose(v1, v2), f"non-deterministic elbow loadings for {key}"
