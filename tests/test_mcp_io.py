"""IO helpers for the DIALOGUE method (biology-free)."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.multicellular_programs._dialogue_io import (
    export_dialogue_inputs,
    read_dialogue_outputs,
)


def _adata():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.poisson(2.0, size=(n, 40)).astype(float)
    obs = pd.DataFrame(
        {
            "cell_type": ["Type_A" if i % 2 else "TypeB" for i in range(n)],
            "sample_id": [f"s{i % 6}" for i in range(n)],
            "condition": ["case" if (i % 6) % 2 else "ctrl" for i in range(n)],
        },
        index=[f"c{i}" for i in range(n)],
    )
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = [f"G{i}" for i in range(40)]
    a.obsm["X_pca"] = rng.normal(size=(n, 8))
    return a


def test_export_writes_per_celltype_files(tmp_path):
    a = _adata()
    meta = export_dialogue_inputs(
        a,
        cell_type_col="cell_type",
        sample_col="sample_id",
        use_rep="X_pca",
        n_pcs=5,
        layer=None,
        quality_col=None,
        condition_col="condition",
        confounders=[],
        min_cells_per_type=20,
        scratch=tmp_path,
    )
    ct_map = json.loads((tmp_path / "celltypes.json").read_text())
    assert set(m["label"] for m in ct_map.values()) == {"Type_A", "TypeB"}
    # underscores stripped in the key, original label preserved in value
    assert "TypeA" in ct_map and ct_map["TypeA"]["label"] == "Type_A"
    assert meta["n_samples"] == 6
    for stripped in ct_map:
        d = tmp_path / stripped
        assert (d / "expr.mtx").exists()
        assert (d / "X.csv").exists()
        m = pd.read_csv(d / "meta.csv")
        assert {"cell", "sample", "cellQ"} <= set(m.columns)
        assert pd.read_csv(d / "X.csv").shape[1] == 6  # cell + 5 PCs


def test_read_outputs_missing_files_are_empty(tmp_path):
    out = read_dialogue_outputs(tmp_path)
    assert list(out["programs"].columns) == ["program", "cell_type", "gene", "loading", "direction"]
    assert out["programs"].empty
