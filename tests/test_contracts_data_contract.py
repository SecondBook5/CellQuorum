"""Tests for the composite DataContract validator."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.contracts.data_contract import DataContract
from cellquorum.core.contracts.exceptions import CellQuorumContractError
from cellquorum.core.contracts.layer_tags import set_layer_tag


def _adata_lognorm():
    # 4 cells x 3 genes; a valid log-normalized layer + counts layer.
    rng = np.arange(12, dtype=np.float32).reshape(4, 3) / 7.0  # fractional => non-integer
    counts = np.arange(12, dtype=np.float32).reshape(4, 3)
    a = ad.AnnData(
        X=rng,
        obs=pd.DataFrame(
            {"condition": ["Normal", "Lymphedema", "Normal", "Lymphedema"]},
            index=[f"c{i}" for i in range(4)],
        ),
        var=pd.DataFrame(index=["KRT14", "IL33", "TPSB2"]),
    )
    a.layers["lognorm"] = rng
    a.layers["counts"] = counts
    set_layer_tag(a, "lognorm", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    set_layer_tag(a, "counts", kind="counts")
    a.uns["cellquorum"]["preprocessing"] = {
        "normalization": {"recipe": "cellquorum_pf_log1p_pf_v1"}
    }
    a.obsm["X_pca"] = np.zeros((4, 2), dtype=np.float32)
    return a


def test_valid_object_passes():
    c = DataContract(
        required_layers=["lognorm", "counts"],
        required_obs=["condition"],
        required_var=["IL33"],
        required_obsm=["X_pca"],
        expression_layer="lognorm",
        expected_kind="lognorm",
        expected_recipe="cellquorum_pf_log1p_pf_v1",
    )
    c.validate(_adata_lognorm())  # no raise


def test_missing_layer_raises():
    c = DataContract(required_layers=["scaled"])
    with pytest.raises(CellQuorumContractError, match="layer"):
        c.validate(_adata_lognorm())


def test_missing_obs_raises():
    c = DataContract(required_obs=["patient_id"])
    with pytest.raises(CellQuorumContractError, match="obs"):
        c.validate(_adata_lognorm())


def test_missing_var_raises():
    c = DataContract(required_var=["FOXP3"])
    with pytest.raises(CellQuorumContractError, match="var"):
        c.validate(_adata_lognorm())


def test_missing_obsm_raises():
    c = DataContract(required_obsm=["X_umap"])
    with pytest.raises(CellQuorumContractError, match="obsm"):
        c.validate(_adata_lognorm())


def test_wrong_recipe_raises():
    c = DataContract(expression_layer="lognorm", expected_recipe="some_other_recipe")
    with pytest.raises(CellQuorumContractError, match="recipe"):
        c.validate(_adata_lognorm())


def test_raw_counts_in_lognorm_layer_raises():
    a = _adata_lognorm()
    # Sabotage: overwrite lognorm with integer counts, keep the lognorm tag.
    a.layers["lognorm"] = a.layers["counts"].copy()
    c = DataContract(expression_layer="lognorm", expected_kind="lognorm")
    with pytest.raises(CellQuorumContractError, match="integers"):
        c.validate(a)


def test_forbid_imputed_raises_on_imputed_layer():
    a = _adata_lognorm()
    a.layers["magic"] = a.layers["lognorm"].copy()
    set_layer_tag(a, "magic", kind="imputed", recipe="magic_v1")
    c = DataContract(expression_layer="magic", forbid_imputed=True)
    with pytest.raises(CellQuorumContractError, match="imputed"):
        c.validate(a)
