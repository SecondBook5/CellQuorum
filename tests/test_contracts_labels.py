"""Tests for label/condition token validation."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.contracts.exceptions import CellQuorumContractError
from cellquorum.contracts.labels import LabelContract


def _adata():
    obs = pd.DataFrame(
        {
            "ref_state": ["KC 1", "KC 3", "KC 1", "KC cycling"],
            "condition": ["Normal", "Lymphedema", "Normal", "Lymphedema"],
        },
        index=[f"c{i}" for i in range(4)],
    )
    return ad.AnnData(X=np.zeros((4, 2), dtype=np.float32), obs=obs)


def test_valid_labels_and_conditions_pass():
    c = LabelContract(
        label_col="ref_state",
        expected_labels=["KC 1", "KC 3"],
        condition_col="condition",
        expected_conditions=["Normal", "Lymphedema"],
    )
    c.validate(_adata())  # no raise


def test_missing_label_col_raises():
    c = LabelContract(label_col="kc_named")
    with pytest.raises(CellQuorumContractError, match="kc_named"):
        c.validate(_adata())


def test_stale_expected_label_raises():
    # 'LE-enriched KC' is a legacy token absent from the rebuilt object.
    c = LabelContract(label_col="ref_state", expected_labels=["LE-enriched KC"])
    with pytest.raises(CellQuorumContractError, match="LE-enriched KC"):
        c.validate(_adata())


def test_stale_condition_token_raises():
    # 'LE' is the hardcoded-literal bug; real label is 'Lymphedema'.
    c = LabelContract(label_col="ref_state", condition_col="condition", expected_conditions=["LE"])
    with pytest.raises(CellQuorumContractError, match="LE"):
        c.validate(_adata())


def test_select_returns_subset():
    c = LabelContract(label_col="ref_state")
    sub = c.select(_adata(), "KC 1")
    assert sub.n_obs == 2


def test_select_unknown_label_raises_not_empty():
    c = LabelContract(label_col="ref_state")
    with pytest.raises(CellQuorumContractError, match="KC 9"):
        c.select(_adata(), "KC 9")
