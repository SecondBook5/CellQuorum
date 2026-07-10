"""Regression: the corrected lekc KC object satisfies its declared contract,
and a sabotaged copy is rejected. Skips when the data file is unavailable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")

from cellquorum.contracts import (  # noqa: E402
    CellQuorumContractError,
    DataContract,
    LabelContract,
    set_layer_tag,
)

KC_PATH = Path(
    "/mnt/c/Users/ajboo/BookAbraham/le_kc_signaling_hubs/"
    "data/processed/internal/cohort_objects/le_kc_keratinocyte_refmapped.h5ad"
)


@pytest.fixture(scope="module")
def kc_adata():
    if not KC_PATH.is_file():
        pytest.skip(f"lekc KC object not present at {KC_PATH}")
    return anndata.read_h5ad(KC_PATH)


def test_corrected_kc_object_meets_contract(kc_adata):
    # The object stores PFlog1pPF in .X (uns['normalization']=='pflog1ppf').
    # Tag X as lognorm from the recorded recipe so the semantic check has a tag.
    a = kc_adata.copy()
    set_layer_tag(a, "X", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    contract = DataContract(
        required_obs=["condition", "patient_id", "ref_state"],
        expression_layer="X",
        expected_kind="lognorm",
    )
    contract.validate(a)  # no raise

    labels = LabelContract(
        label_col="ref_state",
        expected_labels=["KC 1", "KC 3", "KC cycling"],
        condition_col="condition",
        expected_conditions=["Normal", "Lymphedema"],
    )
    labels.validate(a)


def test_sabotaged_kc_object_is_rejected(kc_adata):
    # Put raw counts into X and tag it lognorm — must be caught.
    a = kc_adata.copy()
    a.X = (
        a.layers["counts"].copy()
        if "counts" in a.layers
        else np.round(np.abs(a.X.toarray()) if hasattr(a.X, "toarray") else np.round(np.abs(a.X)))
    )
    set_layer_tag(a, "X", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    contract = DataContract(expression_layer="X", expected_kind="lognorm")
    with pytest.raises(CellQuorumContractError):
        contract.validate(a)
