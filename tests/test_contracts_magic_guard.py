"""Tests for the standalone MAGIC/imputed-layer guard."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.contracts.exceptions import CellQuorumContractError
from cellquorum.contracts.layer_tags import set_layer_tag
from cellquorum.contracts.magic_guard import assert_not_imputed


def _adata():
    a = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
    a.layers["lognorm"] = np.zeros((3, 2), dtype=np.float32)
    a.layers["magic"] = np.zeros((3, 2), dtype=np.float32)
    set_layer_tag(a, "lognorm", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    set_layer_tag(a, "magic", kind="imputed", recipe="magic_v1")
    return a


def test_non_imputed_layer_passes():
    assert_not_imputed(_adata(), "lognorm")  # no raise


def test_imputed_layer_raises():
    with pytest.raises(CellQuorumContractError, match="imputed"):
        assert_not_imputed(_adata(), "magic")


def test_untagged_layer_passes():
    # No tag => not known to be imputed => allowed (kept permissive by design).
    a = _adata()
    a.layers["raw"] = np.zeros((3, 2), dtype=np.float32)
    assert_not_imputed(a, "raw")
