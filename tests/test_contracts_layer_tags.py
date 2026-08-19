"""Tests for layer provenance tag read/write helpers."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.contracts.layer_tags import (
    get_layer_tag,
    get_normalization_recipe,
    set_layer_tag,
)


def _adata():
    return ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))


def test_set_and_get_layer_tag():
    a = _adata()
    set_layer_tag(a, "lognorm", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")
    tag = get_layer_tag(a, "lognorm")
    assert tag == {"kind": "lognorm", "recipe": "cellquorum_pf_log1p_pf_v1"}


def test_get_layer_tag_absent_is_none():
    assert get_layer_tag(_adata(), "missing") is None


def test_set_layer_tag_counts_has_no_recipe():
    a = _adata()
    set_layer_tag(a, "counts", kind="counts")
    assert get_layer_tag(a, "counts") == {"kind": "counts", "recipe": None}


def test_get_normalization_recipe_reads_preprocessing_provenance():
    a = _adata()
    a.uns["cellquorum"] = {
        "preprocessing": {"normalization": {"recipe": "cellquorum_pf_log1p_pf_v1"}}
    }
    assert get_normalization_recipe(a) == "cellquorum_pf_log1p_pf_v1"


def test_get_normalization_recipe_absent_is_none():
    assert get_normalization_recipe(_adata()) is None


def test_set_layer_tag_rejects_invalid_kind():
    a = _adata()
    with pytest.raises(ValueError, match="Unknown layer kind"):
        set_layer_tag(a, "lognorm", kind="bogus")


def test_get_normalization_recipe_partial_path_is_none():
    # Defensive navigation: a partial uns path (no 'normalization' key) returns None, not KeyError.
    a = _adata()
    a.uns["cellquorum"] = {"preprocessing": {}}
    assert get_normalization_recipe(a) is None
