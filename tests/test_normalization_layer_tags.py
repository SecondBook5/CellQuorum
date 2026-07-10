"""Normalization output carries layer kind/recipe tags satisfying a contract."""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.contracts import DataContract, get_layer_tag
from cellquorum.preprocessing.config import NormalizationConfig
from cellquorum.preprocessing.normalization import normalize_adata


def _counts_adata(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.poisson(2.0, size=(50, 20)).astype(np.float32)
    return ad.AnnData(X=x)


def test_normalization_tags_counts_and_output_layers():
    a = _counts_adata()
    # Use the PROJECT DEFAULT recipe (cellquorum_pf_log1p_pf_v1, shifted-CLR).
    # It legitimately produces small negative values; the lognorm contract does
    # not assert non-negativity, so the default must satisfy an expected_kind
    # contract — that is the whole point of this seam.
    cfg = NormalizationConfig()
    # normalize_adata(copy=True by default) RETURNS a NormalizationResult; the
    # tags live on result.adata, not the input object.
    result = normalize_adata(a, cfg)
    out = result.adata

    counts_tag = get_layer_tag(out, cfg.preserve_counts_layer)
    out_tag = get_layer_tag(out, cfg.output_layer)
    assert counts_tag is not None and counts_tag["kind"] == "counts"
    assert out_tag is not None and out_tag["kind"] == "lognorm"
    assert out_tag["recipe"] == cfg.recipe


def test_normalized_output_satisfies_expected_kind_contract():
    a = _counts_adata()
    # The project-default PFlog1pPF output must satisfy the contract with no
    # manual tagging — including its small negative (centered) values.
    cfg = NormalizationConfig()
    result = normalize_adata(a, cfg)
    DataContract(
        expression_layer=cfg.output_layer,
        expected_kind="lognorm",
        expected_recipe=cfg.recipe,
    ).validate(result.adata)
