"""Normalization output carries layer kind/recipe tags satisfying a contract."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.contracts import DataContract, get_layer_tag
from cellquorum.preprocessing.config import NormalizationConfig
from cellquorum.preprocessing.normalization import normalize_adata


def _counts_adata(seed=0):
    # NB-distributed (overdispersed) so scclr's target="auto" alpha estimate is valid.
    rng = np.random.default_rng(seed)
    x = rng.negative_binomial(2, 0.15, size=(60, 30)).astype(np.float32)
    return ad.AnnData(X=x)


def _scclr_backend_or_skip():
    """Return an available scclr backend, or skip when its isolated env is absent."""

    from cellquorum.backends.scclr_backend import build_scclr_backend

    backend = build_scclr_backend()
    if not backend.status().available:
        pytest.skip("scclr environment unavailable (isolated micromamba env not built)")
    return backend


def test_normalization_tags_counts_and_output_layers(tmp_path):
    a = _counts_adata()
    # The PROJECT DEFAULT recipe (cellquorum_pf_log1p_pf_v1) is the real PFlog1pPF
    # run through the scclr backend. Its lognorm output must satisfy an
    # expected_kind/recipe contract — that is the whole point of this seam.
    backend = _scclr_backend_or_skip()
    cfg = NormalizationConfig()
    # normalize_adata(copy=True by default) RETURNS a NormalizationResult; the
    # tags live on result.adata, not the input object.
    result = normalize_adata(a, cfg, backend=backend, scratch_dir=tmp_path)
    out = result.adata

    counts_tag = get_layer_tag(out, cfg.preserve_counts_layer)
    out_tag = get_layer_tag(out, cfg.output_layer)
    assert counts_tag is not None and counts_tag["kind"] == "counts"
    assert out_tag is not None and out_tag["kind"] == "lognorm"
    assert out_tag["recipe"] == cfg.recipe


def test_normalized_output_satisfies_expected_kind_contract(tmp_path):
    a = _counts_adata()
    # The project-default PFlog1pPF output must satisfy the contract with no
    # manual tagging — including its small negative (centered) values.
    backend = _scclr_backend_or_skip()
    cfg = NormalizationConfig()
    result = normalize_adata(a, cfg, backend=backend, scratch_dir=tmp_path)
    DataContract(
        expression_layer=cfg.output_layer,
        expected_kind="lognorm",
        expected_recipe=cfg.recipe,
    ).validate(result.adata)
