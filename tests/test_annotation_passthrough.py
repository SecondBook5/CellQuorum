"""Tests for the passthrough annotation method.

The passthrough method exists for the hypothesis-repo workflow: a per-cell-type
subrepo subsets an ALREADY-annotated global object, so its cells carry a trusted
label (e.g. cell_type=Fibroblasts) and the mandatory annotation stage must
PRESERVE that label rather than recompute (and destroy) it.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.annotation.passthrough import PassthroughAnnotationMethod
from cellquorum.core.contracts import CellQuorumContractError
from cellquorum.methods.base import MethodSkip


def _labeled_adata(seed=0):
    """A subset object that already carries a trusted cell_type label."""
    rng = np.random.default_rng(seed)
    n = 40
    x = rng.random((n, 5)).astype(np.float32)
    a = ad.AnnData(X=x, var=pd.DataFrame(index=[f"g{i}" for i in range(5)]))
    a.obs["leiden"] = pd.Categorical(["0"] * (n // 2) + ["1"] * (n // 2))
    a.obs["cell_type"] = pd.Categorical(["Fibroblasts"] * n)
    return a


def test_passthrough_preserves_existing_label():
    """With key_added already present, passthrough leaves it untouched."""
    m = PassthroughAnnotationMethod()
    a = _labeled_adata()
    cfg = {"key_added": "cell_type"}
    result = m.run(a, cfg, context=None)

    assert not isinstance(result, MethodSkip)
    ct = result.adata.obs["cell_type"]
    # Every cell keeps its trusted label; nothing is nulled.
    assert ct.notna().all()
    assert set(ct.unique()) == {"Fibroblasts"}


def test_passthrough_copies_from_source_key():
    """When source_key differs from key_added, the label is copied over."""
    m = PassthroughAnnotationMethod()
    a = _labeled_adata()
    a.obs["global_label"] = a.obs["cell_type"]
    del a.obs["cell_type"]
    cfg = {"key_added": "cell_type", "source_key": "global_label"}
    result = m.run(a, cfg, context=None)

    assert not isinstance(result, MethodSkip)
    assert "cell_type" in result.adata.obs
    assert set(result.adata.obs["cell_type"].unique()) == {"Fibroblasts"}


def test_passthrough_raises_when_source_absent():
    """No trusted label to preserve is a hard contract violation, not a null."""
    m = PassthroughAnnotationMethod()
    a = _labeled_adata()
    del a.obs["cell_type"]
    cfg = {"key_added": "cell_type"}
    with pytest.raises(CellQuorumContractError, match="cell_type"):
        m.run(a, cfg, context=None)


def test_passthrough_is_registered():
    """The method self-registers under the annotation stage category."""
    import cellquorum.stages.annotation  # noqa: F401  (import triggers registration)
    from cellquorum.methods.registry import METHOD_REGISTRY

    method_cls = METHOD_REGISTRY.get("annotation", "passthrough")
    assert method_cls is PassthroughAnnotationMethod
