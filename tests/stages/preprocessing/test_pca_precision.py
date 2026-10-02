"""PCA precision changes arithmetic without rewriting normalized expression."""

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from cellquorum.config.models import DimensionalityConfig
from cellquorum.stages.preprocessing.dimensionality.pca import PCAMethod


@pytest.mark.parametrize("fails", [False, True])
def test_precision_restores_source_layer(monkeypatch, fails):
    matrix = sparse.csr_matrix(np.eye(4, dtype=np.float32))
    data = ad.AnnData(matrix.copy())
    data.layers["normalized"] = matrix
    method = PCAMethod()

    def backend(adata, **kwargs):
        assert adata.layers["normalized"].dtype == np.float64
        if fails:
            raise RuntimeError("device failure")
        adata.uns["pca"] = {"params": {}}
        return "gpu", None

    monkeypatch.setattr(method, "_run_scanpy_pca_backend", backend)
    kwargs = dict(
        input_layer="normalized",
        n_comps=2,
        mask_var=None,
        random_state=0,
        context=None,
        precision="float64",
    )
    if fails:
        with pytest.raises(RuntimeError, match="device failure"):
            method._run_scanpy_pca(data, **kwargs)
    else:
        method._run_scanpy_pca(data, **kwargs)
        assert data.uns["pca"]["params"]["arithmetic_dtype"] == "float64"
    assert data.layers["normalized"] is matrix
    assert matrix.dtype == np.float32


def test_precision_config_rejects_unknown_dtype():
    with pytest.raises(ValueError):
        DimensionalityConfig(precision="float16")
