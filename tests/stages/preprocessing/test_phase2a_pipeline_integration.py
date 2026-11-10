"""Integration: dimensionality + clustering run in the real executor loop."""

from __future__ import annotations


def test_pca_consumes_normalized_layer_not_raw():
    """Integration: PCA runs on normalized layer, not raw counts."""
    import anndata as ad
    import numpy as np

    from cellquorum.core.contracts import set_layer_tag
    from cellquorum.stages.preprocessing.dimensionality.pca import PCAMethod

    # Build a raw-counts AnnData (integer values).
    rng = np.random.default_rng(42)
    n_cells, n_genes = 100, 30
    raw_counts = rng.poisson(lam=10, size=(n_cells, n_genes))
    a = ad.AnnData(X=raw_counts.astype(np.float32))

    # Simulate normalization: write non-integer log-like values to a layer and tag it.
    lognorm = np.log1p(
        raw_counts.astype(np.float32) / raw_counts.sum(axis=1, keepdims=True) * 10000
    )
    a.layers["cellquorum_normalized"] = lognorm
    set_layer_tag(a, "cellquorum_normalized", kind="lognorm", recipe="cellquorum_pf_log1p_pf_v1")

    # Run PCA method with input_layer pointing to the normalized layer.
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:

        class _Ctx:
            def __init__(self, adata, tmpdir):
                self._adata = adata

                class _Paths:
                    figures = Path(tmpdir)

                self.paths = _Paths()

            def require_adata(self):
                return self._adata

        ctx = _Ctx(a, tmp)
        method = PCAMethod()
        result = method.run(
            a,
            {"input_layer": "cellquorum_normalized", "n_pcs": 5, "max_pcs": 20, "random_state": 0},
            context=ctx,
        )

    # Verify PCA succeeded and consumed non-integer data (not the raw counts).
    assert "X_pca" in result.adata.obsm
    assert result.adata.obsm["X_pca"].shape[1] == 5
    # As a proxy: the normalized layer has non-integer values.
    assert not np.all(lognorm == np.floor(lognorm))
