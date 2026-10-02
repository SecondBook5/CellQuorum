"""Shared provenance-recording for the integration methods (Harmony, scVI, scANVI).

All three wrote the same two-record pattern -- a single-method 'last wins' slot plus a
per-method slot namespaced by output_rep -- by hand, independently. Extracted so a future
field can't be added to one method's copy and forgotten on the other two.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.stages.integration._provenance import record_integration_provenance


def _adata(n: int = 5) -> ad.AnnData:
    return ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))


def test_writes_both_the_single_method_and_per_method_records():
    adata = _adata()

    record_integration_provenance(
        adata, method="harmony", batch_key="batch", output_rep="X_pca_harmony", input_rep="X_pca"
    )

    cq = adata.uns["cellquorum"]
    assert cq["integration"] == {
        "method": "harmony",
        "batch_key": "batch",
        "output_rep": "X_pca_harmony",
        "input_rep": "X_pca",
    }
    assert cq["integration_methods"]["X_pca_harmony"] == cq["integration"]


def test_a_second_method_call_adds_a_namespaced_record_without_clobbering_the_first():
    adata = _adata()
    record_integration_provenance(
        adata, method="harmony", batch_key="batch", output_rep="X_pca_harmony", input_rep="X_pca"
    )

    record_integration_provenance(
        adata, method="scvi", batch_key="batch", output_rep="X_scvi", n_latent=30
    )

    cq = adata.uns["cellquorum"]
    # Single-method slot is last-wins, by design.
    assert cq["integration"]["method"] == "scvi"
    # But both per-method records survive, namespaced by output_rep.
    assert set(cq["integration_methods"]) == {"X_pca_harmony", "X_scvi"}
    assert cq["integration_methods"]["X_pca_harmony"]["method"] == "harmony"
    assert cq["integration_methods"]["X_scvi"]["n_latent"] == 30
