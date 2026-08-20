"""The ambient stage must join manifest metadata onto obs by sample_id."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.ambient_correction.stage import _join_manifest_metadata


def _corrected(sample_id, n):
    a = ad.AnnData(X=np.ones((n, 3), dtype="float32"))
    a.obs_names = [f"{sample_id}_bc{i}" for i in range(n)]
    a.obs["sample_id"] = sample_id
    return a


def test_join_manifest_metadata_maps_columns_by_sample_id():
    adata = ad.concat([_corrected("s1", 2), _corrected("s2", 3)], join="outer")
    manifest = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "cellranger_path": ["A/s1", "A/s2"],
            "condition": ["Normal", "LE"],
            "batch": ["B1", "B1"],
            "donor_id": ["P1", "P2"],
        }
    )

    _join_manifest_metadata(adata, manifest)

    assert list(adata.obs["condition"]) == ["Normal", "Normal", "LE", "LE", "LE"]
    assert list(adata.obs["batch"]) == ["B1"] * 5
    assert list(adata.obs["donor_id"]) == ["P1", "P1", "P2", "P2", "P2"]


def test_join_manifest_metadata_skips_absent_columns():
    adata = ad.concat([_corrected("s1", 2)], join="outer")
    # Manifest without a 'batch' column: no batch obs column is created.
    manifest = pd.DataFrame(
        {"sample_id": ["s1"], "cellranger_path": ["A/s1"], "condition": ["Normal"]}
    )

    _join_manifest_metadata(adata, manifest)

    assert "condition" in adata.obs.columns
    assert "batch" not in adata.obs.columns
