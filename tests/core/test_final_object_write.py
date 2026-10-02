"""The pipeline persists the final in-memory AnnData to the objects dir.

Regression: a from-scratch run threaded the annotated object through stages in
memory and never wrote it, leaving no deliverable on disk.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.context import PipelinePaths
from cellquorum.core.pipeline import _write_final_object


class _Ctx:
    def __init__(self, adata, paths):
        self.adata = adata
        self.paths = paths


def _paths(tmp_path):
    p = PipelinePaths.from_output_dir(tmp_path)
    p.ensure_directories()
    return p


def _adata():
    a = ad.AnnData(X=np.ones((5, 3), dtype=np.float32))
    a.obs["cell_type"] = ["T/NK"] * 5
    return a


def test_writes_final_object_by_default(tmp_path):
    cfg = CellQuorumConfig(project={"name": "t"})
    paths = _paths(tmp_path)
    out = _write_final_object(config=cfg, context=_Ctx(_adata(), paths))
    assert out is not None and out.is_file()
    assert out.name == "final_annotated.h5ad"
    # And it round-trips with the annotation intact.
    back = ad.read_h5ad(out)
    assert "cell_type" in back.obs.columns
    assert back.n_obs == 5


def test_respects_disable_flag(tmp_path):
    cfg = CellQuorumConfig(project={"name": "t"}, run={"write_final_object": False})
    paths = _paths(tmp_path)
    out = _write_final_object(config=cfg, context=_Ctx(_adata(), paths))
    assert out is None
    assert not (paths.objects / "final_annotated.h5ad").exists()


def test_custom_final_object_name(tmp_path):
    cfg = CellQuorumConfig(project={"name": "t"}, run={"final_object_name": "deliverable.h5ad"})
    paths = _paths(tmp_path)
    out = _write_final_object(config=cfg, context=_Ctx(_adata(), paths))
    assert out is not None and out.name == "deliverable.h5ad"


def test_none_adata_is_noop(tmp_path):
    cfg = CellQuorumConfig(project={"name": "t"})
    paths = _paths(tmp_path)
    out = _write_final_object(config=cfg, context=_Ctx(None, paths))
    assert out is None


def test_writes_object_with_slash_labels_in_uns(tmp_path):
    # Cell-type labels like "T/NK" land as dict KEYS in uns count-maps; h5py
    # forbids '/' in keys, so the write must sanitize them or it raises.
    cfg = CellQuorumConfig(project={"name": "t"})
    paths = _paths(tmp_path)
    a = _adata()
    a.obs["cell_type"] = ["T/NK", "T/NK", "Pericyte/SMC", "DC", "DC"]
    a.uns["cellquorum"] = {
        "annotation_consensus": {
            "label_counts": {"T/NK": 2, "Pericyte/SMC": 1, "DC": 2},
        }
    }
    out = _write_final_object(config=cfg, context=_Ctx(a, paths))
    assert out is not None and out.is_file()
    back = ad.read_h5ad(out)
    # Full object round-trips (genes intact, not a truncated partial write).
    assert back.shape == (5, 3)
    # Label values on obs are preserved (only uns dict keys were sanitized).
    assert set(back.obs["cell_type"]) == {"T/NK", "Pericyte/SMC", "DC"}
    # The uns key was sanitized so it could serialize.
    counts = back.uns["cellquorum"]["annotation_consensus"]["label_counts"]
    assert "T_NK" in counts and "Pericyte_SMC" in counts


def test_writes_object_with_unserializable_uns_payload(tmp_path):
    # Stages stash rich structures (list-of-dicts with ragged/mixed types) under
    # uns['cellquorum'][<stage>]; anndata can't write those. They must be
    # json-coerced so the object serializes, while simple entries stay intact.
    import json

    cfg = CellQuorumConfig(project={"name": "t"})
    paths = _paths(tmp_path)
    a = _adata()
    a.uns["cellquorum"] = {
        "adjudication": {
            "n_clusters": 2,  # friendly scalar — must stay a scalar
            "results": [
                {"cluster_id": "0", "reasons": ["a", "b"], "confidence": 0.9},
                {"cluster_id": "1", "vetoes": [{"name": "donor"}], "confidence": 0.3},
            ],
        }
    }
    out = _write_final_object(config=cfg, context=_Ctx(a, paths))
    assert out is not None and out.is_file()
    back = ad.read_h5ad(out)
    adj = back.uns["cellquorum"]["adjudication"]
    # Simple scalar preserved as-is.
    assert int(adj["n_clusters"]) == 2
    # The list-of-dicts was jsonified and is recoverable.
    restored = json.loads(adj["results"])
    assert restored[0]["cluster_id"] == "0"
    assert restored[1]["confidence"] == 0.3


def test_writes_object_with_slash_in_obs_columns_and_obsm(tmp_path):
    # scArches writes per-class probability columns like "refprob_Pericyte/SMC";
    # the '/' in the COLUMN NAME breaks write_h5ad. Sanitize column names + obsm
    # keys (values untouched).
    cfg = CellQuorumConfig(project={"name": "t"})
    paths = _paths(tmp_path)
    a = _adata()
    a.obs["refprob_Pericyte/SMC"] = np.linspace(0, 1, a.n_obs)
    a.obs["refprob_T/NK"] = np.linspace(1, 0, a.n_obs)
    a.obsm["X_scANVI/latent"] = np.zeros((a.n_obs, 2), dtype=np.float32)
    out = _write_final_object(config=cfg, context=_Ctx(a, paths))
    assert out is not None and out.is_file()
    back = ad.read_h5ad(out)
    assert back.shape == (5, 3)
    assert "refprob_Pericyte_SMC" in back.obs.columns
    assert "refprob_T_NK" in back.obs.columns
    assert "X_scANVI_latent" in back.obsm
    # A normal (no-slash) column is left exactly as-is.
    assert "cell_type" in back.obs.columns
