"""Guard the controller/worker exchange without requiring a GPU in CPU CI."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from cellquorum.backends import rapids_worker


def test_pca_worker_uses_named_layer_and_preserves_cell_and_gene_identity(monkeypatch):
    data = ad.AnnData(np.zeros((4, 3), dtype="float32"))
    data.layers["normalized"] = np.arange(12, dtype="float32").reshape(4, 3)
    data.var["highly_variable"] = [True, False, True]
    data.obs_names = ["lib2:d", "lib1:b", "lib1:a", "lib2:c"]
    shuffled = False

    def invoke(python, operation, directory, raw_options):
        assert python == "/worker/python" and operation == "pca"
        options = json.loads(raw_options)
        transfer = ad.read_h5ad(Path(directory) / "input.h5ad")
        assert transfer.X is None
        np.testing.assert_array_equal(transfer.layers[options["layer"]], data.layers["normalized"])
        assert transfer.var[options["mask_var"]].tolist() == [True, False, True]
        transfer.obsm["X_pca"] = np.ones((4, 1))
        transfer.varm["PCs"] = np.ones((3, 1))
        transfer.uns["pca"] = {"variance": [1.0]}
        if shuffled:
            transfer = transfer[::-1].copy()
        transfer.write_h5ad(Path(directory) / "output.h5ad")

    monkeypatch.setenv("CELLQUORUM_RAPIDS_PYTHON", "/worker/python")
    monkeypatch.setattr(rapids_worker, "_invoke", invoke)
    opts = {"layer": "normalized", "mask_var": "highly_variable", "n_comps": 1, "random_state": 1}
    rapids_worker.run_rapids(data, "pca", **opts)
    assert "PCs" in data.varm and "pca" in data.uns and "X_pca" in data.obsm
    np.testing.assert_array_equal(data.X, np.zeros((4, 3)))
    shuffled = True
    with pytest.raises(ValueError, match="identifiers"):
        rapids_worker.run_rapids(data, "pca", **opts)
