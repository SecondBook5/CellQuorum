"""CellRank runtime repairs: stranded pseudotime rows and the Lineage h5ad round trip."""

import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellquorum.core.h5ad_io import write_h5ad
from cellquorum.stages.trajectory._cellrank import _pseudotime_kernel


def test_stranded_terminal_row_uses_native_soft_scheme():
    a = ad.AnnData(np.ones((4, 2)))
    a.obs["pt"] = [0.0, 0.3, 0.6, 1.0]
    graph = sp.csr_matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
    a.obsp["connectivities"] = graph
    a.obsp["distances"] = graph.copy()
    a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances"}
    notes = []
    k = _pseudotime_kernel(a, "pt", notes)
    np.testing.assert_allclose(np.asarray(k.transition_matrix.sum(axis=1)).ravel(), 1.0)
    assert notes and "soft threshold" in notes[0]
    assert ((k.transition_matrix != 0).multiply(graph == 0)).nnz == 0


def test_lineage_roundtrip_preserves_values_and_metadata(tmp_path):
    from cellrank._utils._lineage import Lineage

    a = ad.AnnData(np.ones((3, 2)))
    vals = np.array([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
    a.obsm["fate"] = Lineage(vals, names=["early", "late"], colors=["#123456", "#abcdef"])
    path = tmp_path / "fate.h5ad"
    notes = write_h5ad(a, path)
    b = ad.read_h5ad(path)
    np.testing.assert_array_equal(b.obsm["fate"], vals)
    assert list(b.uns["cellrank_lineage_array_metadata"]["obsm:fate"]["names"]) == ["early", "late"]
    assert any("Lineage" in n for n in notes)
