import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellquorum.stages.annotation.pbmc_refinement import CD4, CD8, NK, refine_pbmc


def test_mixed_cluster_preserves_baseline_and_explains_unresolved_rows():
    genes = ["CD3D", "TYROBP", "GNLY", "CD8A", "IL7R"]
    x = np.array(
        [[2, 0, 2, 1, 0], [0, 2, 2, 1, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 2, 0, 0, 0]],
        dtype=float,
    )
    a = ad.AnnData(sparse.csr_matrix(x), var=pd.DataFrame(index=genes))
    a.layers["log"] = a.X
    a.obs["cluster"] = "mixed"
    a.obs["vote"] = [NK, NK, NK, NK, "B cells"]
    a.obsp["connectivities"] = sparse.diags([1, 1, 0, 1, 1], dtype=float, format="csr")
    cells = refine_pbmc(a, "log", "vote", "cluster")
    assert list(cells["call"]) == [CD8, NK, "Unassigned", CD4, "B cells"]
    assert list(a.obs["vote_cluster_vote"]) == [NK, NK, NK, NK, "B cells"]
    assert cells.loc[2, "reason"] == "no connected RNA neighbors"
    assert list(cells["barcode"]) == list(a.obs_names)
    # Without a helper marker, an unavailable assay cannot masquerade as zero expression.
    with pytest.raises(ValueError, match="helper"):
        refine_pbmc(a[:, :-1].copy(), "log", "vote", "cluster")
