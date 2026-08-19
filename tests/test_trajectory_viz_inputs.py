import anndata as ad
import numpy as np
import pytest

from cellquorum.trajectory.viz import inputs


def _adata(n=20):
    a = ad.AnnData(np.zeros((n, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    return a


def test_resolve_basis_prefers_configured_then_umap_then_diffmap():
    a = _adata()
    a.obsm["X_diffmap"] = np.zeros((20, 2))
    assert inputs.resolve_basis(a, None) == "X_diffmap"
    a.obsm["X_umap"] = np.zeros((20, 2))
    assert inputs.resolve_basis(a, None) == "X_umap"
    assert inputs.resolve_basis(a, "X_diffmap") == "X_diffmap"
    assert inputs.resolve_basis(a, "X_missing") == "X_umap"  # falls back


def test_available_pseudotimes_sorted_and_filtered():
    a = _adata()
    a.obs["palantir_pseudotime"] = np.linspace(0, 1, 20)
    a.obs["dpt_pseudotime"] = np.linspace(0, 1, 20)
    assert inputs.available_pseudotimes(a, None) == ["dpt_pseudotime", "palantir_pseudotime"]
    assert inputs.available_pseudotimes(a, ["palantir_pseudotime"]) == ["palantir_pseudotime"]


def test_numeric_obs_retypes_on_non_numeric():
    a = _adata()
    a.obs["bad"] = ["x"] * 20
    with pytest.raises(inputs.VizInputError):
        inputs.numeric_obs(a, "bad")


def test_results_file_joins_under_trajectory(tmp_path):
    class Ctx:
        class paths:
            results = tmp_path

    p = inputs.results_file(Ctx, "cellrank", "fate_mapping.h5ad")
    assert p == tmp_path / "trajectory" / "cellrank" / "fate_mapping.h5ad"
