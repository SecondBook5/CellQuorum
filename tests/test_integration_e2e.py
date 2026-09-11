"""integration (Harmony) actually runs inside a real pipeline, alongside its neighbors.

Every other integration test exercises HarmonyMethod/ScVIMethod/ScANVIMethod in
isolation, with hand-built configs/contexts. integration was explicitly disabled in
both existing full-pipeline smoke tests (test_generalization_smoke.py,
test_feature_selection_e2e.py), so nothing had ever run it through the real executor,
alongside preprocessing/dimensionality/clustering, the way a user's config actually
would.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum import run_pipeline
from cellquorum.backends.base import BaseBackend
from cellquorum.backends.registry import BackendRegistry


def _cpu_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(BaseBackend(name="python", kind="python"))
    return registry


def _pbmc_adata(n: int = 80, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    genes = ["MT-CO1", "MT-ND1", "RPS3", "RPL13"] + [f"G{i}" for i in range(46)]
    x = rng.poisson(2.0, size=(n, len(genes))).astype(np.float32)
    donor = np.where(np.arange(n) % 2 == 0, "D1", "D2")
    stim = np.where((np.arange(n) // 2) % 2 == 0, "control", "stimulated")
    batch = np.where((np.arange(n) // 4) % 2 == 0, "b1", "b2")
    obs = pd.DataFrame(
        {
            "sample_id": [f"{d}_{s}" for d, s in zip(donor, stim, strict=True)],
            "donor_id": donor,
            "stim": stim,
            "batch": batch,
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = x.copy()
    return adata


def _config_with_integration(h5ad_path: Path) -> dict:
    return {
        "project": {"name": "integration_e2e"},
        "input": {"h5ad": str(h5ad_path), "counts_layer": "counts"},
        "run": {"random_seed": 7, "verbose": False},
        "compute": {"backend": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
        "r": {"enabled": False},
        "cohort": {
            "sample_key": "sample_id",
            "donor_key": "donor_id",
            "condition_key": "stim",
            "batch_key": "batch",
            "condition_levels": ["control", "stimulated"],
        },
        "stages": {
            "qc": True,
            "preprocessing": True,
            "feature_selection": False,
            "dimensionality": True,
            "integration": True,
            "clustering": True,
            "population_identity": True,
            "reference_mapping": False,
            "annotation_diagnostics": False,
            "integration_benchmark": False,
            "adjudication": False,
        },
        "qc": {
            "metrics": {"layer": "counts", "percent_top": [2]},
            "floors": {"min_genes_per_cell": 5, "min_cells_per_gene": 1},
            "outputs": {"write_h5ad": False, "write_figures": False},
        },
        "preprocessing": {
            "normalization": {
                "output_layer": "cellquorum_normalized",
                "recipe": "cellquorum_log1p_cp10k_v1",
            },
            "write_figures": False,
        },
        "dimensionality": {"input_layer": "cellquorum_normalized", "n_pcs": 5, "max_pcs": 5},
        "integration": {
            "method": "harmony",
            "batch_key": "batch",
            "input_rep": "X_pca",
            "output_rep": "X_pca_harmony",
            "max_iter_harmony": 5,
        },
        "clustering": {"method": "leiden", "use_rep": "X_pca_harmony"},
        "population_identity": {"cluster_key": "leiden", "write_figures": False},
        "report": {"enabled": True, "html": True, "markdown": True},
    }


def test_integration_runs_end_to_end_and_writes_the_corrected_embedding(tmp_path: Path) -> None:
    h5ad_path = tmp_path / "pbmc.h5ad"
    _pbmc_adata().write_h5ad(h5ad_path)
    output_dir = tmp_path / "run"

    result = run_pipeline(
        _config_with_integration(h5ad_path),
        output_dir=output_dir,
        backend_registry=_cpu_registry(),
        execute=True,
    )

    execution = result.execution_result
    assert execution is not None
    assert not execution.has_failures()

    succeeded = set(execution.succeeded_stage_names())
    assert "integration" in succeeded
    assert "dimensionality" in succeeded
    assert "clustering" in succeeded

    final_adata = execution.context.require_adata()
    assert "X_pca_harmony" in final_adata.obsm
    assert final_adata.obsm["X_pca_harmony"].shape[0] == final_adata.n_obs
    # Clustering was configured to read the corrected embedding, not raw PCA.
    assert "leiden" in final_adata.obs.columns
