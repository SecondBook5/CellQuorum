"""Run a portable analysis and checkpoint continuation from the installed wheel."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import cellquorum
from cellquorum import run_pipeline
from cellquorum.backends.base import BaseBackend
from cellquorum.backends.gpu import GPUBackend
from cellquorum.backends.rapids import RapidsBackend
from cellquorum.backends.registry import BackendRegistry
from cellquorum.config.models import StageSelectionConfig
from cellquorum.core.checkpoint import CheckpointError

pytestmark = pytest.mark.integration
BACKEND = os.environ.get("CELLQUORUM_ACCEPTANCE_BACKEND", "cpu")
if BACKEND not in {"cpu", "gpu"}:
    raise pytest.UsageError(
        f"CELLQUORUM_ACCEPTANCE_BACKEND must be 'cpu' or 'gpu'; received {BACKEND!r}"
    )


def registry() -> BackendRegistry:
    backends = BackendRegistry()
    backends.register(BaseBackend(name="python", kind="python"))
    if BACKEND == "gpu":
        backends.register(GPUBackend())
        backends.register(RapidsBackend())
    return backends


def config_for(source: Path) -> dict:
    enabled = {
        "qc",
        "preprocessing",
        "dimensionality",
        "clustering",
        "population_identity",
        "embeddings",
    }
    return {
        "project": {"name": "installed_acceptance"},
        "input": {"h5ad": str(source), "counts_layer": "counts"},
        "compute": {"backend": "cpu", "prefer_gpu": False},
        "r": {"enabled": False},
        "run": {"random_seed": 7, "verbose": False, "checkpoint": True, "checkpoint_after": ["qc"]},
        "cohort": {
            "sample_key": "sample",
            "donor_key": "donor",
            "condition_key": "condition",
            "condition_levels": ["control", "treated"],
        },
        "design": {
            "donor_col": "donor",
            "condition_col": "condition",
            "case": "treated",
            "control": "control",
            "paired": True,
        },
        "stages": {name: name in enabled for name in StageSelectionConfig.model_fields},
        "qc": {
            "metrics": {"layer": "counts", "percent_top": [20]},
            "floors": {"min_genes_per_cell": 20, "min_cells_per_gene": 1},
            "doublets": {"enabled": False},
            "graded": {"archetype_audit": False},
            "outputs": {"write_figures": False},
        },
        "preprocessing": {
            "normalization": {"input_layer": "counts", "recipe": "cellquorum_log1p_cp10k_v1"}
        },
        "dimensionality": {"input_layer": "cellquorum_normalized", "n_pcs": 5, "max_pcs": 5},
        "clustering": {"method": "leiden", "use_rep": "X_pca"},
        "population_identity": {"cluster_key": "leiden", "write_figures": False},
        "embeddings": {
            "use_rep": "X_pca",
            "embeddings": ["umap"],
            "color_by": ["leiden"],
            "figure_formats": ["png", "pdf"],
            "dpi": 90,
        },
        "report": {"enabled": True, "html": True, "markdown": True, "fail_on_report_error": True},
    }


def fixture_file(path: Path) -> None:
    rng = np.random.default_rng(71)
    counts = rng.poisson(3, size=(120, 80)).astype(np.float32)
    counts[:, :2] = rng.poisson(0.15, size=(120, 2))
    obs = pd.DataFrame(
        {
            "sample": np.repeat([f"s{i}" for i in range(6)], 20),
            "donor": np.repeat(["d1", "d1", "d2", "d2", "d3", "d3"], 20),
            "condition": np.tile(np.repeat(["control", "treated"], 20), 3),
        },
        index=[f"cell{i}" for i in range(120)],
    )
    data = ad.AnnData(
        sparse.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=["MT-CO1", "MT-ND1"] + [f"G{i}" for i in range(78)]),
    )
    data.layers["counts"] = data.X.copy()
    data.write_h5ad(path)


def test_installed_analysis_checkpoint_and_outputs(tmp_path: Path) -> None:
    if os.environ.get("CELLQUORUM_REQUIRE_WHEEL"):
        assert "site-packages" in Path(cellquorum.__file__).parts
    source = tmp_path / "input.h5ad"
    fixture_file(source)
    config = config_for(source)
    gpu = BACKEND == "gpu"
    if gpu:
        from cellquorum.backends.compute import gpu_compute_available

        assert gpu_compute_available(), "GPU acceptance requires working CuPy, RAPIDS, and CUDA."
        config["compute"].update(backend="gpu", prefer_gpu=True, fallback_to_cpu=False)
        config["dimensionality"]["precision"] = "float64"
    complete_dir, resumed_dir = tmp_path / "complete", tmp_path / "resumed"
    complete = run_pipeline(config, output_dir=complete_dir, backend_registry=registry())
    assert complete.execution_result is not None
    assert not complete.execution_result.has_failures()
    expected = {name for name, enabled in config["stages"].items() if enabled}
    assert expected <= set(complete.execution_result.succeeded_stage_names())
    if gpu:
        records = {
            record.stage_name: record
            for record in complete.execution_result.stage_execution_records
        }
        for name in ("dimensionality", "clustering"):
            assert records[name].metrics["compute"] == "gpu"
    assert (complete_dir / "reports" / "report.html").is_file()
    assert (complete_dir / "reports" / "methods.txt").is_file()
    assert (complete_dir / "provenance" / "stage_execution_records.json").is_file()
    pngs = list(complete_dir.rglob("*.png"))
    pdfs = list(complete_dir.rglob("*.pdf"))
    assert pngs and pdfs
    assert all(path.stat().st_size > 100 for path in pngs + pdfs)
    comparisons = list(complete_dir.rglob("qc_donor_comparisons.csv"))
    assert comparisons and pd.read_csv(comparisons[0]).n_group1.eq(3).all()
    final_path = complete_dir / "objects" / "final_annotated.h5ad"
    final = ad.read_h5ad(final_path)
    assert {"counts", "cellquorum_normalized"} <= set(final.layers)
    assert {"X_pca", "X_umap"} <= set(final.obsm)
    assert "leiden" in final.obs
    assert np.isfinite(final.obsm["X_pca"]).all()
    assert np.isfinite(final.obsm["X_umap"]).all()
    original = ad.read_h5ad(source)[final.obs_names, final.var_names]
    assert (final.layers["counts"] != original.layers["counts"]).nnz == 0
    if gpu:
        assert final.uns["pca"]["params"]["arithmetic_dtype"] == "float64"

    stopped = run_pipeline(
        config, output_dir=resumed_dir, backend_registry=registry(), until_stage="qc"
    )
    assert stopped.execution_result.succeeded_stage_names() == ["qc"]
    resumed = run_pipeline(
        config, output_dir=resumed_dir, backend_registry=registry(), from_stage="preprocessing"
    )
    assert not resumed.execution_result.has_failures()
    assert (resumed.context.adata.layers["counts"] != final.layers["counts"]).nnz == 0
    np.testing.assert_allclose(
        final.obsm["X_pca"], resumed.context.adata.obsm["X_pca"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_equal(final.obs.leiden, resumed.context.adata.obs.leiden)

    changed = copy.deepcopy(config)
    changed["qc"]["floors"]["min_genes_per_cell"] = 25
    with pytest.raises(CheckpointError, match="(?i)fingerprint|settings|config.*chang|stale"):
        run_pipeline(
            changed, output_dir=resumed_dir, backend_registry=registry(), from_stage="preprocessing"
        )
