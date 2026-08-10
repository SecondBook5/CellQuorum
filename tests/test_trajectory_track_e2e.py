"""Executor-level end-to-end test for the trajectory track.

This drives the SAME path ``cq run`` uses — ``execute_pipeline_run`` — through
the ``trajectory`` producer stage (DPT) and the ``trajectory_viz`` figure stage,
on a small deterministic synthetic dataset. Every other stage is disabled so the
test isolates the trajectory track: it proves config → planner → executor →
trajectory → trajectory_viz wiring end to end, that the producer's obs output
lands on the final AnnData, and that figures reach disk.

DPT + matplotlib are always installed in this environment, so this test runs in
CI (unlike the skippable real-loom velocity test). It does not touch the optional
velocity/cellrank/palantir/cytotrace backends.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.backends.base import BaseBackend
from cellquorum.backends.registry import BackendRegistry
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.executor import PipelineExecutionResult
from cellquorum.core.pipeline import execute_pipeline_run


def build_test_backend_registry() -> BackendRegistry:
    """Build a deterministic registry holding one available Python backend."""

    registry = BackendRegistry()
    registry.register(BaseBackend(name="python", kind="python"))
    return registry


def make_trajectory_adata(n_cells: int = 60, n_genes: int = 40) -> ad.AnnData:
    """Build a small deterministic AnnData carrying a PCA + UMAP embedding.

    The object already holds everything the trajectory track needs so no upstream
    stage has to run: an ``X_pca`` rep for DPT's diffusion map, an ``X_umap``
    basis for the pseudotime figure, and a numeric ``stemness`` obs score whose
    argmax deterministically seeds the DPT root.
    """

    rng = np.random.default_rng(0)

    # Counts are unused by DPT (it works off the rep) but give the object an X.
    matrix = rng.poisson(1.0, size=(n_cells, n_genes)).astype("float32")

    # A structured PCA rep: a smooth gradient in the first component plus noise so
    # the diffusion map has a real axis of variation to root along.
    gradient = np.linspace(0.0, 1.0, n_cells)
    pca = rng.normal(0.0, 0.1, size=(n_cells, 20))
    pca[:, 0] = gradient
    umap = np.column_stack([gradient, rng.normal(0.0, 0.1, size=n_cells)])

    obs = pd.DataFrame(
        {"stemness": gradient.astype("float64")},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obsm["X_pca"] = pca.astype("float32")
    adata.obsm["X_umap"] = umap.astype("float32")
    return adata


def write_trajectory_h5ad(tmp_path: Path) -> Path:
    """Write the deterministic trajectory input h5ad and return its path."""

    h5ad_path = tmp_path / "trajectory_input.h5ad"
    make_trajectory_adata().write_h5ad(h5ad_path)
    return h5ad_path


def build_trajectory_config(h5ad_path: Path) -> CellQuorumConfig:
    """Build a config that runs ONLY trajectory (DPT) + trajectory_viz.

    Every other stage flag is disabled so the run isolates the trajectory track.
    The trajectory stage runs the single DPT method, rooted at the argmax of the
    ``stemness`` obs score, on the precomputed ``X_pca`` rep.
    """

    return CellQuorumConfig(
        project={"name": "trajectory_e2e"},
        input={"h5ad": str(h5ad_path)},
        compute={"backend": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
        r={"enabled": False},
        stages={
            # Disable everything upstream/adjacent; keep only the trajectory track.
            "ambient_correction": False,
            "qc": False,
            "preprocessing": False,
            "feature_selection": False,
            "dimensionality": False,
            "clustering": False,
            "integration": False,
            "annotation": False,
            "annotation_diagnostics": False,
            "annotation_consensus": False,
            "reference_mapping": False,
            "integration_benchmark": False,
            "integration_gate": False,
            "population_identity": False,
            "state_scoring": False,
            "discovery": False,
            "subclustering": False,
            "adjudication": False,
            "composition": False,
            "differential_expression": False,
            "differential_abundance": False,
            "enrichment": False,
            "enrichment_viz": False,
            "ccc_viz": False,
            "embeddings": False,
            "trajectory": True,
            "trajectory_viz": True,
            "molecular_inference": False,
            "cell_cell_communication": False,
            "network_analysis": False,
        },
        trajectory={
            "methods": [{"method": "dpt"}],
            "dpt": {
                "use_rep": "X_pca",
                "use_rep_fallback": ["X_pca"],
                "n_neighbors": 10,
                "n_comps": 10,
                "n_dcs": 5,
                "root_marker_score_key": "stemness",
            },
        },
    )


def test_trajectory_track_runs_end_to_end(tmp_path: Path) -> None:
    """execute_pipeline_run drives DPT + viz and lands obs + figures on disk."""

    h5ad_path = write_trajectory_h5ad(tmp_path)
    config = build_trajectory_config(h5ad_path)
    output_dir = tmp_path / "run"

    result = execute_pipeline_run(
        config,
        output_dir=output_dir,
        backend_registry=build_test_backend_registry(),
    )

    # The run produced an executor result via the real cq-run path.
    assert isinstance(result.execution_result, PipelineExecutionResult)
    execution = result.execution_result

    # The trajectory producer stage succeeded (DPT ran on X_pca).
    assert "trajectory" in execution.succeeded_stage_names()
    assert "trajectory" in execution.stage_results

    # The trajectory_viz stage did not fail (it succeeds or records a skip; a
    # crash would surface here).
    assert "trajectory_viz" not in execution.failed_stage_names()

    # The producer's pseudotime output survived onto the final AnnData.
    assert isinstance(result.context.adata, ad.AnnData)
    assert "dpt_pseudotime" in result.context.adata.obs
    pseudotime = np.asarray(result.context.adata.obs["dpt_pseudotime"], dtype="float64")
    assert np.isfinite(pseudotime).any()

    # DPT provenance was recorded under uns.
    assert result.context.adata.uns["trajectory"]["dpt"]["root_source"] == "marker_score"

    # The producer wrote its pseudotime h5ad artifact to the results tree.
    dpt_results = list((output_dir / "results" / "trajectory" / "dpt").glob("*.h5ad"))
    assert dpt_results, "no DPT results h5ad written"

    # The viz stage rendered pseudotime figures to the figures tree.
    figures = list((output_dir / "figures" / "trajectory").glob("pseudotime_*"))
    assert figures, "no trajectory pseudotime figures written"
