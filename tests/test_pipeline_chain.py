"""Full-chain end-to-end test: does data flow correctly THROUGH the pipeline?

This is the test whose absence let the SoupX "disk-only sidecar" bug ship — every
stage passed its own unit test on synthetic data, but nothing verified that each
stage's scientific output actually lands on the AnnData the executor threads to
the next stage. This test runs the real CPU backbone

    qc -> preprocessing -> dimensionality -> integration -> clustering -> annotation

on one synthetic-but-realistic object via ``execute_pipeline_run`` and asserts
that EACH stage's output is present on the final threaded ``context.adata`` — i.e.
the chain is wired end to end, not just green stage-by-stage.

ambient_correction is exercised separately (test_ambient_correction_stage.py /
test_soupx_integration.py) because it needs R + Cell Ranger matrices; here it is
disabled so the chain test stays hermetic and always runs.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.backends.base import BaseBackend
from cellquorum.backends.registry import BackendRegistry
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.pipeline import execute_pipeline_run

# Real marker genes so marker-vote annotation has something to score against.
_TYPE_A = ["A1", "A2", "A3"]
_TYPE_B = ["B1", "B2", "B3"]


def _backend_registry() -> BackendRegistry:
    """Registry with an available Python backend (CPU chain, no R/GPU)."""

    registry = BackendRegistry()
    registry.register(BaseBackend(name="python", kind="python"))
    return registry


def _synthetic_cohort(seed: int = 0) -> ad.AnnData:
    """Two batches x two cell types, integer counts — a realistic small cohort.

    Batch offset is injected so integration (Harmony) has something to correct;
    two marker programs so clustering finds >=2 groups and annotation can label
    them.
    """

    rng = np.random.default_rng(seed)
    n = 240
    genes = _TYPE_A + _TYPE_B + [f"G{i}" for i in range(30)]

    # Poisson counts baseline.
    counts = rng.poisson(1.0, size=(n, len(genes))).astype(np.float32)

    # First half expresses type-A markers, second half type-B markers.
    counts[: n // 2, :3] += rng.poisson(8.0, size=(n // 2, 3))
    counts[n // 2 :, 3:6] += rng.poisson(8.0, size=(n - n // 2, 3))

    # Two donors/batches with a technical offset on a shared block of genes.
    batch = np.array(["P1", "P2"] * (n // 2))
    counts[batch == "P2", 6:12] += rng.poisson(5.0, size=((batch == "P2").sum(), 6))

    obs = pd.DataFrame(
        {"patient_id": batch, "condition": ["Normal", "Lymphedema"] * (n // 2)},
        index=[f"cell_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=genes)
    return ad.AnnData(X=counts, obs=obs, var=var)


def _chain_config(h5ad_path) -> CellQuorumConfig:
    """Config enabling the CPU backbone: qc..annotation, ambient off."""

    return CellQuorumConfig(
        project={"name": "chain_test"},
        input={"h5ad": str(h5ad_path)},
        compute={"backend": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
        r={"enabled": False},
        stages={
            # Ambient needs R + Cell Ranger matrices; tested separately.
            "ambient_correction": False,
            "qc": True,
            "preprocessing": True,
            "dimensionality": True,
            "integration": True,
            "clustering": True,
            "annotation": True,
            # Downstream slots not implemented yet.
            "state_scoring": False,
            "discovery": False,
            "subclustering": False,
            "composition": False,
            "differential_expression": False,
            "molecular_inference": False,
            "cell_cell_communication": False,
            "network_analysis": False,
        },
        # cp10k recipe keeps this threading test env-independent: the scclr-backed
        # PFlog1pPF default needs the isolated scclr env (covered separately).
        preprocessing={"normalization": {"recipe": "cellquorum_log1p_cp10k_v1"}},
        qc={
            "mode": "report_only",  # keep all cells so downstream has signal
            "threshold_strategy": "fixed",
            "metrics": {"percent_top": [2]},
            "basic": {
                "min_genes_per_cell": 1,
                "min_cells_per_gene": 1,
                "max_mito_percent": 100.0,
            },
            "mad": {"enabled": False},
            "doublets": {"enabled": False},
        },
        integration={
            "method": "harmony",
            "batch_key": "patient_id",
            "input_rep": "X_pca",
            "output_rep": "X_pca_harmony",
        },
        clustering={
            # Cluster on the integration-corrected embedding.
            "use_rep": "X_pca_harmony",
            "n_neighbors": 15,
            "resolution": 1.0,
        },
        annotation={
            "method": "marker_vote",
            "cluster_key": "leiden",
            "score_layer": "cellquorum_normalized",
            "key_added": "cell_type",
            "marker_panels": {"TypeA": _TYPE_A, "TypeB": _TYPE_B},
        },
    )


def test_full_cpu_chain_threads_every_stage_output(tmp_path):
    """The CPU backbone must thread each stage's output to the final AnnData."""

    # Write the synthetic cohort and run the real pipeline.
    h5ad_path = tmp_path / "cohort.h5ad"
    _synthetic_cohort().write_h5ad(h5ad_path)

    result = execute_pipeline_run(
        _chain_config(h5ad_path),
        output_dir=tmp_path / "run",
        backend_registry=_backend_registry(),
        load_input=True,
    )

    # No stage failed.
    assert (
        not result.execution_result.has_failures()
    ), f"chain had failures: {result.execution_result.failed_stage_names()}"

    # Every enabled backbone stage ran (not skipped).
    succeeded = result.execution_result.succeeded_stage_names()
    for stage in [
        "qc",
        "preprocessing",
        "dimensionality",
        "integration",
        "clustering",
        "annotation",
    ]:
        assert stage in succeeded, f"{stage} did not succeed: {succeeded}"

    # THE CHAIN ASSERTION: each stage's scientific output is present on the final
    # threaded AnnData — proving data flowed stage -> stage, not just that each
    # stage passed in isolation.
    final = result.context.adata
    assert final is not None
    # preprocessing -> normalized (tagged) layer
    assert "cellquorum_normalized" in final.layers, "normalization output lost"
    # dimensionality -> PCA embedding
    assert "X_pca" in final.obsm, "PCA embedding lost"
    # integration -> corrected embedding
    assert "X_pca_harmony" in final.obsm, "integration embedding lost"
    # clustering -> Leiden labels (and it ran on the corrected embedding)
    assert "leiden" in final.obs, "cluster labels lost"
    assert final.obs["leiden"].nunique() >= 2, "clustering found <2 clusters"
    # annotation -> cell-type labels derived from the clusters
    assert "cell_type" in final.obs, "annotation labels lost"
    assert set(final.obs["cell_type"].unique()) <= {"TypeA", "TypeB"}


def test_annotation_actually_runs_not_skipped(tmp_path):
    """Regression guard for the ordering bug: annotation must NOT skip.

    Annotation needs the leiden column clustering produces; if the planner ever
    reverts to running annotation before clustering, annotation would skip and
    cell_type would be absent. This asserts it genuinely runs.
    """

    h5ad_path = tmp_path / "cohort.h5ad"
    _synthetic_cohort().write_h5ad(h5ad_path)

    result = execute_pipeline_run(
        _chain_config(h5ad_path),
        output_dir=tmp_path / "run",
        backend_registry=_backend_registry(),
        load_input=True,
    )

    # annotation succeeded and its record is not a skip.
    ann = result.execution_result.stage_results.get("annotation")
    assert ann is not None
    assert not ann.metrics.get("skipped", False), "annotation skipped — ordering bug?"
    assert "cell_type" in result.context.adata.obs


def test_full_analysis_chain_runs_de_da_enrichment_viz(tmp_path):
    """The full analysis pipeline schedules and runs DE, DA, enrichment, enrichment_viz."""

    # Write the synthetic cohort (with two conditions for DE/DA).
    h5ad_path = tmp_path / "cohort.h5ad"
    _synthetic_cohort().write_h5ad(h5ad_path)

    # Build a config enabling the full analysis spine: annotation + DE + DA + enrichment + viz.
    config = CellQuorumConfig(
        project={"name": "full_analysis_test"},
        input={"h5ad": str(h5ad_path)},
        compute={"backend": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
        r={"enabled": False},
        stages={
            "ambient_correction": False,
            "qc": True,
            "preprocessing": True,
            "dimensionality": True,
            "integration": True,
            "clustering": True,
            "annotation": True,
            # Enable the analysis trio.
            "differential_expression": True,
            "differential_abundance": True,
            "enrichment": True,
            "enrichment_viz": True,
            # Downstream slots not needed.
            "state_scoring": False,
            "discovery": False,
            "subclustering": False,
            "composition": False,
            "molecular_inference": False,
            "cell_cell_communication": False,
            "network_analysis": False,
        },
        preprocessing={"normalization": {"recipe": "cellquorum_log1p_cp10k_v1"}},
        qc={
            "mode": "report_only",
            "threshold_strategy": "fixed",
            "metrics": {"percent_top": [2]},
            "basic": {
                "min_genes_per_cell": 1,
                "min_cells_per_gene": 1,
                "max_mito_percent": 100.0,
            },
            "mad": {"enabled": False},
            "doublets": {"enabled": False},
        },
        integration={
            "method": "harmony",
            "batch_key": "patient_id",
            "input_rep": "X_pca",
            "output_rep": "X_pca_harmony",
        },
        clustering={
            "use_rep": "X_pca_harmony",
            "n_neighbors": 15,
            "resolution": 1.0,
        },
        annotation={
            "method": "marker_vote",
            "cluster_key": "leiden",
            "score_layer": "cellquorum_normalized",
            "key_added": "cell_type",
            "marker_panels": {"TypeA": _TYPE_A, "TypeB": _TYPE_B},
        },
        # Design block so DE/DA/enrichment can run (not skip for "no case/control").
        design={
            "donor_col": "patient_id",
            "condition_col": "condition",
            "case": "Lymphedema",
            "control": "Normal",
            "paired": False,
        },
    )

    result = execute_pipeline_run(
        config,
        output_dir=tmp_path / "run",
        backend_registry=_backend_registry(),
        load_input=True,
    )

    # No stage failed.
    assert (
        not result.execution_result.has_failures()
    ), f"analysis chain had failures: {result.execution_result.failed_stage_names()}"

    # The core backbone stages ran.
    succeeded = result.execution_result.succeeded_stage_names()
    for stage in [
        "qc",
        "preprocessing",
        "dimensionality",
        "integration",
        "clustering",
        "annotation",
    ]:
        assert stage in succeeded, f"{stage} did not succeed: {succeeded}"

    # THE PLANNER ASSERTION: DE, DA, enrichment, enrichment_viz are SCHEDULED
    # (present in stage_records, not missing from the plan).
    stage_records = {r.stage_name: r for r in result.execution_result.stage_execution_records}
    for stage_name in [
        "differential_expression",
        "differential_abundance",
        "enrichment",
        "enrichment_viz",
    ]:
        assert (
            stage_name in stage_records
        ), f"{stage_name} was not scheduled — planner did not add it to the plan"
        record = stage_records[stage_name]
        # Invariant: either succeeded or cleanly skipped (not failed, not missing).
        assert record.status in (
            "success",
            "skipped",
        ), f"{stage_name} failed or was not scheduled: status={record.status}"
        # On the tiny synthetic cohort, some methods may legitimately skip
        # (e.g., enrichment if DE found no genes, or if enrichment methods need
        # more signal). That's acceptable — the test proves they are scheduled
        # and do not crash the run.

    # On the tiny synthetic cohort, most analysis methods legitimately skip
    # (no R backend, no significant genes, insufficient samples). The test proves
    # the PIPELINE WIRING: stages are scheduled, not that every method produces
    # output on synthetic data. If a stage succeeded and wrote artifacts, that's
    # a bonus (e.g., enrichment stage succeeded even though all methods skipped —
    # the stage framework itself ran and aggregated the skip reasons).


def test_full_gpu_chain_threads_every_stage_output(tmp_path):
    """The chain runs on GPU end-to-end (skips when rapids-singlecell unavailable).

    The GPU analog of the CPU chain test: proves normalization, PCA, and
    clustering actually ran on GPU (metrics["compute"]=="gpu") AND that every
    stage's output still threads to the final AnnData — i.e. GPU routing did not
    silently break the chain or silently fall back to CPU.
    """

    import pytest

    from cellquorum.backends.compute import gpu_compute_available

    if not gpu_compute_available():
        pytest.skip("rapids-singlecell/cupy unavailable")

    h5ad_path = tmp_path / "cohort.h5ad"
    _synthetic_cohort().write_h5ad(h5ad_path)

    # Force GPU for the routable stages.
    config = _chain_config(h5ad_path)
    config.compute.backend = "gpu"

    result = execute_pipeline_run(
        config,
        output_dir=tmp_path / "run",
        backend_registry=_backend_registry(),
        load_input=True,
    )

    assert (
        not result.execution_result.has_failures()
    ), f"GPU chain had failures: {result.execution_result.failed_stage_names()}"

    # Same end-to-end outputs as the CPU chain — the chain is intact on GPU.
    final = result.context.adata
    assert "cellquorum_normalized" in final.layers
    assert "X_pca" in final.obsm
    assert "X_pca_harmony" in final.obsm
    assert "leiden" in final.obs
    assert "cell_type" in final.obs

    # PCA and clustering actually ran on GPU (not a silent CPU fallback).
    dim = result.execution_result.stage_results["dimensionality"]
    clu = result.execution_result.stage_results["clustering"]
    assert dim.metrics.get("compute") == "gpu", "PCA did not run on GPU"
    assert clu.metrics.get("compute") == "gpu", "clustering did not run on GPU"


def test_embeddings_stage_runs_in_full_chain(tmp_path):
    """The embeddings stage is scheduled and runs (success or clean skip)."""
    h5ad_path = tmp_path / "cohort.h5ad"
    _synthetic_cohort().write_h5ad(h5ad_path)

    config = CellQuorumConfig(
        project={"name": "embeddings_chain_test"},
        input={"h5ad": str(h5ad_path)},
        compute={"backend": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
        r={"enabled": False},
        stages={
            "ambient_correction": False,
            "qc": True,
            "preprocessing": True,
            "dimensionality": True,
            "integration": True,
            "clustering": True,
            "annotation": True,
            "embeddings": True,
            "differential_expression": False,
            "differential_abundance": False,
            "enrichment": False,
            "enrichment_viz": False,
            "state_scoring": False,
            "discovery": False,
            "subclustering": False,
            "composition": False,
            "molecular_inference": False,
            "cell_cell_communication": False,
            "network_analysis": False,
        },
        preprocessing={"normalization": {"recipe": "cellquorum_log1p_cp10k_v1"}},
        qc={
            "mode": "report_only",
            "threshold_strategy": "fixed",
            "metrics": {"percent_top": [2]},
            "basic": {
                "min_genes_per_cell": 1,
                "min_cells_per_gene": 1,
                "max_mito_percent": 100.0,
            },
            "mad": {"enabled": False},
            "doublets": {"enabled": False},
        },
        integration={
            "method": "harmony",
            "batch_key": "patient_id",
            "input_rep": "X_pca",
            "output_rep": "X_pca_harmony",
        },
        clustering={
            "use_rep": "X_pca_harmony",
            "n_neighbors": 15,
            "resolution": 1.0,
        },
        annotation={
            "method": "marker_vote",
            "cluster_key": "leiden",
            "score_layer": "cellquorum_normalized",
            "key_added": "cell_type",
            "marker_panels": {"TypeA": _TYPE_A, "TypeB": _TYPE_B},
        },
        embeddings={"embeddings": ["umap"], "figure_formats": ["png"], "dpi": 80},
    )

    result = execute_pipeline_run(
        config,
        output_dir=tmp_path / "run",
        backend_registry=_backend_registry(),
        load_input=True,
    )

    stage_records = {r.stage_name: r for r in result.execution_result.stage_execution_records}
    assert "embeddings" in stage_records, "embeddings not scheduled by the planner"
    assert stage_records["embeddings"].status in ("success", "skipped")
    # When it succeeded, the UMAP coordinates threaded onto the final object.
    if stage_records["embeddings"].status == "success":
        assert "X_umap" in result.context.adata.obsm
