"""Freezes the documented public API surface so consolidation moves cannot
silently drop or rename an exported symbol."""

import importlib

import pytest

# Captured from Step 1 at baseline commit 11a81032e32d1445fecea9675aa63edeb952a3a9
EXPECTED = {
    "cellquorum": {
        "adjudication",
        "ambient_correction",
        "annotation",
        "annotation_consensus",
        "annotation_diagnostics",
        "annotations",
        "api",
        "backends",
        "ccc_network",
        "ccc_viz",
        "cell_cell_communication",
        "clustering",
        "coexpression",
        "compute",
        "config",
        "contracts",
        "core",
        "de_viz",
        "diag",
        "differential_abundance",
        "differential_expression",
        "dimensionality",
        "embeddings",
        "enrichment",
        "enrichment_viz",
        "evidence",
        "feature_selection",
        "grn",
        "integration",
        "integration_benchmark",
        "io",
        "methods",
        "perturbation",
        "population_identity",
        "pp",
        "preprocessing",
        "qc",
        "reference_mapping",
        "run_pipeline",
        "subclustering",
        "tl",
        "trajectory",
        "trajectory_viz",
        "version",
        "visualization",
    },
    "cellquorum.api": {
        "BackendRegistry",
        "CellQuorumConfig",
        "Path",
        "PipelineRunResult",
        "annotations",
        "bootstrap_pipeline_run",
        "bootstrap_pipeline_run_from_config_file",
        "execute_pipeline_run",
        "execute_pipeline_run_from_config_file",
        "run_pipeline",
        "validate_config_dict",
    },
    "cellquorum.pp": {
        "Any",
        "NotebookStageOutput",
        "TYPE_CHECKING",
        "annotations",
        "correct_ambient",
        "normalize",
        "qc",
        "run_stage",
        "select_features",
    },
    "cellquorum.tl": {
        "Any",
        "NotebookStageOutput",
        "TYPE_CHECKING",
        "adjudicate",
        "annotate",
        "annotations",
        "cluster",
        "integrate",
        "population_identity",
        "reduce_dimensions",
        "reference_map",
        "run_stage",
        "subcluster",
    },
}


@pytest.mark.parametrize("module_name", sorted(EXPECTED))
def test_public_surface_is_stable(module_name):
    module = importlib.import_module(module_name)
    exported = {name for name in dir(module) if not name.startswith("_")}
    assert exported == EXPECTED[module_name], (
        f"{module_name} public API changed:\n"
        f"  dropped: {sorted(EXPECTED[module_name] - exported)}\n"
        f"  added:   {sorted(exported - EXPECTED[module_name])}"
    )
