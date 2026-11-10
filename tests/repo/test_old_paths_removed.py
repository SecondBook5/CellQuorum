"""Retired imports stay unavailable and moved stages resolve at their public paths."""

import importlib

import pytest

# Retired re-export shims.
REMOVED_SHIM_PATHS = [
    "cellquorum.differential_expression",
    "cellquorum.differential_abundance",
    "cellquorum.enrichment",
    "cellquorum.multicellular_programs",
    "cellquorum.tl",
    "cellquorum.pp",
    "cellquorum.diag",
    "cellquorum.evidence",
    "cellquorum._notebook",
    "cellquorum.qc.ambient",
    "cellquorum.qc.publication",
    "cellquorum.qc.visualization",
]


# Packages that moved under cellquorum.stages.
MOVED_STEP_PACKAGES = [
    "cellquorum.ambient_correction",
    "cellquorum.qc",
    "cellquorum.preprocessing",
    "cellquorum.integration",
    "cellquorum.clustering",
    "cellquorum.annotation",
    "cellquorum.comparative",
    "cellquorum.cell_cell_communication",
    "cellquorum.gene_regulation",
    "cellquorum.trajectory",
    "cellquorum.state_scoring",
    "cellquorum.discovery",
]


@pytest.mark.parametrize("path", REMOVED_SHIM_PATHS + MOVED_STEP_PACKAGES)
def test_retired_import_path(path):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)
    if path in MOVED_STEP_PACKAGES:
        stages_path = path.replace("cellquorum.", "cellquorum.stages.", 1)
        assert importlib.import_module(stages_path) is not None
