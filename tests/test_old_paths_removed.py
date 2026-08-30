"""Freezes the clean break: every retired re-export shim path is gone for good.

Importing any of these must raise ModuleNotFoundError — there is exactly one
canonical import path per public thing. Task 3 extends this with the step
packages that moved under `cellquorum.stages.*`.
"""

import importlib

import pytest

# The 12 re-export shims deleted in the clean break (Deliverable 1).
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


@pytest.mark.parametrize("path", REMOVED_SHIM_PATHS)
def test_removed_shim_path_is_gone(path):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)


# The 12 pipeline-step packages moved under cellquorum.stages.* (Deliverable 3);
# their old top-level paths no longer import.
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


@pytest.mark.parametrize("path", MOVED_STEP_PACKAGES)
def test_moved_step_package_old_path_is_gone(path):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)


@pytest.mark.parametrize("path", MOVED_STEP_PACKAGES)
def test_moved_step_package_resolves_under_stages(path):
    stages_path = path.replace("cellquorum.", "cellquorum.stages.", 1)
    assert importlib.import_module(stages_path) is not None
