"""Every bundled R script a stage references must actually resolve.

Regression guard for a real outage: stage modules used to locate bundled R scripts
with a relative ``Path(__file__).parent.parent...`` chain, which encodes the
calling module's nesting depth. Regrouping stages into category packages added a
level, so all eleven constants silently pointed at a nonexistent
``stages/backends/r_scripts`` directory. Nothing failed at import time — each
method failed only when a pipeline actually reached that stage, one at a time.

These tests fail loudly at collection time instead, and they assert the *file
exists*, not merely that a path was constructed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from cellquorum.backends.script_paths import (
    R_SCRIPTS_DIR,
    available_r_scripts,
    r_script_exists,
    r_script_path,
)

# (module path, constant name) for every stage-side reference to a bundled R script.
# Add a row whenever a stage gains an R backend.
R_SCRIPT_REFERENCES = [
    ("cellquorum.stages.clustering.subclustering.partition", "_CHOIR_R"),
    ("cellquorum.stages.clustering.subclustering.partition", "_SCSHC_TEST_R"),
    ("cellquorum.stages.comparative.differential_abundance.milo_method", "_MILO_R"),
    ("cellquorum.stages.comparative.differential_abundance.propeller_method", "_PROPELLER_R"),
    ("cellquorum.stages.comparative.differential_expression.pseudobulk_edger_method", "_EDGER_R"),
    ("cellquorum.stages.comparative.multicellular_programs.dialogue_method", "_DIALOGUE_R"),
    ("cellquorum.stages.annotation.diagnostics.scdiagnostics_method", "_SCDIAGNOSTICS_R"),
    ("cellquorum.stages.qc.doublets", "_SCDBLFINDER_R"),
    ("cellquorum.stages.ambient_correction.soupx", "_SOUPX_R"),
    ("cellquorum.stages.cell_cell_communication.nichenet_method", "_NICHENET_R"),
    ("cellquorum.stages.cell_cell_communication.multinichenet_method", "_MNN_R"),
]


@pytest.mark.parametrize(("module_path", "constant"), R_SCRIPT_REFERENCES)
def test_stage_r_script_constant_resolves_to_an_existing_file(module_path, constant):
    module = importlib.import_module(module_path)
    path = Path(getattr(module, constant))
    assert path.is_file(), f"{module_path}.{constant} -> {path} does not exist"


def test_r_scripts_dir_is_inside_the_backends_package():
    # The whole point of the resolver: the directory is found relative to the
    # backends package, never relative to a calling stage module.
    assert R_SCRIPTS_DIR.is_dir()
    assert R_SCRIPTS_DIR.parent.name == "backends"


def test_no_stage_module_rebuilds_an_r_scripts_path_by_hand():
    # A reintroduced relative chain would drift again the next time stages move.
    stages_dir = Path(importlib.import_module("cellquorum.stages").__file__).parent
    offenders = [
        py.relative_to(stages_dir).as_posix()
        for py in stages_dir.rglob("*.py")
        if '"r_scripts"' in py.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "these modules build an r_scripts path by hand; use "
        f"backends.script_paths.r_script_path instead: {offenders}"
    )


def test_available_r_scripts_lists_the_bundled_scripts():
    scripts = available_r_scripts()
    assert "choir.R" in scripts
    assert all(name.endswith(".R") for name in scripts)


def test_r_script_exists_discriminates():
    assert r_script_exists("choir.R")
    assert not r_script_exists("definitely_not_a_bundled_script.R")


def test_r_script_path_returns_a_path_even_when_absent():
    # Callers report a missing script as a stage skip with their own context, so
    # the resolver must not raise for an unknown name.
    assert r_script_path("nope.R").name == "nope.R"
