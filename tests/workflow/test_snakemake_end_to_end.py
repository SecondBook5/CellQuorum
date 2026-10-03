"""The Snakemake matrix workflow runs for real: manifest -> configs -> run -> bundle -> status.

``test_snakefile_dag.py`` checks the plan Snakemake would follow. This runs it, on the
synthetic demo project, so a change that breaks the hand-off between the generator, the
engine and the bundler is caught here and not on a lab member's first multi-hypothesis run.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from repo_paths import REPO_ROOT

from cellquorum.datasets.demo import write_demo_project

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("snakemake") is None, reason="snakemake is not installed"),
]

SCAFFOLD_METHODS = [
    "pseudobulk",
    "subclustering",
    "pathway_enrichment",
    "rna_velocity",
    "phate_pseudotime",
    "cell_cell_communication",
    "progeny",
]
# The demo has no batches or labels, so the stages that need them are switched off.
STAGES_THE_DEMO_CANNOT_RUN = [
    "integration",
    "annotation",
    "annotation_consensus",
    "annotation_diagnostics",
    "population_identity",
]


@pytest.fixture(scope="module")
def workflow_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("snakemake_e2e")
    data_path, demo_config_path = write_demo_project(root / "demo")

    demo_config = yaml.safe_load(demo_config_path.read_text())
    template = {key: value for key, value in demo_config.items() if key not in ("project", "input")}
    template["project"] = {"name": "placeholder"}
    template["input"] = {"h5ad": "/placeholder.h5ad", "counts_layer": "counts"}
    (root / "template.yaml").write_text(yaml.safe_dump(template, sort_keys=False))

    manifest = {
        "demo_hypothesis": {
            "title": "Demo hypothesis",
            "cell_types": ["ALL"],
            "inputs": {"ALL": str(data_path)},
            "skip": {method: "not applicable to the synthetic demo" for method in SCAFFOLD_METHODS},
            "config_overrides": {"stages": dict.fromkeys(STAGES_THE_DEMO_CANNOT_RUN, False)},
        }
    }
    (root / "hypotheses.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    # The Snakefile calls `cellquorum` and `gen-configs` by name, so they must resolve to the
    # interpreter under test and not to whatever else is on PATH.
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    workdir = root / "out"
    result = subprocess.run(
        [
            "snakemake",
            "--snakefile",
            str(REPO_ROOT / "workflow" / "Snakefile"),
            "--cores",
            "1",
            "--config",
            f"manifest={root / 'hypotheses.yaml'}",
            f"template={root / 'template.yaml'}",
            f"workdir={workdir}",
        ],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=1500,
    )
    return {"root": root, "workdir": workdir, "result": result}


def test_the_workflow_completes(workflow_run: dict) -> None:
    result = workflow_run["result"]
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def test_each_cell_type_gets_a_full_engine_run(workflow_run: dict) -> None:
    run_dir = workflow_run["workdir"] / "runs" / "demo_hypothesis" / "ALL"
    records = json.loads((run_dir / "provenance" / "stage_execution_records.json").read_text())
    by_stage = {record["stage_name"]: record["status"] for record in records}
    for stage in ("qc_evidence", "preprocessing", "dimensionality", "clustering"):
        assert by_stage[stage] == "success", by_stage
    assert not [stage for stage, status in by_stage.items() if status == "failed"]
    assert (run_dir / "reports" / "report.html").is_file()


def test_the_hypothesis_bundle_is_assembled(workflow_run: dict) -> None:
    bundle = workflow_run["workdir"] / "bundles" / "demo_hypothesis"
    assert (bundle / "report.html").is_file()
    assert (bundle / "bundle_status.json").is_file()


def test_the_status_matrix_accounts_for_every_method_and_names_the_omissions(
    workflow_run: dict,
) -> None:
    rows = list(
        csv.DictReader((workflow_run["workdir"] / "runs" / "matrix_status.csv").open(newline=""))
    )
    status = {row["method"]: row["status"] for row in rows}
    assert set(status) == set(SCAFFOLD_METHODS)
    assert set(status.values()) == {"skipped"}, status
