"""The command-line journey a lab member takes: plan, run, run again, and make mistakes.

``test_installed_workflow`` proves the engine works through the Python API. This proves the
*command* a person actually types works from the installed package, and that the usual
mistakes (a typo in the YAML, a wrong file path, a misspelled column) produce a one-line
message and a non-zero exit code rather than a traceback or, worse, a clean-looking run.
"""

from __future__ import annotations

import copy
import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from test_installed_workflow import config_for, fixture_file

import cellquorum

pytestmark = pytest.mark.integration

TIMEOUT_SECONDS = 600


def cli(*args: str, cwd: Path, answers: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the CellQuorum command line in a fresh process, as a user would."""
    return subprocess.run(
        [sys.executable, "-m", "cellquorum.cli.app", *args],
        capture_output=True,
        text=True,
        input=answers,
        cwd=cwd,
        timeout=TIMEOUT_SECONDS,
        env={**os.environ, "COLUMNS": "200"},
    )


def output_of(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def test_own_data_setup_to_donor_abundance_report(tmp_path: Path) -> None:
    source = tmp_path / "annotated study.h5ad"
    fixture_file(source, n_genes=400, labeled=True)
    answers = [
        "abundance",
        "layers/counts",
        "human",
        "cells",
        "sample",
        "cell_type",
        "donor",
        "condition",
        "control",
        "treated",
        "paired",
    ]
    made = cli(
        "init", "my study", "--input", str(source), cwd=tmp_path, answers="\n".join(answers) + "\n"
    )
    assert made.returncode == 0, output_of(made)
    assert "3 complete replicates" in made.stdout
    ran = cli("run", "-c", "my study/config.yaml", cwd=tmp_path)
    assert ran.returncode == 0, output_of(ran)
    run = tmp_path / "my study" / "run"
    with (run / "results" / "da_proportion_ttest.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["cell_type"] for row in rows} == {"T cell", "B cell"}
    assert all(row["n_case"] == row["n_control"] == "3" for row in rows)
    assert "da_proportion_ttest.csv" in (run / "reports" / "report.html").read_text()


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A fresh input and configuration, also used for invalid-config variants."""
    root = tmp_path_factory.mktemp("cli_journey")
    source = root / "input.h5ad"
    fixture_file(source)
    config = config_for(source)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return {"root": root, "source": source, "config": config, "config_path": config_path}


@pytest.fixture(scope="module")
def completed_run(study: dict) -> dict:
    root = study["root"]
    out = root / "out"
    run = cli("run", "--config", str(study["config_path"]), "--output-dir", str(out), cwd=root)
    return {**study, "out": out, "run": run}


def write_variant(study: dict, name: str, mutate) -> Path:
    config = copy.deepcopy(study["config"])
    mutate(config)
    path = study["root"] / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_version_matches_the_installed_package(tmp_path: Path) -> None:
    result = cli("--version", cwd=tmp_path)
    assert result.returncode == 0
    assert cellquorum.__version__ in result.stdout


def test_the_console_script_is_installed() -> None:
    script = Path(sys.executable).parent / ("cellquorum.exe" if os.name == "nt" else "cellquorum")
    if os.environ.get("CELLQUORUM_REQUIRE_WHEEL"):
        assert script.is_file(), f"the wheel did not install a 'cellquorum' command at {script}"
    if script.is_file():
        result = subprocess.run([str(script), "--version"], capture_output=True, text=True)
        assert result.returncode == 0 and cellquorum.__version__ in result.stdout


def test_plan_lists_the_enabled_stages_without_running_them(study: dict) -> None:
    root = study["root"]
    before = set(root.iterdir())
    result = cli("plan", "--config", str(study["config_path"]), cwd=root)
    assert result.returncode == 0, output_of(result)
    text = output_of(result)
    for stage in ("qc_evidence", "preprocessing", "clustering", "embeddings"):
        assert stage in text
    assert "Traceback" not in text
    assert set(root.iterdir()) == before, "plan must not create a run directory"


def test_run_completes_and_produces_the_outputs_a_user_reads(completed_run: dict) -> None:
    result = completed_run["run"]
    text = output_of(result)
    assert result.returncode == 0, text
    assert "CellQuorum run complete" in text
    assert "Traceback" not in text
    assert "oneDNN" not in text and "absl::InitializeLog" not in text
    out = completed_run["out"]
    assert (out / "reports" / "report.html").is_file()
    assert (out / "reports" / "methods.txt").is_file()
    assert (out / "objects" / "final_annotated.h5ad").is_file()
    assert (out / "provenance" / "resolved_config.json").is_file()
    assert list(out.rglob("*.png")), "the run produced no figures"


def test_running_again_into_the_same_folder_is_safe(completed_run: dict) -> None:
    assert completed_run["run"].returncode == 0, output_of(completed_run["run"])
    again = cli(
        "run",
        "--config",
        str(completed_run["config_path"]),
        "--output-dir",
        str(completed_run["out"]),
        cwd=completed_run["root"],
    )
    assert again.returncode == 0, output_of(again)
    assert (completed_run["out"] / "reports" / "report.html").is_file()
    assert (completed_run["out"] / "objects" / "final_annotated.h5ad").is_file()


MISTAKES = [
    pytest.param(
        lambda c: c["input"].update(h5ad="/no/such/file.h5ad"),
        ["does not exist", "/no/such/file.h5ad"],
        id="wrong-input-path",
    ),
    pytest.param(
        lambda c: c["stages"].update(not_a_real_stage=True),
        ["stages.not_a_real_stage", "unknown setting"],
        id="misspelled-setting",
    ),
    pytest.param(
        lambda c: c["run"].update(random_seed=-5),
        ["run.random_seed", "-5"],
        id="invalid-value",
    ),
]


@pytest.mark.parametrize(("mutate", "expected"), MISTAKES)
def test_a_mistake_in_the_config_is_one_readable_error(study: dict, mutate, expected) -> None:
    path = write_variant(study, "mistake", mutate)
    result = cli(
        "run",
        "--config",
        str(path),
        "--output-dir",
        str(study["root"] / "x"),
        cwd=study["root"],
    )
    text = output_of(result)
    assert result.returncode == 1, text
    assert "Traceback" not in text
    assert "errors.pydantic.dev" not in text
    for fragment in expected:
        assert fragment in text, text


def test_a_broken_yaml_file_is_reported_not_crashed_on(study: dict) -> None:
    broken = study["root"] / "broken.yaml"
    broken.write_text("project: [unclosed\n  input: {")
    result = cli("run", "--config", str(broken), cwd=study["root"])
    text = output_of(result)
    assert result.returncode == 1 and "Traceback" not in text
    assert "Failed to load configuration file" in text


def test_a_missing_config_file_says_so(study: dict) -> None:
    result = cli("run", "--config", "nope.yaml", cwd=study["root"])
    text = output_of(result)
    assert result.returncode == 1 and "Traceback" not in text
    assert "Configuration file does not exist" in text


def test_stage_failure_has_a_failed_run_heading(study: dict) -> None:
    path = write_variant(
        study,
        "impossible_floor",
        lambda c: c["qc_evidence"]["floors"].update(min_genes_per_cell=1000),
    )
    result = cli("run", "-c", str(path), "-o", str(study["root"] / "failed"), cwd=study["root"])
    assert result.returncode == 1, output_of(result)
    assert "CellQuorum run failed" in result.stdout and "Failed: qc_evidence" in result.stdout
    assert "CellQuorum run complete" not in result.stdout


def test_a_misspelled_column_name_is_not_a_silent_success(study: dict) -> None:
    """The run is allowed to finish (stages that need the column check it themselves), but the
    mistake must be in the closing summary and the report, not only in a provenance file."""
    path = write_variant(
        study, "typo", lambda c: c["cohort"].update(condition_key="treatment_typo")
    )
    out = study["root"] / "out_typo"
    result = cli("run", "--config", str(path), "--output-dir", str(out), cwd=study["root"])
    text = output_of(result)
    assert result.returncode == 0, text
    assert "treatment_typo" in text
    report = (out / "reports" / "report.md").read_text()
    assert "treatment_typo" in report
    assert "No warnings were emitted" not in report


def test_the_readme_quickstart_works_as_written(tmp_path: Path) -> None:
    """`cellquorum demo`, then `plan`, then `run`: the three commands the README opens with."""
    demo = tmp_path / "try_it"
    made = cli("demo", str(demo), cwd=tmp_path)
    assert made.returncode == 0, output_of(made)
    config = demo / "config.yaml"
    assert config.is_file() and (demo / "input.h5ad").is_file()

    planned = cli("plan", "--config", str(config), cwd=tmp_path)
    assert planned.returncode == 0, output_of(planned)

    ran = cli("run", "--config", str(config), "--output-dir", str(demo / "run"), cwd=tmp_path)
    text = output_of(ran)
    assert ran.returncode == 0, text
    assert "CellQuorum run complete" in text and "Traceback" not in text
    # The stages the demo leaves switched off are counted, not listed as skipped.
    assert "Not enabled in this configuration" in text
    assert "Skipped" not in text
    report = demo / "run" / "reports" / "report.html"
    html = report.read_text()
    assert "CellQuorum run report" in html and "Open the run report:" in text
    assert 'href="../figures/embeddings/categorical_umap_leiden.pdf"' in html
    assert 'href="../results/qc/cell_metrics.csv"' in html
    with (demo / "run/provenance/artifact_manifest.csv").open() as handle:
        indexed = {Path(row["path"]) for row in csv.DictReader(handle)}
    assert {
        report,
        demo / "run/objects/final_annotated.h5ad",
        demo / "run/figures/embeddings/categorical_umap_leiden.pdf",
    } <= indexed
    assert list((demo / "run").rglob("*.png")), "the demo run produced no figures"


def test_guided_populations_from_public_filtered_10x_counts(tmp_path: Path) -> None:
    """Exercise the previously interrupted single-library publication run from the CLI."""
    import json

    import anndata as ad
    import numpy as np
    import pandas as pd

    matrix = os.environ.get("CELLQUORUM_PBMC_10X_H5")
    if not matrix:
        pytest.skip("set CELLQUORUM_PBMC_10X_H5 to the public 1K PBMC v3 filtered GEX H5")
    marker = tmp_path / "markers.yaml"
    marker.write_text("B: [MS4A1, CD79A, CD79B]\nNK: [GNLY, NKG7, KLRD1]\n")
    answers = [
        "questions",
        "X",
        "human",
        "cells",
        "sample_id",
        "populations",
        "",
        str(marker),
        "",
        "",
    ]
    setup = cli(
        "init",
        "pbmc",
        "--input",
        str(Path(matrix).resolve()),
        cwd=tmp_path,
        answers="\n".join(answers) + "\n",
    )
    assert setup.returncode == 0, output_of(setup)
    for command in ("plan", "run"):
        proc = cli(command, "-c", "pbmc/config.yaml", "--json", cwd=tmp_path)
        assert proc.returncode == 0, output_of(proc)
    summary = json.loads(proc.stdout)
    assert set(summary["enabled_stages"]) == set(summary["successful_stages"]) - {"bootstrap"}
    assert not summary["not_reached_stages"] and not summary["unable_to_run_stages"]
    run = tmp_path / "pbmc/run"
    obj = ad.read_h5ad(run / "objects/final_annotated.h5ad")
    counts = pd.read_csv(run / "results/population_identity/tables/population_by_annotation.csv")
    assert counts.n_cells.sum() == obj.n_obs
    means = pd.read_csv(run / "results/annotation/marker_dotplot_mean.csv", index_col=0)
    marker_expr = obj[:, "MS4A1"].layers["cellquorum_normalized"]
    for cluster in means.columns:
        mask = obj.obs.leiden.astype(str).eq(cluster).to_numpy()
        np.testing.assert_allclose(means.loc["MS4A1", cluster], marker_expr[mask].mean(), atol=1e-5)
    assert (run / "figures/annotation/marker_dotplot.pdf").is_file()
    assert (run / "reports/publication/populations.svg").is_file()
