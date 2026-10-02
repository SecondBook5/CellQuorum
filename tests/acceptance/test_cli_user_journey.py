"""The command-line journey a lab member takes: plan, run, run again, and make mistakes.

``test_installed_workflow`` proves the engine works through the Python API. This proves the
*command* a person actually types works from the installed package, and that the usual
mistakes (a typo in the YAML, a wrong file path, a misspelled column) produce a one-line
message and a non-zero exit code rather than a traceback or, worse, a clean-looking run.
"""

from __future__ import annotations

import copy
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


def cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the CellQuorum command line in a fresh process, as a user would."""
    return subprocess.run(
        [sys.executable, "-m", "cellquorum.cli.app", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=TIMEOUT_SECONDS,
        env={**os.environ, "COLUMNS": "200"},
    )


def output_of(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A valid input file and config, plus the results of one real plan, run and re-run."""
    root = tmp_path_factory.mktemp("cli_journey")
    source = root / "input.h5ad"
    fixture_file(source)
    config = config_for(source)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    out = root / "out"
    before = sorted(path.name for path in root.iterdir())
    plan = cli("plan", "--config", str(config_path), cwd=root)
    created_by_plan = sorted(path.name for path in root.iterdir()) != before
    return {
        "root": root,
        "source": source,
        "config": config,
        "config_path": config_path,
        "out": out,
        "plan": plan,
        "plan_created_a_run_directory": created_by_plan,
        "run": cli("run", "--config", str(config_path), "--output-dir", str(out), cwd=root),
    }


def write_variant(workspace: dict, name: str, mutate) -> Path:
    config = copy.deepcopy(workspace["config"])
    mutate(config)
    path = workspace["root"] / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_version_matches_the_installed_package(workspace: dict) -> None:
    result = cli("--version", cwd=workspace["root"])
    assert result.returncode == 0
    assert cellquorum.__version__ in result.stdout


def test_the_console_script_is_installed(workspace: dict) -> None:
    script = Path(sys.executable).parent / ("cellquorum.exe" if os.name == "nt" else "cellquorum")
    if os.environ.get("CELLQUORUM_REQUIRE_WHEEL"):
        assert script.is_file(), f"the wheel did not install a 'cellquorum' command at {script}"
    if script.is_file():
        result = subprocess.run([str(script), "--version"], capture_output=True, text=True)
        assert result.returncode == 0 and cellquorum.__version__ in result.stdout


def test_plan_lists_the_enabled_stages_without_running_them(workspace: dict) -> None:
    result = workspace["plan"]
    assert result.returncode == 0, output_of(result)
    text = output_of(result)
    for stage in ("qc_evidence", "preprocessing", "clustering", "embeddings"):
        assert stage in text
    assert "Traceback" not in text
    assert not workspace["plan_created_a_run_directory"], "plan must not create a run directory"


def test_run_completes_and_produces_the_outputs_a_user_reads(workspace: dict) -> None:
    result = workspace["run"]
    text = output_of(result)
    assert result.returncode == 0, text
    assert "CellQuorum run complete" in text
    assert "Traceback" not in text
    out = workspace["out"]
    assert (out / "reports" / "report.html").is_file()
    assert (out / "reports" / "methods.txt").is_file()
    assert (out / "objects" / "final_annotated.h5ad").is_file()
    assert (out / "provenance" / "resolved_config.json").is_file()
    assert list(out.rglob("*.png")), "the run produced no figures"


def test_a_run_prints_no_third_party_startup_banner(workspace: dict) -> None:
    text = output_of(workspace["run"])
    assert "oneDNN" not in text and "absl::InitializeLog" not in text


def test_running_again_into_the_same_folder_is_safe(workspace: dict) -> None:
    again = cli(
        "run",
        "--config",
        str(workspace["config_path"]),
        "--output-dir",
        str(workspace["out"]),
        cwd=workspace["root"],
    )
    assert again.returncode == 0, output_of(again)
    assert (workspace["out"] / "reports" / "report.html").is_file()
    assert (workspace["out"] / "objects" / "final_annotated.h5ad").is_file()


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
def test_a_mistake_in_the_config_is_one_readable_error(workspace: dict, mutate, expected) -> None:
    path = write_variant(workspace, "mistake", mutate)
    result = cli(
        "run",
        "--config",
        str(path),
        "--output-dir",
        str(workspace["root"] / "x"),
        cwd=workspace["root"],
    )
    text = output_of(result)
    assert result.returncode == 1, text
    assert "Traceback" not in text
    assert "errors.pydantic.dev" not in text
    for fragment in expected:
        assert fragment in text, text


def test_a_broken_yaml_file_is_reported_not_crashed_on(workspace: dict) -> None:
    broken = workspace["root"] / "broken.yaml"
    broken.write_text("project: [unclosed\n  input: {")
    result = cli("run", "--config", str(broken), cwd=workspace["root"])
    text = output_of(result)
    assert result.returncode == 1 and "Traceback" not in text
    assert "Failed to load configuration file" in text


def test_a_missing_config_file_says_so(workspace: dict) -> None:
    result = cli("run", "--config", "nope.yaml", cwd=workspace["root"])
    text = output_of(result)
    assert result.returncode == 1 and "Traceback" not in text
    assert "Configuration file does not exist" in text


def test_a_misspelled_column_name_is_not_a_silent_success(workspace: dict) -> None:
    """The run is allowed to finish (stages that need the column check it themselves), but the
    mistake must be in the closing summary and the report, not only in a provenance file."""
    path = write_variant(
        workspace, "typo", lambda c: c["cohort"].update(condition_key="treatment_typo")
    )
    out = workspace["root"] / "out_typo"
    result = cli("run", "--config", str(path), "--output-dir", str(out), cwd=workspace["root"])
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
    report = demo / "run" / "reports" / "report.html"
    assert report.is_file() and "CellQuorum run report" in report.read_text()
    assert list((demo / "run").rglob("*.png")), "the demo run produced no figures"
