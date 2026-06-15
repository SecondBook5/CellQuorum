"""Tests for the CellQuorum command-line interface."""

from __future__ import annotations

# Import JSON so CLI JSON output can be parsed in tests.
import json

# Import Path for temporary config file paths.
from pathlib import Path

# Import Typer's test runner for CLI tests.
from typer.testing import CliRunner

# Import the Typer application.
from cellquorum.cli.app import app

# Import the package version for version-output testing.
from cellquorum.version import __version__

# Create a reusable CLI runner for all tests in this file.
runner = CliRunner()


def test_cli_version_option_prints_version() -> None:
    """
    Verify that the global --version option prints the CellQuorum version.

    The CLI previously reached the plan command correctly, but the version flag
    needed explicit eager callback handling. This test protects that behavior so
    `cellquorum --version` remains a working lightweight health check.
    """

    # Invoke the CLI with the global version flag.
    result = runner.invoke(app, ["--version"])

    # Confirm the command exited successfully.
    assert result.exit_code == 0

    # Confirm the package version appears in stdout.
    assert f"cellquorum {__version__}" in result.stdout


def test_cli_no_arguments_prints_help() -> None:
    """
    Verify that running the CLI without a command prints help.

    CellQuorum should fail gently when users run the root command without
    arguments. Showing help is better than returning an unclear missing-command
    error.
    """

    # Invoke the CLI without arguments.
    result = runner.invoke(app, [])

    # Confirm the command exits successfully after printing help.
    assert result.exit_code == 0

    # Confirm the help output includes the application name.
    assert "CellQuorum" in result.stdout

    # Confirm the help output includes the plan command.
    assert "plan" in result.stdout


def test_cli_plan_command_prints_stage_and_backend_tables(tmp_path: Path) -> None:
    """
    Verify that the plan command prints a human-readable execution plan.

    The plan command should validate configuration, build a stage plan, report
    backend availability, and exit without running heavy analysis.
    """

    # Create a temporary configuration file.
    config_path = tmp_path / "config.yaml"

    # Write a valid CellQuorum config with R disabled to reduce environment warnings.
    config_path.write_text(
        """
project:
  name: cli_plan_project
run:
  profile: standard
compute:
  backend: cpu
  prefer_gpu: false
r:
  enabled: false
""",
        encoding="utf-8",
    )

    # Invoke the plan command with the temporary config.
    result = runner.invoke(app, ["plan", "--config", str(config_path)])

    # Confirm the command exited successfully.
    assert result.exit_code == 0

    # Confirm the plan header appears.
    assert "CellQuorum plan" in result.stdout

    # Confirm the selected profile appears.
    assert "Profile:" in result.stdout

    # Confirm the stage plan table appears.
    assert "CellQuorum stage plan" in result.stdout

    # Confirm a core stage appears in the plan output.
    assert "qc" in result.stdout

    # Confirm an advanced gated capability appears in the plan output.
    assert "molecular_inference" in result.stdout

    # Confirm the backend status table appears.
    assert "CellQuorum backend status" in result.stdout

    # Confirm the Python backend appears in the backend table.
    assert "python" in result.stdout


def test_cli_plan_json_outputs_machine_readable_plan(tmp_path: Path) -> None:
    """
    Verify that the plan command can emit JSON output.

    Machine-readable plan output is needed for future provenance files, CI
    checks, and programmatic wrappers such as Nextflow.
    """

    # Create a temporary configuration file.
    config_path = tmp_path / "config.yaml"

    # Write a valid CellQuorum config.
    config_path.write_text(
        """
project:
  name: cli_json_project
run:
  profile: publication
compute:
  backend: cpu
  prefer_gpu: false
r:
  enabled: false
""",
        encoding="utf-8",
    )

    # Invoke the plan command with JSON output and a wide terminal to avoid wrapping.
    result = runner.invoke(
        app,
        ["plan", "--config", str(config_path), "--json"],
        terminal_width=240,
    )

    # Confirm the command exited successfully.
    assert result.exit_code == 0

    # Parse the JSON payload from stdout.
    payload = json.loads(result.stdout)

    # Confirm the selected profile was serialized.
    assert payload["profile"] == "publication"

    # Confirm stages are serialized as a list.
    assert isinstance(payload["stages"], list)

    # Confirm backend statuses are serialized as a list.
    assert isinstance(payload["backend_status_table"], list)

    # Confirm planner warnings are serialized as a list.
    assert isinstance(payload["warnings"], list)

    # Extract stage names from the serialized plan.
    stage_names = {stage["name"] for stage in payload["stages"]}

    # Confirm a core stage is present.
    assert "qc" in stage_names

    # Confirm an advanced gated capability is present.
    assert "network_analysis" in stage_names

    # Extract backend names from the serialized plan.
    backend_names = {backend["name"] for backend in payload["backend_status_table"]}

    # Confirm the default Python backend is present.
    assert "python" in backend_names

    # Confirm the RAPIDS backend is represented even if unavailable.
    assert "rapids" in backend_names


def test_cli_plan_rejects_missing_config_file(tmp_path: Path) -> None:
    """
    Verify that the plan command reports missing configuration files clearly.

    CLI users should receive a concise configuration error rather than a raw
    traceback when a config path is wrong.
    """

    # Define a missing config path.
    missing_config_path = tmp_path / "missing.yaml"

    # Invoke the plan command with the missing config path.
    result = runner.invoke(app, ["plan", "--config", str(missing_config_path)])

    # Confirm the command exits with an error.
    assert result.exit_code == 1

    # Confirm the error output identifies a configuration problem.
    assert "Configuration error" in result.stdout

    # Confirm the missing-file reason appears.
    assert "does not exist" in result.stdout


def test_cli_plan_rejects_invalid_config_file(tmp_path: Path) -> None:
    """
    Verify that the plan command reports invalid configuration files clearly.

    Configs that parse as YAML but fail Pydantic validation should produce a
    CellQuorum-specific configuration error.
    """

    # Create a temporary invalid configuration file.
    config_path = tmp_path / "invalid.yaml"

    # Write a config with an unsupported run profile.
    config_path.write_text(
        """
run:
  profile: impossible_profile
""",
        encoding="utf-8",
    )

    # Invoke the plan command with the invalid config.
    result = runner.invoke(app, ["plan", "--config", str(config_path)])

    # Confirm the command exits with an error.
    assert result.exit_code == 1

    # Confirm the output identifies a configuration problem.
    assert "Configuration error" in result.stdout

    # Confirm the validation failure message appears.
    assert "Invalid CellQuorum configuration" in result.stdout


def test_cli_run_command_initializes_pipeline_run(tmp_path: Path) -> None:
    """
    Verify that the run command initializes a CellQuorum execution frame.

    The run command should validate configuration, create the standardized output
    layout, write provenance artifacts, and report the important output paths to
    the user.
    """

    # Create a temporary configuration file.
    config_path = tmp_path / "config.yaml"

    # Create a temporary output directory path.
    output_dir = tmp_path / "run_output"

    # Write a valid CellQuorum config with stable local settings.
    config_path.write_text(
        """
project:
  name: cli_run_project
run:
  profile: standard
compute:
  backend: cpu
  prefer_gpu: false
r:
  enabled: false
""",
        encoding="utf-8",
    )

    # Invoke the run command with bootstrap-only mode.
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-only",
        ],
    )

    # Confirm the command exited successfully.
    assert result.exit_code == 0

    # Confirm the run initialization message appears.
    assert "CellQuorum run initialized" in result.stdout

    # Confirm the run ID appears.
    assert "cli_run_project" in result.stdout

    # Confirm the output directory field appears (path may be word-wrapped).
    assert "Output directory:" in result.stdout

    # Confirm the provenance directory was created.
    assert (output_dir / "provenance").exists()

    # Confirm the artifact manifest was written.
    assert (output_dir / "provenance" / "artifact_manifest.csv").exists()

    # Confirm the pipeline plan was written.
    assert (output_dir / "provenance" / "pipeline_plan.json").exists()

    # Confirm the backend status was written.
    assert (output_dir / "provenance" / "backend_status.json").exists()


def test_cli_run_json_outputs_machine_readable_summary(tmp_path: Path) -> None:
    """
    Verify that the run command can emit a machine-readable JSON summary.

    JSON output is useful for workflow wrappers, CI checks, and future Nextflow
    integration.
    """

    # Create a temporary configuration file.
    config_path = tmp_path / "config.yaml"

    # Create a temporary output directory path.
    output_dir = tmp_path / "json_run_output"

    # Write a valid CellQuorum config.
    config_path.write_text(
        """
project:
  name: cli_run_json_project
run:
  profile: publication
compute:
  backend: cpu
  prefer_gpu: false
r:
  enabled: false
""",
        encoding="utf-8",
    )

    # Invoke the run command with JSON output and bootstrap-only mode.
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--json",
            "--bootstrap-only",
        ],
        terminal_width=240,
    )

    # Confirm the command exited successfully.
    assert result.exit_code == 0

    # Parse the JSON payload.
    payload = json.loads(result.stdout)

    # Confirm the run ID was serialized.
    assert payload["run_id"] == "cli_run_json_project"

    # Confirm the profile was serialized.
    assert payload["profile"] == "publication"

    # Confirm the output directory was serialized.
    assert payload["output_dir"] == str(output_dir.resolve())

    # Confirm the provenance directory was serialized.
    assert payload["provenance_dir"] == str((output_dir / "provenance").resolve())

    # Confirm the artifact manifest path was serialized.
    assert payload["artifact_manifest"] == str(
        (output_dir / "provenance" / "artifact_manifest.csv").resolve()
    )

    # Confirm enabled stages were serialized.
    assert "qc" in payload["enabled_stages"]

    # Confirm advanced gated stages were serialized.
    assert "network_analysis" in payload["enabled_stages"]

    # Confirm warnings were serialized as a list.
    assert isinstance(payload["warnings"], list)

    # Confirm provenance was actually written.
    assert (output_dir / "provenance" / "artifact_manifest.csv").exists()


def test_cli_run_rejects_missing_config_file(tmp_path: Path) -> None:
    """
    Verify that the run command reports missing config files clearly.

    A missing config should fail before any run output is initialized.
    """

    # Define a missing config path.
    missing_config_path = tmp_path / "missing.yaml"

    # Define an output directory path.
    output_dir = tmp_path / "missing_config_output"

    # Invoke the run command with a missing config path.
    result = runner.invoke(
        app,
        ["run", "--config", str(missing_config_path), "--output-dir", str(output_dir)],
    )

    # Confirm the command exits with an error.
    assert result.exit_code == 1

    # Confirm the configuration error appears.
    assert "Configuration error" in result.stdout

    # Confirm the missing-file reason appears.
    assert "does not exist" in result.stdout

    # Confirm the output directory was not created.
    assert not output_dir.exists()


def test_cli_run_rejects_invalid_config_file(tmp_path: Path) -> None:
    """
    Verify that the run command reports invalid config files clearly.

    YAML files that parse but fail Pydantic validation should not initialize a
    run directory.
    """

    # Create a temporary invalid config file.
    config_path = tmp_path / "invalid.yaml"

    # Define an output directory path.
    output_dir = tmp_path / "invalid_config_output"

    # Write a config with an unsupported profile.
    config_path.write_text(
        """
run:
  profile: impossible_profile
""",
        encoding="utf-8",
    )

    # Invoke the run command with the invalid config.
    result = runner.invoke(
        app,
        ["run", "--config", str(config_path), "--output-dir", str(output_dir)],
    )

    # Confirm the command exits with an error.
    assert result.exit_code == 1

    # Confirm the configuration error appears.
    assert "Configuration error" in result.stdout

    # Confirm the validation failure appears.
    assert "Invalid CellQuorum configuration" in result.stdout

    # Confirm the output directory was not created.
    assert not output_dir.exists()
