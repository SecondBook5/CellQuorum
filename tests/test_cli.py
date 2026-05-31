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