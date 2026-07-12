"""Command-line interface for CellQuorum."""

from __future__ import annotations

# Import JSON for optional machine-readable CLI output.
import json

# Import Path for config and output directory arguments.
from pathlib import Path

# Import Annotated for Typer-compatible option metadata without B008 violations.
from typing import Annotated

# Import Typer for the command-line interface.
import typer

# Import Rich console rendering for readable terminal output.
from rich.console import Console

# Import Rich tables for stage and backend summaries.
from rich.table import Table

# Import the public pipeline API.
from cellquorum.api import run_pipeline

# Import configuration loading utilities.
from cellquorum.config.loader import ConfigLoadError, load_config

# Import shared CellQuorum exception base.
from cellquorum.core.exceptions import CellQuorumError

# Import the planner entry point.
from cellquorum.core.planner import PipelinePlan, build_pipeline_plan

# Import the package version.
from cellquorum.version import __version__

# Store the default configuration path as a module-level constant.
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


# Create the Typer application.
app = typer.Typer(
    name="cellquorum",
    help="CellQuorum: publication-grade scRNA-seq workflow engine.",
    no_args_is_help=False,
    invoke_without_command=True,
)

# Create a Rich console for formatted CLI output.
console = Console()


def _print_stage_table(plan: PipelinePlan) -> None:
    """
    Print the planned CellQuorum stage table.

    The stage table gives users a direct preview of which major workflow layers
    are enabled by configuration. Later planner versions will add method-level
    gates, but this already prevents hidden behavior by showing the configured
    stage-level plan.

    Args:
        plan: PipelinePlan object produced by the planner.
    """

    # Create a Rich table for planned stages.
    table = Table(title="CellQuorum stage plan")

    # Add the stage name column.
    table.add_column("Stage", style="cyan", no_wrap=True)

    # Add the status column.
    table.add_column("Status", style="bold")

    # Add the reason column.
    table.add_column("Reason")

    # Iterate over each planned stage.
    for stage in plan.stages:
        # Choose a readable status label.
        status_label = "enabled" if stage.enabled else "disabled"

        # Choose a display style based on stage status.
        status_style = "green" if stage.enabled else "red"

        # Add the stage row.
        table.add_row(
            stage.name,
            f"[{status_style}]{status_label}[/{status_style}]",
            stage.reason,
        )

    # Print the table to the console.
    console.print(table)


def _print_backend_table(plan: PipelinePlan) -> None:
    """
    Print the backend availability table.

    Backend visibility is central to CellQuorum's design because R, Rscript, GPU,
    and RAPIDS support are optional but first-class. The CLI should show which
    backends are registered and whether they are currently available.

    Args:
        plan: PipelinePlan object produced by the planner.
    """

    # Create a Rich table for backend statuses.
    table = Table(title="CellQuorum backend status")

    # Add the backend name column.
    table.add_column("Backend", style="cyan", no_wrap=True)

    # Add the backend kind column.
    table.add_column("Kind", no_wrap=True)

    # Add the availability column.
    table.add_column("Available", style="bold", no_wrap=True)

    # Add the missing requirements column.
    table.add_column("Missing")

    # Add the warnings column.
    table.add_column("Warnings")

    # Iterate over each backend status row.
    for row in plan.backend_status_table:
        # Extract the availability flag.
        available = bool(row["available"])

        # Choose a readable availability label.
        available_label = "yes" if available else "no"

        # Choose a display style based on availability.
        available_style = "green" if available else "red"

        # Build a readable missing requirements string.
        missing = ", ".join(str(item) for item in row["missing"]) or "-"

        # Build a readable warning string.
        warnings = "; ".join(str(item) for item in row["warnings"]) or "-"

        # Add the backend row.
        table.add_row(
            str(row["name"]),
            str(row["kind"]),
            f"[{available_style}]{available_label}[/{available_style}]",
            missing,
            warnings,
        )

    # Print the table to the console.
    console.print(table)


def _print_planner_warnings(plan: PipelinePlan) -> None:
    """
    Print planner-level warnings.

    Planner warnings are not necessarily fatal. They tell the user when a desired
    backend or capability is enabled in config but unavailable in the current
    environment.

    Args:
        plan: PipelinePlan object produced by the planner.
    """

    # Return early when no planner warnings exist.
    if not plan.warnings:
        return

    # Print a warning heading.
    console.print("\n[bold yellow]Planner warnings[/bold yellow]")

    # Iterate over planner warnings.
    for warning in plan.warnings:
        # Print each warning as a bullet.
        console.print(f"[yellow]- {warning}[/yellow]")


@app.callback()
def callback(
    context: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the CellQuorum version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """
    Handle global CellQuorum CLI options.

    Args:
        context: Typer command context.
        version: Whether to print the package version and exit.
    """

    # Print the package version when requested.
    if version:
        # Print the version string.
        console.print(f"cellquorum {__version__}")

        # Exit after printing the version.
        raise typer.Exit()

    # Show help when no subcommand is provided.
    if context.invoked_subcommand is None:
        # Print command help.
        console.print(context.get_help())

        # Exit successfully after printing help.
        raise typer.Exit()


@app.command("plan")
def plan_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to a CellQuorum YAML configuration file.",
        ),
    ] = DEFAULT_CONFIG_PATH,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the pipeline plan as JSON instead of Rich tables.",
        ),
    ] = False,
) -> None:
    """
    Build and display a CellQuorum execution plan.

    This command validates the configuration, checks registered backend
    availability, and prints the configured stage plan before any heavy analysis
    is run.

    Args:
        config: Path to the CellQuorum YAML configuration file.
        json_output: Whether to print machine-readable JSON.
    """

    # Try to load and validate the configuration.
    try:
        # Load the resolved CellQuorum configuration.
        loaded_config = load_config(config)

        # Build the pipeline plan.
        pipeline_plan = build_pipeline_plan(loaded_config)

    # Convert configuration errors into CLI-friendly failures.
    except ConfigLoadError as error:
        # Print the error message in red.
        console.print(f"[bold red]Configuration error:[/bold red] {error}")

        # Exit with a non-zero status code.
        raise typer.Exit(code=1) from error

    # Print JSON output when requested.
    if json_output:
        # Serialize the plan dictionary as formatted JSON without Rich wrapping.
        typer.echo(json.dumps(pipeline_plan.to_dict(), indent=2))

        # Return after printing JSON.
        return

    # Print a concise plan header.
    console.print("[bold]CellQuorum plan[/bold]")

    # Print the selected profile.
    console.print(f"Profile: [cyan]{pipeline_plan.profile}[/cyan]\n")

    # Print the stage plan table.
    _print_stage_table(pipeline_plan)

    # Print the backend status table.
    _print_backend_table(pipeline_plan)

    # Print planner warnings.
    _print_planner_warnings(pipeline_plan)


@app.command("run")
def run_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to a CellQuorum YAML configuration file.",
        ),
    ] = DEFAULT_CONFIG_PATH,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for the CellQuorum run.",
        ),
    ] = None,
    bootstrap_only: Annotated[
        bool,
        typer.Option(
            "--bootstrap-only",
            help="Initialize directory structure without executing stages.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the initialized run summary as JSON.",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress progress output and run silently.",
        ),
    ] = False,
) -> None:
    """
    Execute a CellQuorum pipeline run.

    This command validates configuration, creates the standardized run directory
    layout, builds the backend-aware execution plan, and executes enabled
    pipeline stages. When --bootstrap-only is specified, it only initializes
    the run structure without executing stages.

    Args:
        config: Path to the CellQuorum YAML configuration file.
        output_dir: Optional output directory override.
        bootstrap_only: Whether to skip stage execution and only bootstrap.
        json_output: Whether to print machine-readable JSON.
        quiet: Whether to suppress progress output.
    """

    # Try to initialize and optionally execute the CellQuorum pipeline run.
    try:
        # Run the public pipeline API from the supplied config path.
        result = run_pipeline(
            config,
            output_dir=output_dir,
            execute=not bootstrap_only,
            quiet=quiet,
        )

    # Convert configuration failures into CLI-friendly errors.
    except ConfigLoadError as error:
        # Print the configuration error message in red.
        console.print(f"[bold red]Configuration error:[/bold red] {error}")

        # Exit with a non-zero status code.
        raise typer.Exit(code=1) from error

    # Convert CellQuorum runtime failures into CLI-friendly errors.
    except CellQuorumError as error:
        # Print the run error message in red.
        console.print(f"[bold red]Run error:[/bold red] {error}")

        # Exit with a non-zero status code.
        raise typer.Exit(code=1) from error

    # Convert unexpected bootstrap failures into CLI-friendly errors.
    except (TypeError, ValueError, RuntimeError) as error:
        # Print the unexpected run error message in red.
        console.print(f"[bold red]Run error:[/bold red] {error}")

        # Exit with a non-zero status code.
        raise typer.Exit(code=1) from error

    # Determine execution state from result.
    execution_result = result.execution_result
    executed = execution_result is not None
    successful_stages = execution_result.succeeded_stage_names() if executed else []
    skipped_stages = execution_result.skipped_stage_names() if executed else []
    failed_stages = execution_result.failed_stage_names() if executed else []

    # Build a machine-readable run summary.
    summary = {
        "run_id": result.context.run_id,
        "profile": result.plan.profile,
        "output_dir": str(result.context.paths.root),
        "provenance_dir": str(result.context.paths.provenance),
        "artifact_manifest": str(result.context.paths.provenance / "artifact_manifest.csv"),
        "pipeline_plan": str(result.context.paths.provenance / "pipeline_plan.json"),
        "backend_status": str(result.context.paths.provenance / "backend_status.json"),
        "enabled_stages": result.plan.enabled_stage_names(),
        "warnings": list(result.plan.warnings),
        "executed": executed,
        "successful_stages": successful_stages,
        "skipped_stages": skipped_stages,
        "failed_stages": failed_stages,
    }

    # Print JSON output when requested.
    if json_output:
        # Serialize the run summary without Rich wrapping.
        typer.echo(json.dumps(summary, indent=2))

        # Exit with non-zero status when stages failed.
        if failed_stages:
            raise typer.Exit(code=1)

        # Return after printing JSON.
        return

    # Print the run header.
    if executed:
        console.print("[bold]CellQuorum run complete[/bold]")
    else:
        console.print("[bold]CellQuorum run initialized[/bold]")

    # Print the run identifier.
    console.print(f"Run ID: [cyan]{summary['run_id']}[/cyan]")

    # Print the selected profile.
    console.print(f"Profile: [cyan]{summary['profile']}[/cyan]")

    # Print the root output directory.
    console.print(f"Output directory: [cyan]{summary['output_dir']}[/cyan]")

    # Print the provenance directory.
    console.print(f"Provenance: [cyan]{summary['provenance_dir']}[/cyan]")

    # Print the artifact manifest path.
    console.print(f"Artifact manifest: [cyan]{summary['artifact_manifest']}[/cyan]")

    # Print the pipeline plan path.
    console.print(f"Pipeline plan: [cyan]{summary['pipeline_plan']}[/cyan]")

    # Print the backend status path.
    console.print(f"Backend status: [cyan]{summary['backend_status']}[/cyan]")

    # Print execution summary when stages were executed.
    if executed:
        console.print("\n[bold]Stage execution summary[/bold]")

        # Print successful stages.
        if successful_stages:
            console.print(f"[green]Successful: {', '.join(successful_stages)}[/green]")

        # Print skipped stages.
        if skipped_stages:
            console.print(f"[yellow]Skipped: {', '.join(skipped_stages)}[/yellow]")

        # Print failed stages.
        if failed_stages:
            console.print(f"[red]Failed: {', '.join(failed_stages)}[/red]")

    # Print planner warnings when present.
    if summary["warnings"]:
        # Print a warning heading.
        console.print("\n[bold yellow]Planner warnings[/bold yellow]")

        # Print each warning as a bullet.
        for warning in summary["warnings"]:
            console.print(f"[yellow]- {warning}[/yellow]")

    # Exit with non-zero status when stages failed.
    if failed_stages:
        raise typer.Exit(code=1)


def main() -> None:
    """
    Run the CellQuorum command-line interface.

    This function exists so pyproject console scripts can point to a stable
    callable while tests can still import the Typer app directly.
    """

    # Execute the Typer application.
    app()
