"""Public Python API for CellQuorum."""

from __future__ import annotations

# Import Path for file and output directory arguments.
from pathlib import Path

# Import backend registry typing for custom execution contexts.
from cellquorum.backends.registry import BackendRegistry

# Import config validation helper for dictionary-based API calls.
from cellquorum.config.loader import validate_config_dict

# Import the validated top-level config model.
from cellquorum.config.models import CellQuorumConfig

# Import bootstrap and execution pipeline utilities.
from cellquorum.core.pipeline import (
    PipelineRunResult,
    bootstrap_pipeline_run,
    bootstrap_pipeline_run_from_config_file,
    execute_pipeline_run,
    execute_pipeline_run_from_config_file,
)


def run_pipeline(
    config: str | Path | CellQuorumConfig | dict,
    *,
    output_dir: str | Path | None = None,
    backend_registry: BackendRegistry | None = None,
    execute: bool = True,
    load_input: bool = True,
    quiet: bool = False,
) -> PipelineRunResult:
    """
    Run the CellQuorum pipeline through the public Python API.

    This is the main programmatic entry point for users who want to run
    CellQuorum from Python instead of the command line. By default, it executes
    registered stages through the CellQuorum executor. Pass ``execute=False`` to
    preserve the bootstrap-only behavior used for planning, provenance setup, or
    dry-run-style infrastructure checks.

    The API accepts three configuration styles because different users work in
    different modes. YAML paths are best for reproducible projects. Pydantic
    models are best for internal package code and strongly typed workflows.
    Dictionaries are useful for notebooks, tests, and quick programmatic runs.

    Args:
        config: Configuration source. Supported values are a YAML path, a
            validated CellQuorumConfig object, or a plain dictionary that can be
            validated into CellQuorumConfig.
        output_dir: Optional explicit output directory override.
        backend_registry: Optional backend registry for custom execution or tests.
        execute: Whether to execute registered stages. If False, only bootstrap
            the run frame and write initial provenance.
        load_input: Whether executed runs should load config.input.h5ad into
            context.adata before stage execution.
        quiet: Whether to suppress progress output by overriding config.run.verbose.

    Returns:
        PipelineRunResult containing the validated config, pipeline plan,
        initialized or final context, provenance artifacts, and optional
        execution result.

    Raises:
        TypeError: If config is not a supported configuration source.
    """

    # Run from an existing validated CellQuorumConfig object.
    if isinstance(config, CellQuorumConfig):
        # Override verbosity when quiet is requested.
        if quiet:
            # Create a copy with verbose=False (config is immutable/frozen).
            config = config.model_copy(update={"run": {"verbose": False}})

        # Execute registered stages when requested.
        if execute:
            return execute_pipeline_run(
                config,
                output_dir=output_dir,
                backend_registry=backend_registry,
                load_input=load_input,
            )

        # Preserve bootstrap-only behavior when execution is disabled.
        return bootstrap_pipeline_run(
            config,
            output_dir=output_dir,
            backend_registry=backend_registry,
        )

    # Run from a dictionary by validating it into CellQuorumConfig.
    if isinstance(config, dict):
        # Validate the dictionary into the strict top-level config model.
        validated_config = validate_config_dict(config)

        # Override verbosity when quiet is requested.
        if quiet:
            validated_config = validated_config.model_copy(update={"run": {"verbose": False}})

        # Execute registered stages when requested.
        if execute:
            return execute_pipeline_run(
                validated_config,
                output_dir=output_dir,
                backend_registry=backend_registry,
                load_input=load_input,
            )

        # Preserve bootstrap-only behavior when execution is disabled.
        return bootstrap_pipeline_run(
            validated_config,
            output_dir=output_dir,
            backend_registry=backend_registry,
        )

    # Run from a YAML config path.
    if isinstance(config, str | Path):
        # Load and optionally override verbosity.
        if quiet:
            # Import config loader.
            from cellquorum.config.loader import load_config

            loaded_config = load_config(config)
            loaded_config = loaded_config.model_copy(update={"run": {"verbose": False}})

            # Execute registered stages when requested.
            if execute:
                return execute_pipeline_run(
                    loaded_config,
                    output_dir=output_dir,
                    backend_registry=backend_registry,
                    load_input=load_input,
                )
            # Bootstrap-only when execution disabled.
            return bootstrap_pipeline_run(
                loaded_config,
                output_dir=output_dir,
                backend_registry=backend_registry,
            )

        # Execute registered stages from the config file when requested.
        if execute:
            return execute_pipeline_run_from_config_file(
                config,
                output_dir=output_dir,
                backend_registry=backend_registry,
                load_input=load_input,
            )

        # Preserve bootstrap-only behavior from the config file.
        return bootstrap_pipeline_run_from_config_file(
            config,
            output_dir=output_dir,
            backend_registry=backend_registry,
        )

    # Reject unsupported config inputs with a clear error.
    raise TypeError(
        "run_pipeline expected config to be a path, CellQuorumConfig, or dictionary. "
        f"Received: {type(config).__name__}"
    )


__all__ = [
    "PipelineRunResult",
    "run_pipeline",
]
