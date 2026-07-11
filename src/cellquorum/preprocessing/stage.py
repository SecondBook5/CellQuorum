"""Preprocessing pipeline stage for CellQuorum."""

from __future__ import annotations

# Import JSON for summary artifact writing.
import json

# Import Mapping for dictionary-like config resolution.
from collections.abc import Mapping

# Import dataclass for the concrete stage object.
from dataclasses import dataclass

# Import Path for stage output directory handling.
from pathlib import Path

# Import AnnData for stage input and output typing.
import anndata as ad

# Import GPU compute routing.
from cellquorum.compute.router import should_use_gpu

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError

# Import pipeline stage artifact and result contracts.
from cellquorum.core.stage import StageArtifact, StageResult

# Import preprocessing configuration.
from cellquorum.preprocessing.config import PreprocessingConfig, validate_preprocessing_config_dict

# Import normalization implementation.
from cellquorum.preprocessing.normalization import (
    NormalizationResult,
    normalize_adata,
)


class PreprocessingStageError(CellQuorumDataError):
    """
    Report preprocessing stage execution failures.

    The preprocessing stage transforms raw count matrices into analysis-ready
    expression values. Errors here should explain whether the failure came from
    missing context state, invalid configuration, or normalization execution.
    """


@dataclass(frozen=True)
class PreprocessingStage:
    """
    Execute the complete CellQuorum preprocessing module.

    The stage wires together preprocessing transformations:

    1. validate and resolve preprocessing configuration
    2. normalize count matrix using configured recipe
    3. write machine-readable artifacts
    4. return a StageResult for provenance and downstream stages

    Args:
        config: Optional PreprocessingConfig override. If omitted, the stage
            resolves preprocessing configuration from context.config.preprocessing
            when available, otherwise it uses PreprocessingConfig().
        output_subdir: Subdirectory under context.paths.results where
            preprocessing artifacts should be written.
    """

    # Store the stable stage name expected by the pipeline contract.
    name: str = "preprocessing"

    # Store an optional explicit preprocessing configuration override.
    config: PreprocessingConfig | None = None

    # Store the results subdirectory used for preprocessing artifacts.
    output_subdir: str = "preprocessing"

    def run(self, context: object) -> StageResult:
        """
        Execute the preprocessing stage.

        Args:
            context: PipelineContext-like object containing config, paths, and
                AnnData.

        Returns:
            StageResult containing the preprocessing-updated AnnData object,
            written artifacts, notes, warnings, and structured preprocessing
            metrics.

        Raises:
            PreprocessingStageError: If required context state is missing or
                preprocessing execution fails.
        """

        # Retrieve the active AnnData object.
        adata = get_context_adata(context)

        # Resolve the effective preprocessing configuration.
        preprocessing_config = resolve_preprocessing_config(context, override=self.config)

        # Return an explicit no-op result when preprocessing is disabled.
        if not is_preprocessing_stage_enabled(context, preprocessing_config):
            return build_disabled_preprocessing_stage_result(
                adata=adata,
                stage_name=self.name,
                preprocessing_config=preprocessing_config,
            )

        # Return an explicit no-op result when normalization is disabled.
        if not preprocessing_config.normalization.enabled:
            return build_disabled_normalization_stage_result(
                adata=adata,
                stage_name=self.name,
                preprocessing_config=preprocessing_config,
            )

        # Resolve the preprocessing artifact output directory.
        output_dir = get_preprocessing_output_dir(context, self.output_subdir)

        # Ensure the output directory exists.
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine whether to use GPU compute.
        use_gpu = should_use_gpu(context)

        # Normalize the AnnData object.
        normalization_result = normalize_adata(
            adata,
            preprocessing_config.normalization,
            copy=True,
            use_gpu=use_gpu,
        )

        # Write preprocessing summary artifact.
        summary_path = write_preprocessing_summary(
            output_dir=output_dir,
            normalization_result=normalization_result,
            preprocessing_config=preprocessing_config,
            context=context,
            stage_name=self.name,
        )

        # Build stage artifacts.
        stage_artifacts = [
            StageArtifact(
                name="preprocessing_summary",
                path=summary_path,
                kind="json",
                description="Structured preprocessing summary JSON.",
            )
        ]

        # Combine warnings from all preprocessing layers.
        warnings = normalization_result.warnings

        # Build human-readable stage notes.
        notes = build_preprocessing_stage_notes(
            preprocessing_config=preprocessing_config,
            normalization_result=normalization_result,
            input_adata=adata,
            output_adata=normalization_result.adata,
        )

        # Build structured stage metrics for provenance.
        stage_metrics = build_preprocessing_stage_metrics(
            stage_name=self.name,
            preprocessing_config=preprocessing_config,
            normalization_result=normalization_result,
            input_adata=adata,
            output_adata=normalization_result.adata,
        )

        # Return the stage result.
        return StageResult(
            adata=normalization_result.adata,
            artifacts=stage_artifacts,
            notes=notes,
            warnings=warnings,
            metrics=stage_metrics,
        )


def resolve_preprocessing_config(
    context: object,
    *,
    override: PreprocessingConfig | None = None,
) -> PreprocessingConfig:
    """
    Resolve the effective preprocessing configuration for a stage run.

    Resolution order:

    1. explicit PreprocessingStage(config=...) override
    2. context.config when it is already a PreprocessingConfig
    3. context.config.preprocessing when present
    4. context.config["preprocessing"] when present
    5. PreprocessingConfig() defaults

    Args:
        context: PipelineContext-like object.
        override: Optional explicit PreprocessingConfig override.

    Returns:
        Resolved PreprocessingConfig.

    Raises:
        PreprocessingStageError: If the resolved preprocessing config is invalid.
    """

    # Prefer the explicit stage-level override.
    if override is not None:
        # Validate override type.
        if not isinstance(override, PreprocessingConfig):
            raise PreprocessingStageError(
                "PreprocessingStage config override must be a PreprocessingConfig object. "
                f"Received: {type(override).__name__}."
            )

        # Return the override.
        return override

    # Read the context-level config if present.
    context_config = getattr(context, "config", None)

    # Accept context.config as a PreprocessingConfig directly.
    if isinstance(context_config, PreprocessingConfig):
        return context_config

    # Resolve context.config["preprocessing"] for dictionary-like configs.
    if isinstance(context_config, Mapping) and "preprocessing" in context_config:
        return coerce_preprocessing_config(context_config["preprocessing"])

    # Resolve context.config.preprocessing for object-like configs.
    if hasattr(context_config, "preprocessing"):
        return coerce_preprocessing_config(context_config.preprocessing)

    # Fall back to default preprocessing configuration.
    return PreprocessingConfig()


def coerce_preprocessing_config(value: object) -> PreprocessingConfig:
    """
    Coerce a candidate preprocessing config value into PreprocessingConfig.

    Args:
        value: Candidate preprocessing configuration value.

    Returns:
        Validated PreprocessingConfig.

    Raises:
        PreprocessingStageError: If the candidate cannot become PreprocessingConfig.
    """

    # Preserve PreprocessingConfig objects.
    if isinstance(value, PreprocessingConfig):
        return value

    # Validate dictionary-like preprocessing configuration.
    if isinstance(value, Mapping):
        return validate_preprocessing_config_dict(value)

    # Reject unsupported values.
    raise PreprocessingStageError(
        "Preprocessing configuration must be a PreprocessingConfig object or mapping. "
        f"Received: {type(value).__name__}."
    )


def is_preprocessing_stage_enabled(
    context: object, preprocessing_config: PreprocessingConfig
) -> bool:
    """
    Return whether the preprocessing stage should execute.

    The stage is enabled only when PreprocessingConfig.enabled is true and any
    top-level context.config.stages.preprocessing flag is also true.

    Args:
        context: PipelineContext-like object.
        preprocessing_config: Resolved preprocessing configuration.

    Returns:
        True when preprocessing should run, otherwise False.
    """

    # Respect the preprocessing module-level enabled flag first.
    if not preprocessing_config.enabled:
        return False

    # Read the context-level config if present.
    context_config = getattr(context, "config", None)

    # Handle dictionary-style stage selection.
    if isinstance(context_config, Mapping):
        # Extract the stages mapping.
        stages = context_config.get("stages")

        # Respect a dictionary-style stages.preprocessing flag when present.
        if isinstance(stages, Mapping) and "preprocessing" in stages:
            return bool(stages["preprocessing"])

    # Handle object-style stage selection.
    stages = getattr(context_config, "stages", None)

    # Respect object-style stages.preprocessing when present.
    if stages is not None and hasattr(stages, "preprocessing"):
        return bool(stages.preprocessing)

    # Default to enabled when no top-level stage selection is present.
    return True


def get_context_adata(context: object) -> ad.AnnData:
    """
    Retrieve AnnData from a PipelineContext-like object.

    Args:
        context: PipelineContext-like object.

    Returns:
        Active AnnData object.

    Raises:
        PreprocessingStageError: If AnnData is missing or invalid.
    """

    # Prefer the formal PipelineContext helper when present.
    require_adata = getattr(context, "require_adata", None)

    # Use require_adata when callable.
    if callable(require_adata):
        try:
            # Retrieve AnnData through the context helper.
            adata = require_adata()

        # Convert context errors into preprocessing stage errors.
        except Exception as error:
            raise PreprocessingStageError(
                "Preprocessing stage requires an AnnData object in context."
            ) from error

    # Fall back to a direct context.adata attribute.
    else:
        # Retrieve direct AnnData attribute.
        adata = getattr(context, "adata", None)

    # Validate AnnData type.
    if not isinstance(adata, ad.AnnData):
        raise PreprocessingStageError(
            "Preprocessing stage requires context.adata to be an AnnData object. "
            f"Received: {type(adata).__name__}."
        )

    # Return AnnData.
    return adata


def get_preprocessing_output_dir(context: object, output_subdir: str) -> Path:
    """
    Resolve the preprocessing stage output directory.

    Args:
        context: PipelineContext-like object with paths.results.
        output_subdir: Preprocessing subdirectory under results.

    Returns:
        Preprocessing artifact output directory.

    Raises:
        PreprocessingStageError: If context paths are missing or invalid.
    """

    # Reject empty output subdirectories.
    if not isinstance(output_subdir, str) or not output_subdir.strip():
        raise PreprocessingStageError(
            "PreprocessingStage output_subdir must be a non-empty string."
        )

    # Retrieve the context paths object.
    paths = getattr(context, "paths", None)

    # Require context paths.
    if paths is None:
        raise PreprocessingStageError(
            "Preprocessing stage requires context.paths with a results directory."
        )

    # Require a results directory on the paths object.
    if not hasattr(paths, "results"):
        raise PreprocessingStageError("Preprocessing stage requires context.paths.results.")

    # Resolve the results directory.
    results_dir = Path(paths.results)

    # Return the preprocessing output directory.
    return results_dir / output_subdir


def build_disabled_preprocessing_stage_result(
    *,
    adata: ad.AnnData,
    stage_name: str,
    preprocessing_config: PreprocessingConfig,
) -> StageResult:
    """
    Build a no-op StageResult for disabled preprocessing.

    Args:
        adata: Active AnnData object.
        stage_name: Stable stage name.
        preprocessing_config: Resolved preprocessing configuration.

    Returns:
        StageResult representing a disabled preprocessing no-op.
    """

    # Return a no-op stage result.
    return StageResult(
        adata=adata,
        artifacts=[],
        notes=["Preprocessing stage skipped because preprocessing is disabled."],
        warnings=[],
        metrics={
            "stage_name": stage_name,
            "enabled": False,
            "reason": "preprocessing_disabled",
        },
    )


def build_disabled_normalization_stage_result(
    *,
    adata: ad.AnnData,
    stage_name: str,
    preprocessing_config: PreprocessingConfig,
) -> StageResult:
    """
    Build a no-op StageResult for disabled normalization.

    Args:
        adata: Active AnnData object.
        stage_name: Stable stage name.
        preprocessing_config: Resolved preprocessing configuration.

    Returns:
        StageResult representing a disabled normalization no-op.
    """

    # Return a no-op stage result.
    return StageResult(
        adata=adata,
        artifacts=[],
        notes=["Preprocessing stage skipped because normalization is disabled."],
        warnings=[],
        metrics={
            "stage_name": stage_name,
            "enabled": True,
            "normalization_enabled": False,
            "reason": "normalization_disabled",
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
        },
    )


def write_preprocessing_summary(
    *,
    output_dir: Path,
    normalization_result: NormalizationResult,
    preprocessing_config: PreprocessingConfig,
    context: object,
    stage_name: str,
) -> Path:
    """
    Write preprocessing summary artifact.

    Args:
        output_dir: Preprocessing output directory.
        normalization_result: Normalization result.
        preprocessing_config: Preprocessing configuration.
        context: PipelineContext-like object.
        stage_name: Stable stage name.

    Returns:
        Path to written summary JSON.
    """

    # Build summary dictionary.
    summary = {
        "stage_name": stage_name,
        "run_id": str(getattr(context, "run_id", "cellquorum-run")),
        "random_seed": int(getattr(context, "random_seed", 1337)),
        "normalization": {
            "recipe": normalization_result.recipe,
            "input_layer": normalization_result.input_layer,
            "output_layer": normalization_result.output_layer,
            "preserve_counts_layer": normalization_result.preserve_counts_layer,
            "diagnostics": normalization_result.diagnostics,
            "warnings": normalization_result.warnings,
        },
    }

    # Write summary to JSON.
    summary_path = output_dir / "preprocessing_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Return summary path.
    return summary_path


def build_preprocessing_stage_notes(
    *,
    preprocessing_config: PreprocessingConfig,
    normalization_result: NormalizationResult,
    input_adata: ad.AnnData,
    output_adata: ad.AnnData,
) -> list[str]:
    """
    Build human-readable preprocessing stage notes.

    Args:
        preprocessing_config: Preprocessing configuration.
        normalization_result: Normalization result.
        input_adata: Input AnnData object.
        output_adata: Output AnnData object.

    Returns:
        Stage note strings.
    """

    # Initialize notes.
    notes = [
        f"Preprocessing completed with recipe '{normalization_result.recipe}'.",
        (
            f"Normalized expression written to layer '{normalization_result.output_layer}'; "
            f"raw counts preserved in layer '{normalization_result.preserve_counts_layer}'."
        ),
    ]

    # Return stage notes.
    return notes


def build_preprocessing_stage_metrics(
    *,
    stage_name: str,
    preprocessing_config: PreprocessingConfig,
    normalization_result: NormalizationResult,
    input_adata: ad.AnnData,
    output_adata: ad.AnnData,
) -> dict[str, object]:
    """
    Build structured preprocessing stage metrics for provenance.

    Args:
        stage_name: Stable stage name.
        preprocessing_config: Preprocessing configuration.
        normalization_result: Normalization result.
        input_adata: Input AnnData object.
        output_adata: Output AnnData object.

    Returns:
        JSON-friendly stage metrics.
    """

    # Return structured metrics.
    return {
        "stage_name": stage_name,
        "enabled": True,
        "recipe": normalization_result.recipe,
        "output_layer": normalization_result.output_layer,
        "preserve_counts_layer": normalization_result.preserve_counts_layer,
        "n_cells": int(output_adata.n_obs),
        "n_genes": int(output_adata.n_vars),
        "normalization_diagnostics": normalization_result.diagnostics,
    }


__all__ = [
    "PreprocessingStage",
    "PreprocessingStageError",
    "build_disabled_normalization_stage_result",
    "build_disabled_preprocessing_stage_result",
    "build_preprocessing_stage_metrics",
    "build_preprocessing_stage_notes",
    "coerce_preprocessing_config",
    "get_context_adata",
    "get_preprocessing_output_dir",
    "is_preprocessing_stage_enabled",
    "resolve_preprocessing_config",
    "write_preprocessing_summary",
]
