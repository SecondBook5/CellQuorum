# Pipeline step (order=20): qc — context helpers for the QC stage.
"""Reading the pipeline context: configuration, the active object, output paths.

Split out of ``stage.py`` because none of it is about being a stage — it is the
adapter between a loosely-typed pipeline context and the typed values QC needs. Keeping
it here means the stage module can be read top to bottom as a description of what QC
does, which is the whole point of separating them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import anndata as ad

from cellquorum.stages.qc._errors import QCStageError
from cellquorum.stages.qc.config import QCConfig, validate_qc_config_dict


def resolve_qc_config(
    context: object,
    *,
    override: QCConfig | None = None,
) -> QCConfig:
    """
    Resolve the effective QC configuration for a stage run.

    Resolution order:

    1. explicit QCStage(config=...) override
    2. context.config when it is already a QCConfig
    3. context.config.qc when present
    4. context.config["qc"] when present
    5. QCConfig() defaults

    Args:
        context: PipelineContext-like object.
        override: Optional explicit QCConfig override.

    Returns:
        Resolved QCConfig.

    Raises:
        QCStageError: If the resolved QC config is invalid.
    """

    # Prefer the explicit stage-level override.
    if override is not None:
        # Validate override type.
        if not isinstance(override, QCConfig):
            raise QCStageError(
                "QCStage config override must be a QCConfig object. "
                f"Received: {type(override).__name__}."
            )

        # Return the override.
        return override

    # Read the context-level config if present.
    context_config = getattr(context, "config", None)

    # Accept context.config as a QCConfig directly.
    if isinstance(context_config, QCConfig):
        return context_config

    # Resolve context.config["qc"] for dictionary-like configs.
    if isinstance(context_config, Mapping) and "qc" in context_config:
        return coerce_qc_config(context_config["qc"])

    # Resolve context.config.qc for object-like configs. Bound via getattr because hasattr
    # cannot narrow a union that still includes None.
    qc_attribute = getattr(context_config, "qc", None)
    if qc_attribute is not None:
        return coerce_qc_config(qc_attribute)

    # Fall back to default QC configuration.
    return QCConfig()


def coerce_qc_config(value: object) -> QCConfig:
    """
    Coerce a candidate QC config value into QCConfig.

    Args:
        value: Candidate QC configuration value.

    Returns:
        Validated QCConfig.

    Raises:
        QCStageError: If the candidate cannot become QCConfig.
    """

    # Preserve QCConfig objects.
    if isinstance(value, QCConfig):
        return value

    # Validate dictionary-like QC configuration.
    if isinstance(value, Mapping):
        return validate_qc_config_dict(value)

    # Reject unsupported values.
    raise QCStageError(
        f"QC configuration must be a QCConfig object or mapping. Received: {type(value).__name__}."
    )


def is_qc_stage_enabled(context: object, qc_config: QCConfig) -> bool:
    """
    Return whether the QC stage should execute.

    The stage is enabled only when QCConfig.enabled is true and any top-level
    context.config.stages.qc flag is also true.

    Args:
        context: PipelineContext-like object.
        qc_config: Resolved QC configuration.

    Returns:
        True when QC should run, otherwise False.
    """

    # Respect the QC module-level enabled flag first.
    if not qc_config.enabled:
        return False

    # Read the context-level config if present.
    context_config = getattr(context, "config", None)

    # Handle dictionary-style stage selection.
    if isinstance(context_config, Mapping):
        # Extract the stages mapping.
        stages = context_config.get("stages")

        # Respect a dictionary-style stages.qc flag when present.
        if isinstance(stages, Mapping) and "qc" in stages:
            return bool(stages["qc"])

    # Handle object-style stage selection.
    stages = getattr(context_config, "stages", None)

    # Respect object-style stages.qc when present.
    if stages is not None and hasattr(stages, "qc"):
        return bool(stages.qc)

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
        QCStageError: If AnnData is missing or invalid.
    """

    # Prefer the formal PipelineContext helper when present.
    require_adata = getattr(context, "require_adata", None)

    # Use require_adata when callable.
    if callable(require_adata):
        try:
            # Retrieve AnnData through the context helper.
            adata = require_adata()

        # Convert context errors into QC stage errors.
        except Exception as error:
            raise QCStageError("QC stage requires an AnnData object in context.") from error

    # Fall back to a direct context.adata attribute.
    else:
        # Retrieve direct AnnData attribute.
        adata = getattr(context, "adata", None)

    # Validate AnnData type.
    if not isinstance(adata, ad.AnnData):
        raise QCStageError(
            "QC stage requires context.adata to be an AnnData object. "
            f"Received: {type(adata).__name__}."
        )

    # Return AnnData.
    return adata


def get_qc_output_dir(context: object, output_subdir: str) -> Path:
    """
    Resolve the QC stage output directory.

    Args:
        context: PipelineContext-like object with paths.results.
        output_subdir: QC subdirectory under results.

    Returns:
        QC artifact output directory.

    Raises:
        QCStageError: If context paths are missing or invalid.
    """

    # Reject empty output subdirectories.
    if not isinstance(output_subdir, str) or not output_subdir.strip():
        raise QCStageError("QCStage output_subdir must be a non-empty string.")

    # Retrieve the context paths object.
    paths = getattr(context, "paths", None)

    # Require context paths.
    if paths is None:
        raise QCStageError("QC stage requires context.paths with a results directory.")

    # Require a results directory on the paths object.
    if not hasattr(paths, "results"):
        raise QCStageError("QC stage requires context.paths.results.")

    # Resolve the results directory.
    results_dir = Path(paths.results)

    # Return the QC output directory.
    return results_dir / output_subdir
