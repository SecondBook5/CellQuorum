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
    """Resolve the effective QC configuration for a stage run.

    Args:
        context: PipelineContext-like object.
        override: Optional explicit QCConfig override.

    Returns:
        Resolved QCConfig.

    Raises:
        QCStageError: If the resolved QC config is invalid.
    """

    if override is not None:
        if not isinstance(override, QCConfig):
            raise QCStageError(
                "QCStage config override must be a QCConfig object. "
                f"Received: {type(override).__name__}."
            )

        return override

    context_config = getattr(context, "config", None)

    if isinstance(context_config, QCConfig):
        return context_config

    if isinstance(context_config, Mapping) and "qc" in context_config:
        return coerce_qc_config(context_config["qc"])

    qc_attribute = getattr(context_config, "qc", None)
    if qc_attribute is not None:
        return coerce_qc_config(qc_attribute)

    return QCConfig()


def coerce_qc_config(value: object) -> QCConfig:
    """Coerce a candidate QC config value into QCConfig.

    Args:
        value: Candidate QC configuration value.

    Returns:
        Validated QCConfig.

    Raises:
        QCStageError: If the candidate cannot become QCConfig.
    """

    if isinstance(value, QCConfig):
        return value

    if isinstance(value, Mapping):
        return validate_qc_config_dict(value)

    raise QCStageError(
        f"QC configuration must be a QCConfig object or mapping. Received: {type(value).__name__}."
    )


def is_qc_stage_enabled(context: object, qc_config: QCConfig) -> bool:
    """Return whether the QC stage should execute.

    Args:
        context: PipelineContext-like object.
        qc_config: Resolved QC configuration.

    Returns:
        True when QC should run, otherwise False.
    """

    if not qc_config.enabled:
        return False

    context_config = getattr(context, "config", None)

    if isinstance(context_config, Mapping):
        stages = context_config.get("stages")

        if isinstance(stages, Mapping) and "qc" in stages:
            return bool(stages["qc"])

    stages = getattr(context_config, "stages", None)

    if stages is not None and hasattr(stages, "qc"):
        return bool(stages.qc)

    return True


def get_context_adata(context: object) -> ad.AnnData:
    """Retrieve AnnData from a PipelineContext-like object.

    Args:
        context: PipelineContext-like object.

    Returns:
        Active AnnData object.

    Raises:
        QCStageError: If AnnData is missing or invalid.
    """

    require_adata = getattr(context, "require_adata", None)

    if callable(require_adata):
        try:
            adata = require_adata()

        except Exception as error:
            raise QCStageError("QC stage requires an AnnData object in context.") from error

    else:
        adata = getattr(context, "adata", None)

    if not isinstance(adata, ad.AnnData):
        raise QCStageError(
            "QC stage requires context.adata to be an AnnData object. "
            f"Received: {type(adata).__name__}."
        )

    return adata


def get_qc_output_dir(context: object, output_subdir: str) -> Path:
    """Resolve the QC stage output directory.

    Args:
        context: PipelineContext-like object with paths.results.
        output_subdir: QC subdirectory under results.

    Returns:
        QC artifact output directory.

    Raises:
        QCStageError: If context paths are missing or invalid.
    """

    if not isinstance(output_subdir, str) or not output_subdir.strip():
        raise QCStageError("QCStage output_subdir must be a non-empty string.")

    paths = getattr(context, "paths", None)

    if paths is None:
        raise QCStageError("QC stage requires context.paths with a results directory.")

    if not hasattr(paths, "results"):
        raise QCStageError("QC stage requires context.paths.results.")

    results_dir = Path(paths.results)

    return results_dir / output_subdir
