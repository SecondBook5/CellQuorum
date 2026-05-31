"""Shared exception classes for CellQuorum."""

from __future__ import annotations


class CellQuorumError(Exception):
    """
    Base exception for all CellQuorum-specific errors.

    This base class gives the package one shared exception root. External callers
    can catch `CellQuorumError` when they want to handle any expected CellQuorum
    failure without accidentally catching unrelated Python exceptions.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the base CellQuorum exception.

        Args:
            message: Human-readable explanation of the error.
        """

        # Initialize the parent exception with the supplied message.
        super().__init__(message)


class CellQuorumConfigError(CellQuorumError):
    """
    Report configuration loading or validation failures.

    Use this when a YAML file, dictionary, or validated configuration object is
    missing required values, contains invalid settings, or cannot be interpreted
    safely by the pipeline.
    """


class CellQuorumExecutionError(CellQuorumError):
    """
    Report high-level pipeline execution failures.

    Use this for failures that occur while initializing or running a CellQuorum
    workflow, especially when the failure is not specific to one analysis stage.
    """


class CellQuorumStageError(CellQuorumExecutionError):
    """
    Report failures from a specific pipeline stage.

    Stage errors should be used when QC, preprocessing, annotation, differential
    analysis, reporting, or another named stage cannot complete correctly.
    """

    def __init__(self, stage_name: str, message: str) -> None:
        """
        Initialize a stage-specific execution error.

        Args:
            stage_name: Stable name of the stage that failed.
            message: Human-readable explanation of the stage failure.
        """

        # Store the failed stage name for structured handling.
        self.stage_name = stage_name

        # Initialize the parent exception with a stage-prefixed message.
        super().__init__(f"Stage '{stage_name}' failed: {message}")


class CellQuorumBackendError(CellQuorumError):
    """
    Report backend discovery or backend execution failures.

    Use this when Python, R, Rscript, GPU, RAPIDS, or another backend cannot be
    found, initialized, selected, or used safely.
    """


class CellQuorumProvenanceError(CellQuorumError):
    """
    Report provenance writing or provenance validation failures.

    Use this when CellQuorum cannot write, serialize, register, or validate
    provenance artifacts such as resolved configs, execution records, manifests,
    plans, metrics, or artifact indexes.
    """


class CellQuorumDataError(CellQuorumError):
    """
    Report input data loading or data validation failures.

    Use this when input data files, AnnData objects, matrices, metadata tables,
    feature annotations, or sample manifests cannot be loaded or do not satisfy
    the assumptions required by downstream methods.
    """


class CellQuorumManifestError(CellQuorumDataError):
    """
    Report manifest loading or manifest validation failures.

    Use this when a manifest file is missing, malformed, contains duplicate
    sample identifiers, lacks required columns, or contains unusable sample paths.
    """


class CellQuorumReportError(CellQuorumError):
    """
    Report report-generation failures.

    Use this when markdown, HTML, figure panels, tables, templates, or final
    report artifacts cannot be rendered correctly.
    """


__all__ = [
    "CellQuorumBackendError",
    "CellQuorumConfigError",
    "CellQuorumDataError",
    "CellQuorumError",
    "CellQuorumExecutionError",
    "CellQuorumManifestError",
    "CellQuorumProvenanceError",
    "CellQuorumReportError",
    "CellQuorumStageError",
]
