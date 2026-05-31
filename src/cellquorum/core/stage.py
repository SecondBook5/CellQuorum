"""Stage contracts for CellQuorum pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anndata as ad


@dataclass(frozen=True)
class StageArtifact:
    """
    Describe a file or directory produced by a pipeline stage.

    Every CellQuorum stage should report its outputs explicitly instead of
    silently writing files to arbitrary locations. This makes reporting,
    testing, provenance tracking, and reruns much easier to manage.

    Args:
        name: Stable artifact name used by reports and downstream stages.
        path: Filesystem path to the artifact.
        kind: Artifact type, such as csv, parquet, json, markdown, figure, h5ad, or directory.
        description: Human-readable explanation of what the artifact contains.
    """

    # Store the stable artifact name used by reports and downstream stages.
    name: str

    # Store the artifact path.
    path: Path

    # Store the artifact kind, such as csv, json, figure, h5ad, or directory.
    kind: str

    # Store a human-readable artifact description.
    description: str


@dataclass
class StageResult:
    """
    Store the complete result of one pipeline stage.

    A publication-grade stage should not only return an AnnData object. It should
    return the updated data object plus all artifacts, notes, warnings, and
    structured metrics needed to audit what happened.

    Args:
        adata: Updated AnnData object after stage execution.
        artifacts: Files or directories produced by the stage.
        notes: Non-critical observations that should appear in reports.
        warnings: Important caveats that should appear in reports and provenance.
        metrics: JSON-serializable structured metrics for summaries and reports.
    """

    # Store the updated AnnData object after the stage has run.
    adata: ad.AnnData

    # Store files or directories produced by the stage.
    artifacts: list[StageArtifact] = field(default_factory=list)

    # Store non-critical observations that should appear in reports.
    notes: list[str] = field(default_factory=list)

    # Store important caveats that should appear in reports and provenance.
    warnings: list[str] = field(default_factory=list)

    # Store JSON-serializable structured metrics for summaries and reports.
    metrics: dict[str, Any] = field(default_factory=dict)


class PipelineStage(Protocol):
    """
    Define the interface every CellQuorum stage must implement.

    Concrete stages should receive a PipelineContext and return a StageResult.
    The context type is kept as Any here to avoid circular imports while the
    execution spine is being bootstrapped.
    """

    # Store the stable stage name.
    name: str

    def run(self, context: Any) -> StageResult:
        """
        Execute the stage.

        Args:
            context: Pipeline execution context containing data, config, paths,
                backend registry, artifact manager, and provenance metadata.

        Returns:
            StageResult containing the updated AnnData object and stage outputs.
        """
        ...