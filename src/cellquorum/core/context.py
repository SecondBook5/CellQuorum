"""Pipeline execution context for CellQuorum stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd


@dataclass(frozen=True)
class PipelinePaths:
    """
    Store all standardized output directories for one CellQuorum run.

    CellQuorum stages should not invent their own output locations. This object
    gives every stage the same directory contract so results, figures, reports,
    intermediate objects, logs, and provenance files are written predictably.
    Keeping these paths centralized also makes report generation and testing
    much easier because every artifact belongs to a known namespace.

    Args:
        root: Root directory for the current CellQuorum run.
        results: Directory for machine-readable result tables.
        figures: Directory for generated figures.
        reports: Directory for human-readable reports.
        objects: Directory for AnnData and other serialized analysis objects.
        provenance: Directory for resolved configs, versions, stage manifests,
            backend status files, and reproducibility metadata.
        logs: Directory for execution logs and warnings.
        scratch: Directory for temporary stage-specific files.
    """

    # Store the root directory for the current run.
    root: Path

    # Store the directory for machine-readable result tables.
    results: Path

    # Store the directory for generated figures.
    figures: Path

    # Store the directory for human-readable reports.
    reports: Path

    # Store the directory for serialized analysis objects.
    objects: Path

    # Store the directory for reproducibility and provenance metadata.
    provenance: Path

    # Store the directory for logs and warning records.
    logs: Path

    # Store the directory for temporary stage-specific files.
    scratch: Path

    @classmethod
    def from_output_dir(cls, output_dir: str | Path) -> PipelinePaths:
        """
        Build the standard CellQuorum run directory layout.

        This constructor turns a single user-provided output directory into the
        full directory structure expected by the pipeline. It does not create the
        directories itself, because some callers may want to inspect or validate
        paths before touching the filesystem. Use `ensure_directories()` when
        filesystem creation is desired.

        Args:
            output_dir: Root directory for the current CellQuorum run.

        Returns:
            PipelinePaths object with all standard run directories resolved.
        """

        # Resolve the root output directory so downstream paths are absolute.
        root = Path(output_dir).expanduser().resolve()

        # Return the standardized run directory layout.
        return cls(
            root=root,
            results=root / "results",
            figures=root / "figures",
            reports=root / "reports",
            objects=root / "objects",
            provenance=root / "provenance",
            logs=root / "logs",
            scratch=root / "scratch",
        )

    def ensure_directories(self) -> None:
        """
        Create all standard run directories if they do not already exist.

        This method is intentionally explicit and small. Pipeline setup code can
        call it once before stage execution, and individual stages can then rely
        on the standard directories existing.
        """

        # Iterate over every directory in the standardized run layout.
        for directory in (
            self.root,
            self.results,
            self.figures,
            self.reports,
            self.objects,
            self.provenance,
            self.logs,
            self.scratch,
        ):
            # Create each directory and any missing parent directories.
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineContext:
    """
    Store shared runtime state for CellQuorum pipeline stages.

    Pipeline stages should receive a single context object instead of a long list
    of loosely related arguments. The context keeps the active AnnData object,
    validated configuration, manifest table, run paths, backend registry, and
    runtime metadata together. This makes stage interfaces stable as the package
    grows from basic QC into R, GPU, molecular inference, and report generation.

    Args:
        config: Validated CellQuorum runtime configuration.
        paths: Standardized output paths for the run.
        adata: Active AnnData object, if already loaded.
        manifest: Sample-level manifest table, if already loaded.
        backend_registry: Backend registry used to dispatch Python, R, GPU, and
            external tool execution.
        run_id: Stable identifier for the current run.
        random_seed: Random seed used by stochastic stages.
        metadata: Additional JSON-serializable runtime metadata.
    """

    # Store the validated runtime configuration.
    config: Any

    # Store the standardized run directories.
    paths: PipelinePaths

    # Store the active AnnData object, if already loaded.
    adata: ad.AnnData | None = None

    # Store the sample manifest table, if already loaded.
    manifest: pd.DataFrame | None = None

    # Store the backend registry for Python, R, GPU, and external execution.
    backend_registry: Any | None = None

    # Store a stable run identifier.
    run_id: str = "cellquorum-run"

    # Store the random seed used by stochastic stages.
    random_seed: int = 1337

    # Store additional runtime metadata.
    metadata: dict[str, Any] = field(default_factory=dict)

    def require_adata(self) -> ad.AnnData:
        """
        Return the active AnnData object or raise a clear error.

        Some stages require an AnnData object to already exist. This helper
        prevents downstream code from failing with vague `NoneType` errors and
        instead reports exactly what is missing from the pipeline context.

        Returns:
            Active AnnData object.

        Raises:
            RuntimeError: If the context does not currently contain AnnData.
        """

        # Check whether an AnnData object is available.
        if self.adata is None:
            # Raise a specific error that explains the missing context state.
            raise RuntimeError(
                "PipelineContext does not contain an AnnData object. "
                "Run an ingestion/loading stage before executing this stage."
            )

        # Return the active AnnData object.
        return self.adata

    def require_manifest(self) -> pd.DataFrame:
        """
        Return the active sample manifest or raise a clear error.

        Several later stages need sample, donor, condition, batch, or perturbation
        metadata. This helper keeps those failures explicit instead of allowing
        unclear downstream pandas errors.

        Returns:
            Active manifest DataFrame.

        Raises:
            RuntimeError: If the context does not currently contain a manifest.
        """

        # Check whether a manifest table is available.
        if self.manifest is None:
            # Raise a specific error that explains the missing context state.
            raise RuntimeError(
                "PipelineContext does not contain a manifest table. "
                "Load and validate a manifest before executing this stage."
            )

        # Return the active manifest table.
        return self.manifest

    def with_adata(self, adata: ad.AnnData) -> PipelineContext:
        """
        Return a shallow context copy with an updated AnnData object.

        This helper supports functional-style stage chaining. A stage can return
        a context with updated data while preserving configuration, paths,
        backend registry, run identifiers, and metadata.

        Args:
            adata: Updated AnnData object to attach to the context.

        Returns:
            PipelineContext with the updated AnnData object.
        """

        # Return a new context with the updated AnnData object.
        return PipelineContext(
            config=self.config,
            paths=self.paths,
            adata=adata,
            manifest=self.manifest,
            backend_registry=self.backend_registry,
            run_id=self.run_id,
            random_seed=self.random_seed,
            metadata=dict(self.metadata),
        )
