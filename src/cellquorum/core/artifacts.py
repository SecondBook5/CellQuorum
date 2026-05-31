"""Artifact writing utilities for CellQuorum pipeline stages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from cellquorum.core.stage import StageArtifact


@dataclass
class ArtifactManager:
    """
    Manage standardized artifact creation for a CellQuorum run.

    Pipeline stages should not write files with ad hoc logic scattered across
    the codebase. This manager centralizes common artifact writing patterns for
    tables, JSON, Markdown, and plain text. Each writer returns a StageArtifact
    object so reports, provenance tracking, and tests can verify exactly what a
    stage produced.

    Args:
        root: Root directory where this manager writes artifacts.
        artifacts: List of artifacts written through this manager.
    """

    # Store the root directory where artifacts are written.
    root: Path

    # Store all artifacts written through this manager.
    artifacts: list[StageArtifact] = field(default_factory=list)

    @classmethod
    def from_root(cls, root: str | Path) -> "ArtifactManager":
        """
        Build an ArtifactManager from a root directory.

        The root path is expanded and resolved so all downstream artifact paths
        are stable and absolute. The directory is not created here because some
        callers may want to validate paths before touching the filesystem.

        Args:
            root: Root directory where artifacts should be written.

        Returns:
            ArtifactManager with a resolved artifact root.
        """

        # Resolve the root directory to an absolute path.
        resolved_root = Path(root).expanduser().resolve()

        # Return a manager configured for the resolved root.
        return cls(root=resolved_root)

    def ensure_root(self) -> None:
        """
        Create the artifact root directory if it does not already exist.

        This method is intentionally separate from initialization so callers can
        decide when filesystem side effects should occur.
        """

        # Create the artifact root and any missing parent directories.
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, relative_path: str | Path) -> Path:
        """
        Resolve an artifact path relative to the manager root.

        Absolute paths are rejected deliberately. CellQuorum artifacts should
        remain inside the declared run directory unless a future explicit export
        mechanism is added. This prevents accidental writes to unexpected
        filesystem locations.

        Args:
            relative_path: Relative artifact path under the manager root.

        Returns:
            Absolute path to the artifact location.

        Raises:
            ValueError: If `relative_path` is absolute.
        """

        # Convert the incoming path to a Path object.
        path = Path(relative_path)

        # Reject absolute artifact paths to keep outputs inside the run root.
        if path.is_absolute():
            raise ValueError(
                "Artifact paths must be relative to the ArtifactManager root. "
                f"Received absolute path: {path}"
            )

        # Return the artifact path resolved under the root directory.
        return (self.root / path).resolve()

    def register(
        self,
        *,
        name: str,
        relative_path: str | Path,
        kind: str,
        description: str,
    ) -> StageArtifact:
        """
        Register an artifact that already exists or will be written elsewhere.

        This method is useful when a specialized stage writes an object through a
        domain-specific library but still needs to expose that artifact to
        CellQuorum's reporting and provenance systems.

        Args:
            name: Stable artifact name.
            relative_path: Relative path under the artifact manager root.
            kind: Artifact type, such as csv, parquet, json, markdown, figure,
                h5ad, or directory.
            description: Human-readable artifact description.

        Returns:
            StageArtifact describing the registered artifact.
        """

        # Resolve the artifact path under the manager root.
        path = self.resolve_path(relative_path)

        # Create a structured artifact record.
        artifact = StageArtifact(
            name=name,
            path=path,
            kind=kind,
            description=description,
        )

        # Store the artifact record for this manager.
        self.artifacts.append(artifact)

        # Return the artifact record to the caller.
        return artifact

    def write_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        name: str,
        relative_path: str | Path,
        description: str,
        index: bool = False,
    ) -> StageArtifact:
        """
        Write a pandas DataFrame artifact as CSV or Parquet.

        The output format is inferred from the file suffix. CSV and Parquet are
        supported because they cover human-readable inspection and efficient
        large-table storage. Unsupported suffixes fail early with a clear error.

        Args:
            dataframe: DataFrame to write.
            name: Stable artifact name.
            relative_path: Relative artifact path ending in .csv or .parquet.
            description: Human-readable artifact description.
            index: Whether to write the DataFrame index.

        Returns:
            StageArtifact describing the written table.

        Raises:
            TypeError: If `dataframe` is not a pandas DataFrame.
            ValueError: If the target suffix is unsupported.
        """

        # Validate that the caller provided a pandas DataFrame.
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "write_dataframe expected a pandas DataFrame. "
                f"Received: {type(dataframe).__name__}"
            )

        # Resolve the target path.
        path = self.resolve_path(relative_path)

        # Create the parent directory if needed.
        path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize the suffix for format dispatch.
        suffix = path.suffix.lower()

        # Write CSV output when requested.
        if suffix == ".csv":
            dataframe.to_csv(path, index=index)

        # Write Parquet output when requested.
        elif suffix == ".parquet":
            dataframe.to_parquet(path, index=index)

        # Reject unsupported table formats.
        else:
            raise ValueError(
                "Unsupported dataframe artifact format. "
                "Use a path ending in '.csv' or '.parquet'. "
                f"Received: {path.name}"
            )

        # Register and return the written artifact.
        return self.register(
            name=name,
            relative_path=relative_path,
            kind=suffix.removeprefix("."),
            description=description,
        )

    def write_json(
        self,
        payload: dict[str, Any] | list[Any],
        *,
        name: str,
        relative_path: str | Path,
        description: str,
        indent: int = 2,
    ) -> StageArtifact:
        """
        Write a JSON artifact.

        JSON artifacts are used for structured metadata, stage summaries,
        warnings, backend status, provenance, and report context. Payloads are
        restricted to dictionaries or lists so accidental scalar dumps are caught
        early.

        Args:
            payload: JSON-serializable dictionary or list.
            name: Stable artifact name.
            relative_path: Relative artifact path ending in .json.
            description: Human-readable artifact description.
            indent: Number of spaces used for pretty-printed JSON.

        Returns:
            StageArtifact describing the written JSON file.

        Raises:
            TypeError: If payload is not a dictionary or list.
            ValueError: If the target file does not end in .json.
        """

        # Validate that the payload has a top-level JSON container type.
        if not isinstance(payload, (dict, list)):
            raise TypeError(
                "write_json expected a dictionary or list payload. "
                f"Received: {type(payload).__name__}"
            )

        # Resolve the target path.
        path = self.resolve_path(relative_path)

        # Validate the expected JSON suffix.
        if path.suffix.lower() != ".json":
            raise ValueError(
                "JSON artifacts must use a '.json' suffix. "
                f"Received: {path.name}"
            )

        # Create the parent directory if needed.
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write the JSON payload with stable formatting.
        path.write_text(json.dumps(payload, indent=indent, sort_keys=True), encoding="utf-8")

        # Register and return the written artifact.
        return self.register(
            name=name,
            relative_path=relative_path,
            kind="json",
            description=description,
        )

    def write_text(
        self,
        text: str,
        *,
        name: str,
        relative_path: str | Path,
        kind: str,
        description: str,
    ) -> StageArtifact:
        """
        Write a plain-text-like artifact.

        This writer supports Markdown, plain text, and other text artifacts where
        the caller controls the exact content. It intentionally validates the
        payload as a string so accidental non-text objects do not get silently
        coerced into unclear output.

        Args:
            text: Text content to write.
            name: Stable artifact name.
            relative_path: Relative artifact path.
            kind: Artifact kind, such as markdown, text, html, css, or log.
            description: Human-readable artifact description.

        Returns:
            StageArtifact describing the written text artifact.

        Raises:
            TypeError: If text is not a string.
        """

        # Validate that the text payload is actually a string.
        if not isinstance(text, str):
            raise TypeError(
                "write_text expected a string payload. "
                f"Received: {type(text).__name__}"
            )

        # Resolve the target path.
        path = self.resolve_path(relative_path)

        # Create the parent directory if needed.
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write the text payload as UTF-8.
        path.write_text(text, encoding="utf-8")

        # Register and return the written artifact.
        return self.register(
            name=name,
            relative_path=relative_path,
            kind=kind,
            description=description,
        )

    def write_markdown(
        self,
        markdown: str,
        *,
        name: str,
        relative_path: str | Path,
        description: str,
    ) -> StageArtifact:
        """
        Write a Markdown artifact.

        Markdown is used for human-readable report fragments and methods text.
        This wrapper exists so stage code can be explicit about report artifacts
        while reusing the common text writer.

        Args:
            markdown: Markdown content to write.
            name: Stable artifact name.
            relative_path: Relative artifact path ending in .md.
            description: Human-readable artifact description.

        Returns:
            StageArtifact describing the written Markdown file.

        Raises:
            ValueError: If the target file does not end in .md.
        """

        # Resolve the target path to validate its suffix.
        path = self.resolve_path(relative_path)

        # Validate the expected Markdown suffix.
        if path.suffix.lower() != ".md":
            raise ValueError(
                "Markdown artifacts must use a '.md' suffix. "
                f"Received: {path.name}"
            )

        # Write the Markdown artifact through the generic text writer.
        return self.write_text(
            markdown,
            name=name,
            relative_path=relative_path,
            kind="markdown",
            description=description,
        )

    def to_manifest_dataframe(self) -> pd.DataFrame:
        """
        Convert registered artifacts into a manifest DataFrame.

        The artifact manifest is one of the most important outputs of a
        publication-grade pipeline because it provides a machine-readable index
        of all files produced by a run.

        Returns:
            DataFrame with artifact name, path, kind, and description.
        """

        # Convert each artifact record to a dictionary row.
        rows = [
            {
                "name": artifact.name,
                "path": str(artifact.path),
                "kind": artifact.kind,
                "description": artifact.description,
            }
            for artifact in self.artifacts
        ]

        # Return the artifact manifest table.
        return pd.DataFrame(rows, columns=["name", "path", "kind", "description"])

    def write_manifest(
        self,
        *,
        relative_path: str | Path = "provenance/artifact_manifest.csv",
    ) -> StageArtifact:
        """
        Write a CSV manifest of all artifacts registered by this manager.

        Args:
            relative_path: Relative output path for the artifact manifest.

        Returns:
            StageArtifact describing the written artifact manifest.
        """

        # Build the current artifact manifest table.
        manifest = self.to_manifest_dataframe()

        # Write the manifest as a CSV artifact.
        return self.write_dataframe(
            manifest,
            name="artifact_manifest",
            relative_path=relative_path,
            description="Machine-readable index of artifacts produced during the run.",
            index=False,
        )