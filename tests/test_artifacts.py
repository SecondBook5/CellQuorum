"""Tests for CellQuorum artifact writing utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cellquorum.core.artifacts import ArtifactManager


def test_artifact_manager_from_root_resolves_root(tmp_path: Path) -> None:
    """
    Verify that ArtifactManager resolves its root directory.

    The artifact manager should store an absolute root path so downstream stages
    write outputs into stable, predictable run directories.
    """

    # Create an artifact manager from a temporary root path.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Confirm the manager root is resolved to an absolute path.
    assert manager.root == (tmp_path / "run").resolve()


def test_artifact_manager_ensure_root_creates_directory(tmp_path: Path) -> None:
    """
    Verify that ArtifactManager can create its root directory.

    The manager intentionally does not create directories during construction,
    so filesystem side effects remain explicit and easy to test.
    """

    # Create an artifact manager for a directory that does not exist yet.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Confirm the root does not exist before explicit creation.
    assert not manager.root.exists()

    # Create the root directory.
    manager.ensure_root()

    # Confirm the root directory now exists.
    assert manager.root.exists()

    # Confirm the root path is a directory.
    assert manager.root.is_dir()


def test_resolve_path_rejects_absolute_paths(tmp_path: Path) -> None:
    """
    Verify that artifact paths must stay inside the run root.

    Absolute artifact paths are rejected so stages cannot accidentally write
    outputs outside the controlled CellQuorum run directory.
    """

    # Create an artifact manager rooted in the temporary directory.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Build an absolute path that should be rejected.
    absolute_path = tmp_path / "outside.csv"

    # Confirm absolute artifact paths raise a clear error.
    with pytest.raises(ValueError, match="must be relative"):
        manager.resolve_path(absolute_path)


def test_register_adds_artifact_metadata(tmp_path: Path) -> None:
    """
    Verify that manually registered artifacts are tracked.

    Some stages may write specialized outputs through external libraries. The
    register method lets those files still appear in reports and provenance.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Register an artifact under the manager root.
    artifact = manager.register(
        name="example_table",
        relative_path="results/example.csv",
        kind="csv",
        description="Example registered table.",
    )

    # Confirm the returned artifact has the expected name.
    assert artifact.name == "example_table"

    # Confirm the returned artifact path is rooted under the manager root.
    assert artifact.path == (tmp_path / "run" / "results" / "example.csv").resolve()

    # Confirm the artifact was stored by the manager.
    assert manager.artifacts == [artifact]


def test_write_dataframe_writes_csv_and_registers_artifact(tmp_path: Path) -> None:
    """
    Verify that DataFrame artifacts can be written as CSV files.

    CSV output is the default human-readable table format for CellQuorum result
    artifacts, so this writer needs to create the file and register metadata.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Create a small DataFrame artifact.
    dataframe = pd.DataFrame(
        {
            "cell_id": ["cell_1", "cell_2"],
            "qc_pass": [True, False],
        }
    )

    # Write the DataFrame as a CSV artifact.
    artifact = manager.write_dataframe(
        dataframe,
        name="qc_decisions",
        relative_path="results/qc/qc_decisions.csv",
        description="Cell-level QC decision table.",
    )

    # Confirm the CSV file was created.
    assert artifact.path.exists()

    # Read the written CSV file back from disk.
    loaded = pd.read_csv(artifact.path)

    # Confirm the written table matches the input table.
    pd.testing.assert_frame_equal(loaded, dataframe)

    # Confirm the artifact kind was inferred from the suffix.
    assert artifact.kind == "csv"

    # Confirm the artifact is tracked by the manager.
    assert manager.artifacts[0].name == "qc_decisions"


def test_write_dataframe_rejects_non_dataframe_payload(tmp_path: Path) -> None:
    """
    Verify that write_dataframe rejects non-DataFrame payloads.

    This protects stages from accidentally writing unclear outputs when a method
    returns an unexpected object type.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Confirm that passing a non-DataFrame object raises a clear error.
    with pytest.raises(TypeError, match="expected a pandas DataFrame"):
        manager.write_dataframe(
            ["not", "a", "dataframe"],  # type: ignore[arg-type]
            name="bad_table",
            relative_path="results/bad.csv",
            description="Invalid table.",
        )


def test_write_dataframe_rejects_unsupported_suffix(tmp_path: Path) -> None:
    """
    Verify that write_dataframe rejects unsupported table suffixes.

    CellQuorum should fail early when a stage requests an unsupported artifact
    format instead of silently writing a misleading file.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Create a small DataFrame.
    dataframe = pd.DataFrame({"x": [1, 2]})

    # Confirm unsupported table formats raise a clear error.
    with pytest.raises(ValueError, match="Unsupported dataframe artifact format"):
        manager.write_dataframe(
            dataframe,
            name="bad_table",
            relative_path="results/bad.txt",
            description="Invalid table format.",
        )


def test_write_json_writes_payload_and_registers_artifact(tmp_path: Path) -> None:
    """
    Verify that JSON artifacts are written and registered.

    JSON artifacts are used for structured stage summaries, backend status,
    warnings, and provenance metadata.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Create a JSON-serializable payload.
    payload = {"stage": "qc", "n_cells": 10}

    # Write the payload as a JSON artifact.
    artifact = manager.write_json(
        payload,
        name="qc_summary",
        relative_path="results/qc/qc_summary.json",
        description="Structured QC summary.",
    )

    # Confirm the JSON file was created.
    assert artifact.path.exists()

    # Load the written JSON payload.
    loaded = json.loads(artifact.path.read_text(encoding="utf-8"))

    # Confirm the written JSON payload matches the input payload.
    assert loaded == payload

    # Confirm the artifact kind is JSON.
    assert artifact.kind == "json"


def test_write_json_rejects_scalar_payload(tmp_path: Path) -> None:
    """
    Verify that write_json rejects scalar payloads.

    Stage summaries should use structured dictionaries or lists, not ambiguous
    scalar outputs.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Confirm scalar JSON payloads raise a clear error.
    with pytest.raises(TypeError, match="expected a dictionary or list"):
        manager.write_json(
            "not structured",  # type: ignore[arg-type]
            name="bad_json",
            relative_path="results/bad.json",
            description="Invalid JSON payload.",
        )


def test_write_json_rejects_non_json_suffix(tmp_path: Path) -> None:
    """
    Verify that write_json requires a .json suffix.

    Explicit suffix validation prevents report and provenance artifacts from
    being saved with misleading filenames.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Confirm a non-JSON suffix raises a clear error.
    with pytest.raises(ValueError, match="must use a '.json' suffix"):
        manager.write_json(
            {"ok": True},
            name="bad_json",
            relative_path="results/bad.txt",
            description="Invalid JSON suffix.",
        )


def test_write_text_writes_payload_and_registers_artifact(tmp_path: Path) -> None:
    """
    Verify that text artifacts are written and registered.

    Generic text output supports logs, plain text summaries, HTML fragments, and
    other report-adjacent artifacts.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Write a text artifact.
    artifact = manager.write_text(
        "CellQuorum report fragment",
        name="report_fragment",
        relative_path="reports/fragment.txt",
        kind="text",
        description="Example report fragment.",
    )

    # Confirm the text file was created.
    assert artifact.path.exists()

    # Confirm the written text matches the input payload.
    assert artifact.path.read_text(encoding="utf-8") == "CellQuorum report fragment"

    # Confirm the artifact kind was preserved.
    assert artifact.kind == "text"


def test_write_text_rejects_non_string_payload(tmp_path: Path) -> None:
    """
    Verify that write_text rejects non-string payloads.

    This avoids silently converting arbitrary Python objects into unclear report
    fragments.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Confirm a non-string text payload raises a clear error.
    with pytest.raises(TypeError, match="expected a string payload"):
        manager.write_text(
            {"not": "text"},  # type: ignore[arg-type]
            name="bad_text",
            relative_path="reports/bad.txt",
            kind="text",
            description="Invalid text payload.",
        )


def test_write_markdown_requires_md_suffix(tmp_path: Path) -> None:
    """
    Verify that Markdown artifacts require a .md suffix.

    Markdown report fragments should have consistent filenames so the final
    report builder can find and render them reliably.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Confirm a non-Markdown suffix raises a clear error.
    with pytest.raises(ValueError, match="must use a '.md' suffix"):
        manager.write_markdown(
            "# Bad suffix",
            name="bad_markdown",
            relative_path="reports/bad.txt",
            description="Invalid Markdown suffix.",
        )


def test_write_markdown_writes_payload_and_registers_artifact(tmp_path: Path) -> None:
    """
    Verify that Markdown artifacts are written and registered.

    Stage-specific Markdown fragments will later feed into the final analysis
    report.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Write a Markdown artifact.
    artifact = manager.write_markdown(
        "# QC Report\n\nQC completed.",
        name="qc_report",
        relative_path="reports/qc_report.md",
        description="Human-readable QC report.",
    )

    # Confirm the Markdown file was created.
    assert artifact.path.exists()

    # Confirm the written Markdown begins with the expected heading.
    assert artifact.path.read_text(encoding="utf-8").startswith("# QC Report")

    # Confirm the artifact kind is Markdown.
    assert artifact.kind == "markdown"


def test_to_manifest_dataframe_returns_registered_artifacts(tmp_path: Path) -> None:
    """
    Verify that registered artifacts can be converted into a manifest table.

    The artifact manifest is a central provenance object, so the manager must be
    able to represent all registered artifacts as a stable table.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Register one artifact.
    manager.register(
        name="example",
        relative_path="results/example.csv",
        kind="csv",
        description="Example artifact.",
    )

    # Convert registered artifacts to a manifest DataFrame.
    manifest = manager.to_manifest_dataframe()

    # Confirm the manifest has one row.
    assert len(manifest) == 1

    # Confirm the expected columns are present in stable order.
    assert list(manifest.columns) == ["name", "path", "kind", "description"]

    # Confirm the artifact name is present in the manifest.
    assert manifest.loc[0, "name"] == "example"


def test_write_manifest_writes_artifact_manifest_csv(tmp_path: Path) -> None:
    """
    Verify that ArtifactManager writes an artifact manifest CSV.

    The manifest gives reports and users a machine-readable index of all outputs
    produced by a run.
    """

    # Create an artifact manager.
    manager = ArtifactManager.from_root(tmp_path / "run")

    # Register one artifact before writing the manifest.
    manager.register(
        name="example",
        relative_path="results/example.csv",
        kind="csv",
        description="Example artifact.",
    )

    # Write the artifact manifest.
    manifest_artifact = manager.write_manifest()

    # Confirm the manifest file exists.
    assert manifest_artifact.path.exists()

    # Read the written manifest table.
    manifest = pd.read_csv(manifest_artifact.path)

    # Confirm the previously registered artifact appears in the manifest file.
    assert "example" in set(manifest["name"])

    # Confirm the manifest artifact itself is now tracked by the manager.
    assert manager.artifacts[-1].name == "artifact_manifest"
