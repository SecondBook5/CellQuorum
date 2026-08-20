"""Tests for the StageArtifactWriter stage-facing artifact facade."""

import json

import pandas as pd
import pytest

from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.core.stage_artifact_writer import StageArtifactWriter


def _context(tmp_path):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    return PipelineContext(config=None, paths=paths)


def test_table_writes_csv_to_results_with_identical_path(tmp_path):
    ctx = _context(tmp_path)
    writer = StageArtifactWriter.from_context(ctx)
    df = pd.DataFrame({"gene": ["A", "B"], "logfc": [1.0, -2.0]})

    artifact = writer.table(
        df, "da_proportion_ttest.csv", name="da_proportion_ttest", description="DA t-test"
    )

    expected = ctx.paths.results / "da_proportion_ttest.csv"
    assert artifact.path == expected
    assert expected.exists()
    assert artifact.name == "da_proportion_ttest"
    assert artifact.kind == "csv"
    assert artifact.description == "DA t-test"
    # index NOT written by default (parity with hand-rolled to_csv(index=False))
    assert pd.read_csv(expected).columns.tolist() == ["gene", "logfc"]


def test_table_respects_index_true(tmp_path):
    ctx = _context(tmp_path)
    writer = StageArtifactWriter.from_context(ctx)
    df = pd.DataFrame({"score": [1, 2]}, index=pd.Index(["s1", "s2"], name="sample"))

    writer.table(df, "scores.csv", name="scores", description="scores", index=True)

    written = pd.read_csv(ctx.paths.results / "scores.csv")
    assert "sample" in written.columns  # index column preserved


def test_table_subdir_and_namespace_resolve_under_root(tmp_path):
    ctx = _context(tmp_path)
    writer = StageArtifactWriter.from_context(ctx)
    df = pd.DataFrame({"x": [1]})

    art = writer.table(df, "t.csv", name="t", description="d", namespace="reports", subdir="de")

    assert art.path == ctx.paths.reports / "de" / "t.csv"
    assert art.path.exists()


def test_table_creates_missing_subdir(tmp_path):
    ctx = _context(tmp_path)
    writer = StageArtifactWriter.from_context(ctx)
    df = pd.DataFrame({"x": [1]})

    art = writer.table(df, "t.csv", name="t", description="d", subdir="deep/nested")

    assert art.path == ctx.paths.results / "deep" / "nested" / "t.csv"
    assert art.path.exists()


def test_json_writes_sorted_indented_payload(tmp_path):
    ctx = _context(tmp_path)
    writer = StageArtifactWriter.from_context(ctx)

    art = writer.json({"b": 2, "a": 1}, "summary.json", name="summary", description="s")

    assert art.path == ctx.paths.results / "summary.json"
    assert art.kind == "json"
    loaded = json.loads(art.path.read_text())
    assert loaded == {"a": 1, "b": 2}
    # sort_keys=True parity with ArtifactManager.write_json
    assert art.path.read_text().index('"a"') < art.path.read_text().index('"b"')


def test_register_records_prewritten_object(tmp_path):
    ctx = _context(tmp_path)
    writer = StageArtifactWriter.from_context(ctx)
    # Simulate a domain library having already written an object.
    obj = ctx.paths.objects / "trajectory.h5ad"
    obj.write_bytes(b"stub")

    art = writer.register(
        name="trajectory",
        filename="trajectory.h5ad",
        kind="h5ad",
        description="trajectory object",
        namespace="objects",
    )

    assert art.path == obj
    assert art.kind == "h5ad"


def test_default_subdir_from_context_applies(tmp_path):
    ctx = _context(tmp_path)
    writer = StageArtifactWriter.from_context(ctx, default_subdir="enrichment")
    art = writer.table(pd.DataFrame({"x": [1]}), "gsea.csv", name="gsea", description="d")
    assert art.path == ctx.paths.results / "enrichment" / "gsea.csv"


def test_unknown_namespace_raises(tmp_path):
    ctx = _context(tmp_path)
    writer = StageArtifactWriter.from_context(ctx)
    with pytest.raises(ValueError, match="namespace"):
        writer.table(
            pd.DataFrame({"x": [1]}), "t.csv", name="t", description="d", namespace="bogus"
        )


def test_import_pulls_no_optional_backend_deps():
    # Importing the writer must not trigger any OPTIONAL/heavy backend import
    # (scvelo/cellrank/liana/celltypist). pandas/anndata/scanpy are CORE deps
    # (scanpy is the single-cell foundation, pulled by the package's notebook
    # namespaces and always installed) and are expected. Run in a fresh
    # subprocess so the check is not polluted by modules other tests imported.
    import subprocess
    import sys

    code = (
        "import cellquorum.core.stage_artifact_writer as m, sys;"
        "heavy=[h for h in ('scvelo','cellrank','liana','celltypist') "
        "if h in sys.modules];"
        "assert not heavy, f'writer import pulled optional backends: {heavy}';"
        "assert hasattr(m, 'StageArtifactWriter')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
