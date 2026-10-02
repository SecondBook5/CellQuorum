"""Tests for the run-report renderer (Markdown, HTML, methods text)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.artifacts import ArtifactManager
from cellquorum.core.reports import (
    render_html,
    render_markdown,
    render_methods_text,
    write_run_report,
)
from cellquorum.core.stage import StageExecutionRecord, StageResult


def _records() -> list[StageExecutionRecord]:
    """Build one success and one skipped record."""

    started = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    ended = started + timedelta(seconds=2)
    success = StageExecutionRecord.success(
        stage_name="qc",
        result=StageResult(adata=ad.AnnData(X=np.ones((1, 1))), backend="python"),
        started_at_utc=started,
        ended_at_utc=ended,
        backend_used="python",
    )
    skipped = StageExecutionRecord.skipped(
        stage_name="reference_mapping",
        reason="Rscript unavailable",
        started_at_utc=ended,
        ended_at_utc=ended,
    )
    return [success, skipped]


def test_render_markdown_lists_stages_and_methods() -> None:
    cfg = CellQuorumConfig.model_validate({"project": {"name": "demo"}})
    md = render_markdown(config=cfg, records=_records())
    assert "CellQuorum run report — demo" in md
    assert "| qc | success |" in md
    assert "reference_mapping" in md
    assert "## Methods" in md
    # Methods text names the seed.
    assert str(cfg.run.random_seed) in md


def test_render_html_is_standalone_and_escaped() -> None:
    cfg = CellQuorumConfig.model_validate({"project": {"name": "demo"}})
    page = render_html(config=cfg, records=_records())
    assert page.startswith("<!DOCTYPE html>")
    assert "<table" in page
    assert "qc" in page


def test_render_methods_text_names_backend_and_seed() -> None:
    cfg = CellQuorumConfig.model_validate({"project": {"name": "demo"}})
    text = render_methods_text(config=cfg, records=_records())
    assert "qc" in text
    assert "python" in text
    assert str(cfg.run.random_seed) in text


def test_write_run_report_writes_files_when_enabled(tmp_path: Path) -> None:
    cfg = CellQuorumConfig.model_validate(
        {"project": {"name": "demo"}, "report": {"enabled": True, "html": True, "markdown": True}}
    )
    manager = ArtifactManager.from_root(tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    write_run_report(config=cfg, records=_records(), artifact_manager=manager)

    assert (tmp_path / "reports" / "report.md").exists()
    assert (tmp_path / "reports" / "report.html").exists()
    assert (tmp_path / "reports" / "methods.txt").exists()


# ---------------------------------------------------------------------------
# Warnings and failures have to be READABLE in the report, not just counted.
# A real 36-stage LEC run emitted 12 warnings — a leiden cluster that lost its
# entire velocity to an eigensolver iteration ceiling, a CellRank kernel that
# never built, a figure that failed to write — and report.md showed the integer
# count in a table cell and nothing else, so the only way to learn any of it was
# to parse provenance/stage_execution_records.json by hand.
# ---------------------------------------------------------------------------


def _records_with_warnings_and_a_failure() -> list[StageExecutionRecord]:
    started = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    ended = started + timedelta(seconds=2)
    warned = StageExecutionRecord.success(
        stage_name="trajectory",
        result=StageResult(
            adata=ad.AnnData(X=np.ones((1, 1))),
            backend="python",
            warnings=[
                "5: velocity computation failed: ARPACK error -1: No convergence",
                "cytotrace kernel failed: Unable to find `'imputed'` in `adata.layers`",
            ],
        ),
        started_at_utc=started,
        ended_at_utc=ended,
        backend_used="python",
    )
    failed = StageExecutionRecord.failed(
        stage_name="differential_expression",
        error=ValueError("design ~ donor + condition is rank-deficient"),
        started_at_utc=ended,
        ended_at_utc=ended,
        backend_used="rscript",
    )
    return [warned, failed]


def test_markdown_prints_every_warning_verbatim_not_just_a_count() -> None:
    cfg = CellQuorumConfig.model_validate({"project": {"name": "demo"}})
    md = render_markdown(config=cfg, records=_records_with_warnings_and_a_failure())

    assert "## Warnings" in md
    assert "### trajectory" in md
    assert "ARPACK error -1: No convergence" in md
    assert "cytotrace kernel failed" in md
    assert "2 warning(s) across 1 stage(s)." in md


def test_markdown_says_so_when_a_run_emitted_no_warnings() -> None:
    """A clean run must state that it was clean rather than omit the section.

    An absent section reads as "the report does not cover warnings", which is the
    ambiguity this whole change removes.
    """
    cfg = CellQuorumConfig.model_validate({"project": {"name": "demo"}})
    md = render_markdown(config=cfg, records=_records())
    assert "## Warnings" in md
    assert "No warnings were emitted." in md


def test_markdown_detail_column_names_what_failed() -> None:
    """A failed stage's StageExecutionError was previously rendered nowhere."""
    cfg = CellQuorumConfig.model_validate({"project": {"name": "demo"}})
    md = render_markdown(config=cfg, records=_records_with_warnings_and_a_failure())

    row = next(line for line in md.splitlines() if line.startswith("| differential_expression |"))
    assert "ValueError" in row
    assert "rank-deficient" in row


def test_markdown_table_survives_a_pipe_or_newline_in_a_message() -> None:
    """One unescaped pipe breaks every table row after it in a Markdown renderer."""
    started = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    record = StageExecutionRecord.skipped(
        stage_name="grn",
        reason="no motif ranking database found | expected one of:\n  a.feather\n  b.feather",
        started_at_utc=started,
        ended_at_utc=started,
    )
    cfg = CellQuorumConfig.model_validate({"project": {"name": "demo"}})
    md = render_markdown(config=cfg, records=[record])

    row = next(line for line in md.splitlines() if line.startswith("| grn |"))
    # Six columns means five internal separators; an unescaped pipe would add more.
    assert row.count("|") - row.count("\\|") == 7
    assert "\n" not in row


def test_html_lists_the_warnings_too() -> None:
    cfg = CellQuorumConfig.model_validate({"project": {"name": "demo"}})
    page = render_html(config=cfg, records=_records_with_warnings_and_a_failure())

    assert "<h2>Warnings</h2>" in page
    assert "<h3>trajectory</h3>" in page
    assert "ARPACK error -1: No convergence" in page
    # Escaped, because warning text carries backticks and quotes from library errors.
    assert "&#x27;imputed&#x27;" in page or "&#39;imputed&#39;" in page


def test_write_run_report_noop_when_disabled(tmp_path: Path) -> None:
    cfg = CellQuorumConfig.model_validate(
        {"project": {"name": "demo"}, "report": {"enabled": False}}
    )
    manager = ArtifactManager.from_root(tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    write_run_report(config=cfg, records=_records(), artifact_manager=manager)

    assert not (tmp_path / "reports" / "report.md").exists()
