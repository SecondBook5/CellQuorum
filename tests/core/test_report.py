"""Tests for the run-report renderer (Markdown, HTML, methods text)."""

from __future__ import annotations

from dataclasses import replace
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
from cellquorum.core.stage import StageArtifact, StageExecutionRecord, StageResult


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


def test_html_links_only_recorded_local_outputs_and_encodes_filenames(tmp_path: Path) -> None:
    root = tmp_path / "run"
    folder = root / "figures"
    folder.mkdir(parents=True)
    panel = folder / "donor #1 & 2.pdf"
    panel.touch()
    (folder / "stale.png").touch()
    outside = tmp_path / "external.csv"
    outside.touch()
    records = _records()
    for path in [panel, outside, folder / "missing.pdf"]:
        records[0].output_artifacts.append(StageArtifact("panel", path, "figure", "A < B"))
    page = render_html(config=CellQuorumConfig(), records=records, run_root=root)
    assert 'href="../figures/donor%20%231%20%26%202.pdf"' in page and "A &lt; B" in page
    assert all(name not in page for name in ["stale.png", "missing.pdf", "external.csv"])


def test_study_report_separates_disabled_from_requested_skips_and_setup():
    records = _records()
    stamp = datetime.now(UTC)
    records.append(
        StageExecutionRecord.success(
            stage_name="bootstrap",
            result=StageResult(adata=ad.AnnData(X=np.ones((1, 1)))),
            started_at_utc=stamp,
            ended_at_utc=stamp,
            backend_used="python",
        )
    )
    for name in ["trajectory", "trajectory_viz"]:
        records.append(
            StageExecutionRecord.skipped(
                stage_name=name,
                reason="The study lacks a specified lineage <root>.",
                started_at_utc=stamp,
                ended_at_utc=stamp,
                details={"plan_status": "disabled", "implemented": True},
            )
        )
    cfg = CellQuorumConfig()
    md = render_markdown(config=cfg, records=records)
    page = render_html(config=cfg, records=records)
    for contents in [md, page]:
        assert "1 completed · 1 skipped · 0 failed · 2 not selected" in contents
        assert contents.count("The study lacks a specified lineage") == 1
        assert "Rscript unavailable" in contents
    assert "&lt;root&gt;" in page and "| trajectory | skipped |" not in md
    records.append(
        StageExecutionRecord.skipped(
            stage_name="population_identity",
            reason="Reused verified output",
            started_at_utc=stamp,
            ended_at_utc=stamp,
            details={"resumed": True},
        )
    )
    md = render_markdown(config=cfg, records=records)
    assert "1 skipped" in md and "1 reused from checkpoint" in md
    assert "| population_identity | reused |" in md


def test_pbmc_uncertainty_counts_and_source_link_in_both_reports() -> None:
    cfg = CellQuorumConfig.model_validate({"project": {"name": "PBMC"}})
    metrics = {
        "per_method": [
            {
                "key_added": "RNA call",
                "pbmc_refinement": {"n_cells": 100, "uncertain": 3, "reasons": {"weak RNA": 3}},
            }
        ]
    }
    record = replace(_records()[0], stage_name="annotation", metrics=metrics)
    for text in [
        render_html(config=cfg, records=[record]),
        render_markdown(config=cfg, records=[record]),
    ]:
        assert "3 of 100 cells (3.0%) are Unassigned" in text
        assert "weak RNA (3)" in text
        assert "../results/annotation/RNA%20call_evidence.csv" in text


def test_report_accounts_for_unvisited_stages_in_the_actual_plan():
    from cellquorum.core.planner import PipelinePlan, PlannedStage

    cfg = CellQuorumConfig()
    plan = PipelinePlan(
        profile="publication",
        stages=[
            PlannedStage("qc", True, "enabled", "Requested"),
            PlannedStage("discovery", True, "enabled", "Requested"),
            PlannedStage("trajectory", False, "disabled", "No lineage root"),
        ],
    )
    for render in (render_html, render_markdown):
        report = render(config=cfg, records=_records()[:1], plan=plan)
        assert "1 selected but not reached" in report
        assert "without an execution record: discovery" in report
        assert "1 not selected" in report and "No lineage root" in report
