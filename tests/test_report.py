"""Tests for the run-report renderer (Markdown, HTML, methods text)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.artifacts import ArtifactManager
from cellquorum.core.stage import StageExecutionRecord, StageResult
from cellquorum.reports.run_report import (
    render_html,
    render_markdown,
    render_methods_text,
    write_run_report,
)


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


def test_write_run_report_noop_when_disabled(tmp_path: Path) -> None:
    cfg = CellQuorumConfig.model_validate(
        {"project": {"name": "demo"}, "report": {"enabled": False}}
    )
    manager = ArtifactManager.from_root(tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    write_run_report(config=cfg, records=_records(), artifact_manager=manager)

    assert not (tmp_path / "reports" / "report.md").exists()
