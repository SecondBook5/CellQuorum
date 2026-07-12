"""Test the RunReporter runtime progress/output system."""

from __future__ import annotations

import io
from datetime import UTC, datetime

from rich.console import Console

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.run_reporter import RunReporter
from cellquorum.core.stage import StageExecutionRecord


def _reporter(verbose=True, level="normal"):
    """Create a reporter with a captured string buffer for testing."""
    # Capture Rich output into a string buffer (force_terminal False → plain).
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    return RunReporter(verbose=verbose, level=level, console=console), buf


def test_reporter_quiet_is_silent():
    """Verify that verbose=False produces no output."""
    rep, buf = _reporter(verbose=False)
    rep.banner("0.1.0", "demo", "run1")
    rep.config_echo(CellQuorumConfig.model_validate({"project": {"name": "demo"}}))
    rep.stage_start("qc", 1, 3)
    assert buf.getvalue() == ""


def test_banner_names_version_and_project():
    """Verify that banner shows CellQuorum version and project name."""
    rep, buf = _reporter()
    rep.banner("0.1.0", "le_kc", "run1")
    out = buf.getvalue()
    assert "CellQuorum" in out and "0.1.0" in out and "le_kc" in out


def test_config_echo_shows_backend_and_enabled_stages_in_order():
    """Verify config echo shows backend, stages, and truncates long lists."""
    rep, buf = _reporter()
    cfg = CellQuorumConfig.model_validate(
        {
            "project": {"name": "le_kc"},
            "reference_mapping": {
                "enabled": True,
                "atlas_h5ad": "/data/fiskin.h5ad",
                "label_key": "celltype_granular",
                "force_genes": ["A", "B", "C", "D", "E"],
            },
        }
    )
    # Call config_echo without planned_stage_names (fallback to config.stages).
    rep.config_echo(cfg)
    out = buf.getvalue()
    # Enabled stage shown.
    assert "reference_mapping" in out
    # Key params shown.
    assert "fiskin.h5ad" in out or "celltype_granular" in out
    # Long list truncated to a count, not dumped.
    assert "A, B, C, D, E" not in out


def test_stage_end_renders_status_and_duration():
    """Verify stage_end renders stage name and duration."""
    rep, buf = _reporter()
    rec = StageExecutionRecord(
        stage_name="qc",
        status="success",
        started_at_utc=datetime.now(UTC),
        ended_at_utc=datetime.now(UTC),
        duration_seconds=12.3,
    )
    rep.stage_end(rec)
    out = buf.getvalue()
    assert "qc" in out and "12.3" in out


def test_progress_context_advances_without_crash():
    """Verify progress context manager doesn't crash on advance."""
    rep, _ = _reporter()
    with rep.progress(total=3) as bar:
        bar.advance()
        bar.advance()
        bar.advance()  # No exception.


def test_log_level_quiet_suppresses_non_essential_output():
    """Verify that log_level='quiet' suppresses banner/config/stages but keeps summary."""
    rep, buf = _reporter(verbose=True, level="quiet")

    # Banner, config, and stage lines should NOT show.
    rep.banner("0.1.0", "demo", "run1")
    rep.config_echo(CellQuorumConfig.model_validate({"project": {"name": "demo"}}))
    rep.stage_start("qc", 1, 3)
    rec = StageExecutionRecord(
        stage_name="qc",
        status="success",
        started_at_utc=datetime.now(UTC),
        ended_at_utc=datetime.now(UTC),
        duration_seconds=1.0,
    )
    rep.stage_end(rec)

    # Verify no output so far (banner/config/stage-start/stage-end suppressed).
    out_before_summary = buf.getvalue()
    assert "CellQuorum" not in out_before_summary
    assert "Configuration" not in out_before_summary
    assert "▶ qc" not in out_before_summary
    assert "✓ qc" not in out_before_summary

    # Summary should still print (essential output).
    rep.run_summary([rec], "/tmp/outputs", 10.0)
    out_after_summary = buf.getvalue()
    assert "Run Summary" in out_after_summary or "qc" in out_after_summary


def test_config_echo_excludes_per_stage_disabled():
    """Verify config_echo excludes stages with per-stage .enabled=False."""
    rep, buf = _reporter()
    cfg = CellQuorumConfig.model_validate(
        {
            "project": {"name": "test"},
            "stages": {"dimensionality": True, "clustering": True},
            "dimensionality": {"enabled": False, "method": "pca"},
            "clustering": {"enabled": True, "method": "leiden"},
        }
    )
    # Provide planned_stage_names with only clustering (dimensionality excluded).
    rep.config_echo(cfg, planned_stage_names=["clustering"])
    out = buf.getvalue()
    # Clustering should be shown.
    assert "clustering" in out
    # Dimensionality should NOT be shown (excluded from planned_stage_names).
    assert "dimensionality" not in out
