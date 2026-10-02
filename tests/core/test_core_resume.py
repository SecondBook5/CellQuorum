"""Tests for stage-level resume decisions."""

from __future__ import annotations

import json
from pathlib import Path

from cellquorum.core.resume import RESUMABLE_STAGES, decide_stage_resume


def _write_sidecar(
    provenance_dir: Path,
    stage_name: str,
    *,
    status: str = "success",
    input_fingerprint: str = "fp-123",
    artifacts: list[Path] | None = None,
) -> None:
    """Write a minimal completion sidecar for a stage."""

    stage_dir = provenance_dir / "stages" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "stage_name": stage_name,
        "status": status,
        "input_fingerprint": input_fingerprint,
        "output_artifacts": [{"path": str(p)} for p in (artifacts or [])],
    }
    (stage_dir / "completion.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resume_when_fingerprint_matches_and_artifacts_exist(tmp_path: Path) -> None:
    """A resumable stage with a matching marker and present artifacts resumes."""

    provenance = tmp_path / "provenance"
    artifact = tmp_path / "results" / "benchmark.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("ok", encoding="utf-8")

    _write_sidecar(
        provenance, "integration_benchmark", input_fingerprint="fp-1", artifacts=[artifact]
    )

    decision = decide_stage_resume(
        stage_name="integration_benchmark",
        provenance_dir=provenance,
        input_fingerprint="fp-1",
    )
    assert decision.resume is True


def test_no_resume_when_fingerprint_differs(tmp_path: Path) -> None:
    """A changed input fingerprint invalidates resume."""

    provenance = tmp_path / "provenance"
    _write_sidecar(provenance, "integration_benchmark", input_fingerprint="fp-old")

    decision = decide_stage_resume(
        stage_name="integration_benchmark",
        provenance_dir=provenance,
        input_fingerprint="fp-new",
    )
    assert decision.resume is False
    assert "fingerprint" in decision.reason


def test_no_resume_when_artifact_missing(tmp_path: Path) -> None:
    """A recorded-but-missing artifact blocks resume."""

    provenance = tmp_path / "provenance"
    missing = tmp_path / "results" / "gone.csv"
    _write_sidecar(
        provenance, "integration_benchmark", input_fingerprint="fp-1", artifacts=[missing]
    )

    decision = decide_stage_resume(
        stage_name="integration_benchmark",
        provenance_dir=provenance,
        input_fingerprint="fp-1",
    )
    assert decision.resume is False
    assert "artifact" in decision.reason


def test_no_resume_for_adata_transforming_stage(tmp_path: Path) -> None:
    """A stage that transforms adata is never resumed (conservative guardrail)."""

    provenance = tmp_path / "provenance"
    _write_sidecar(provenance, "qc", input_fingerprint="fp-1")

    decision = decide_stage_resume(
        stage_name="qc", provenance_dir=provenance, input_fingerprint="fp-1"
    )
    assert decision.resume is False
    assert "qc" not in RESUMABLE_STAGES


def test_no_resume_without_prior_marker(tmp_path: Path) -> None:
    """A resumable stage with no prior completion marker re-runs."""

    provenance = tmp_path / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)

    decision = decide_stage_resume(
        stage_name="report", provenance_dir=provenance, input_fingerprint="fp-1"
    )
    assert decision.resume is False
