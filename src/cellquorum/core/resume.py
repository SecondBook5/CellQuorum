"""Stage-level resume: skip completed stages on rerun when safe.

This is the *read* side of the completion-sidecar contract written by
``core.pipeline._write_stage_completion_sidecars``. On rerun with resume
enabled, a stage may be skipped when:

1. a ``provenance/stages/<stage>/completion.json`` sidecar exists from a prior
   run, and its recorded ``input_fingerprint`` matches the current one; and
2. every artifact the sidecar recorded still exists on disk; and
3. the stage is *side-effect only* — it does not transform the AnnData that
   flows downstream.

The integration benchmark returns metrics and leaves its input object unchanged.
Annotation diagnostics write cell-level evidence into obs, and population identity
writes its audit into uns. Both must run to reconstruct the downstream object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cellquorum.core.fingerprint import artifact_file_checksums
from cellquorum.core.stage import StageArtifact, StageExecutionRecord

# Stages whose ``run`` does not change the AnnData that flows downstream. Only
# these are eligible for resume until AnnData checkpointing exists. Keep this
# conservative: when in doubt, a stage is NOT listed and simply re-runs.
RESUMABLE_STAGES: frozenset[str] = frozenset({"integration_benchmark"})


@dataclass(frozen=True)
class ResumeDecision:
    """The outcome of consulting resume state for one stage."""

    # Whether the stage may be skipped (resumed).
    resume: bool

    # Human-readable reason (for notes/records) whether or not resuming.
    reason: str
    original: StageExecutionRecord | None = None

    @property
    def artifacts(self) -> tuple[StageArtifact, ...]:
        return tuple(self.original.output_artifacts) if self.original else ()


def completion_payload(record: StageExecutionRecord) -> dict:
    """Store a restorable execution record alongside the artifact existence snapshot."""
    payload = record.to_dict()
    payload.update(
        schema_version=1,
        completed_at_utc=record.ended_at_utc.isoformat(),
        n_output_artifacts=len(record.output_artifacts),
        output_artifacts=[
            {**artifact.to_dict(), "exists": artifact.path.exists()}
            for artifact in record.output_artifacts
        ],
    )
    if record.stage_name in RESUMABLE_STAGES:
        payload["artifact_checksums"] = artifact_file_checksums(
            a.path for a in record.output_artifacts
        )
    return payload


def completion_path(provenance_dir: Path, stage_name: str) -> Path:
    """Return the completion.json path for a stage under a provenance dir."""

    safe_stage_name = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_" for char in stage_name
    ).strip("_")
    if not safe_stage_name:
        safe_stage_name = "stage"
    return provenance_dir / "stages" / safe_stage_name / "completion.json"


def decide_stage_resume(
    *,
    stage_name: str,
    provenance_dir: Path,
    input_fingerprint: str | None,
    required_outputs: tuple[Path, ...] = (),
) -> ResumeDecision:
    """
    Decide whether a stage can be resumed (skipped) on this run.

    Args:
        stage_name: Registry stage name.
        provenance_dir: The run's ``provenance`` directory (where sidecars live).
        input_fingerprint: Freshly computed input fingerprint for this run, or
            None if it could not be computed.

    Returns:
        A ResumeDecision. ``resume`` is True only when a prior successful
        completion sidecar matches the current fingerprint and all its recorded
        artifacts still exist, and the stage is side-effect only.
    """

    # Only side-effect-only stages are eligible until AnnData checkpointing.
    if stage_name not in RESUMABLE_STAGES:
        return ResumeDecision(False, "stage transforms adata; not resumable yet")

    # A missing fingerprint means we cannot prove input equivalence — re-run.
    if input_fingerprint is None:
        return ResumeDecision(False, "no input fingerprint; re-running")

    sidecar_path = completion_path(provenance_dir, stage_name)
    if not sidecar_path.exists():
        return ResumeDecision(False, "no prior completion marker; re-running")

    # A malformed sidecar must never crash the run — treat it as "re-run".
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return ResumeDecision(False, "unreadable completion marker; re-running")

    # Only prior *successful* completions are reusable.
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return ResumeDecision(False, "prior stage did not succeed; re-running")

    # The fingerprint must match, or an input/config change invalidated it.
    prior_fingerprint = payload.get("input_fingerprint")
    if prior_fingerprint is None or prior_fingerprint != input_fingerprint:
        return ResumeDecision(False, "input fingerprint changed; re-running")

    try:
        original = StageExecutionRecord.from_dict(payload)
    except (ValueError, TypeError, KeyError, AttributeError):
        return ResumeDecision(False, "missing original execution evidence; re-running")
    if original.stage_name != stage_name:
        return ResumeDecision(False, "original stage name differs; re-running")
    registered = {a.path for a in original.output_artifacts}
    if any(not path.exists() for path in registered):
        return ResumeDecision(False, "a recorded artifact is missing; re-running")
    if payload.get("artifact_checksums") != artifact_file_checksums(registered):
        return ResumeDecision(
            False, "output contents have not been verified or changed; re-running"
        )
    if original.metrics.get("incomplete_reasons") or original.metrics.get("mode") == "fallback":
        return ResumeDecision(False, "incomplete benchmark; re-running")

    if missing_outputs(required_outputs, registered):
        return ResumeDecision(False, "a required stage output is missing or empty; re-running")
    return ResumeDecision(True, "resumed from completion marker (fingerprint match)", original)


def missing_outputs(required: tuple[Path, ...], registered: set[Path]) -> list[Path]:
    """Check only artifacts attributed to this stage execution or its validated sidecar."""
    return [p for p in required if p not in registered or not p.is_file() or p.stat().st_size == 0]


__all__ = ["RESUMABLE_STAGES", "ResumeDecision", "decide_stage_resume"]
