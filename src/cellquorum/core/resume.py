"""Stage-level resume: skip completed stages on rerun when safe.

This is the *read* side of the completion-sidecar contract written by
``core.pipeline._write_stage_completion_sidecars``. On rerun with resume
enabled, a stage may be skipped when:

1. a ``provenance/stages/<stage>/completion.json`` sidecar exists from a prior
   run, and its recorded ``input_fingerprint`` matches the current one; and
2. every artifact the sidecar recorded still exists on disk; and
3. the stage is *side-effect only* — it does not transform the AnnData that
   flows downstream.

Condition (3) is the conservative guardrail for this first implementation: the
completion sidecar stores artifact paths and fingerprints, but NOT the AnnData
state at that point. A stage that mutates ``adata`` (QC filtering, PCA,
clustering, integration) therefore cannot be resumed yet — skipping it would
lose the transformed matrix for downstream stages. Read-only/diagnostic stages
(annotation diagnostics, integration benchmark, population identity, report)
pass ``adata`` through unchanged, so skipping them on a fingerprint match is
safe. AnnData-checkpoint resume for compute-heavy stages is a documented
follow-up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Stages whose ``run`` does not change the AnnData that flows downstream. Only
# these are eligible for resume until AnnData checkpointing exists. Keep this
# conservative: when in doubt, a stage is NOT listed and simply re-runs.
RESUMABLE_STAGES: frozenset[str] = frozenset(
    {
        "annotation_diagnostics",
        "integration_benchmark",
        "population_identity",
        "report",
    }
)


@dataclass(frozen=True)
class ResumeDecision:
    """The outcome of consulting resume state for one stage."""

    # Whether the stage may be skipped (resumed).
    resume: bool

    # Human-readable reason (for notes/records) whether or not resuming.
    reason: str


def _completion_sidecar_path(provenance_dir: Path, stage_name: str) -> Path:
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

    sidecar_path = _completion_sidecar_path(provenance_dir, stage_name)
    if not sidecar_path.exists():
        return ResumeDecision(False, "no prior completion marker; re-running")

    # A malformed sidecar must never crash the run — treat it as "re-run".
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return ResumeDecision(False, "unreadable completion marker; re-running")

    # Only prior *successful* completions are reusable.
    if payload.get("status") != "success":
        return ResumeDecision(False, "prior stage did not succeed; re-running")

    # The fingerprint must match, or an input/config change invalidated it.
    prior_fingerprint = payload.get("input_fingerprint")
    if prior_fingerprint is None or prior_fingerprint != input_fingerprint:
        return ResumeDecision(False, "input fingerprint changed; re-running")

    # Every recorded artifact must still exist, or outputs are incomplete.
    for artifact in payload.get("output_artifacts", []):
        artifact_path = artifact.get("path")
        if artifact_path is None or not Path(artifact_path).exists():
            return ResumeDecision(False, "a recorded artifact is missing; re-running")

    return ResumeDecision(True, "resumed from completion marker (fingerprint match)")


__all__ = ["RESUMABLE_STAGES", "ResumeDecision", "decide_stage_resume"]
