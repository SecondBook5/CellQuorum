"""Refuse to write a run into a directory holding outputs of a DISABLED stage.

A CellQuorum run directory is not a scratch space -- it is the replot source and the
provenance record a manuscript cites. That makes one specific mixture dangerous: a
stage turned OFF in the config, with the previous run's outputs for that stage still
sitting in the tree. Nothing on disk distinguishes them from results this run
produced, so a figure or table built later reads them as current.

This is not hypothetical. Under the global-QC-once architecture every per-lineage run
sets ``stages.qc: false``, because QC ran once on the atlas. A run directory reused
from before that decision kept 197 MB of QC output -- ``thresholds.csv``,
``cell_decisions.csv`` and a 188 MB ``qc.h5ad`` -- inside a run whose own summary said
``qc | skipped``. A table citing those thresholds would be describing a filter that
was never applied to those cells.

The check is deliberately **timestamp-free**. Comparing mtimes against the run start
would flag the legitimate inheritance a ``--from-stage`` resume depends on, and would
turn a real integrity check into noise people learn to ignore. Config intent is the
sharper question: if a stage is disabled, *no* run of this config should ever have
produced output for it, whenever that output was written. So the contradiction is
between the config and the disk, and it holds regardless of resume.

It raises rather than warns, and it runs before any stage executes. A warning is how
the QC leftovers survived a 36-minute run in the first place; failing in the first
second costs nothing and the remedy is one ``rm`` or ``mv`` the message names.

Scope, stated plainly: the check is directory-scoped. It finds outputs a stage wrote
into its own ``results/<stage>/`` or ``figures/<stage>/`` subdirectory. Stages that
write a flat file instead (``results/de_pseudobulk_edger.csv`` for
``differential_expression``) are NOT covered, because there is no reliable mapping
from a stage name to an arbitrary filename, and inventing one by prefix-matching
would produce both misses and false accusations. Directory-scoped coverage catches
the stages that write bulk artifacts, which are the ones worth 197 MB of confusion.

That leaves a second, narrower hazard the gate provably cannot see, so this module
also *reports* it. A stage that runs successfully still writes one artifact per group,
cluster or pathway, and the set of those entities changes between runs. When an entity
drops out, its file from the previous run stays behind looking exactly like output of
this one:

- ``results/trajectory/velocity/4.h5ad`` survived a run in which group 4 held 28 cells
  and was correctly skipped under a 30-cell floor. Eighteen sibling objects were
  rewritten; that one was a different run's, for a different partition.
- ``figures/trajectory/fate_14.png`` survived a run whose macrostate search returned
  groups 5 and 10 and not 14. A supplement assembled from the figure directory would
  have carried a macrostate the analysis did not find.

:func:`find_inherited_artifacts` catches these by asking which files this run did not
write. That question needs timestamps, so unlike the gate it cannot distinguish
legitimate inheritance (a ``--from-stage`` resume) from a leftover, and therefore it
only ever reports -- it never fails a run. Two mechanisms, each honest about what it
can prove: the gate errors on a contradiction it can establish from config alone; the
report inventories a suspicion it cannot resolve on its own.
"""

from __future__ import annotations

# Import dataclass for the frozen finding.
from dataclasses import dataclass

# Import datetime for the inherited-artifact cutoff.
from datetime import UTC, datetime

# Import Path for the directory walk.
from pathlib import Path

# Import the engine's base error so callers catch one family.
from cellquorum.core.exceptions import CellQuorumError

# The output subtrees a stage may own a directory in.
STAGE_OUTPUT_SUBDIRS: tuple[str, ...] = ("results", "figures")


class StaleOutputError(CellQuorumError):
    """Raised when a run directory holds outputs of a stage this config disables."""


@dataclass(frozen=True)
class StaleStageOutput:
    """
    Outputs found for a stage the config has turned off.

    Args:
        stage: Name of the disabled stage.
        directory: The offending directory, relative to the run root.
        n_files: How many files it holds, at any depth.
        total_bytes: Their combined size, which is what makes the confusion costly.
    """

    stage: str
    directory: str
    n_files: int
    total_bytes: int


def find_disabled_stage_outputs(
    root: Path,
    *,
    disabled_stages: list[str],
    subdirs: tuple[str, ...] = STAGE_OUTPUT_SUBDIRS,
) -> list[StaleStageOutput]:
    """
    Find output directories belonging to stages this config disables.

    Args:
        root: The run's output root.
        disabled_stages: Stage names the plan marked disabled.
        subdirs: Output subtrees to look in.

    Returns:
        One finding per non-empty directory, ordered by size descending so the
        message leads with the artifact most likely to be mistaken for a result.
        Empty when the run directory is consistent with the config.
    """

    findings: list[StaleStageOutput] = []
    for stage in disabled_stages:
        for subdir in subdirs:
            directory = root / subdir / stage
            if not directory.is_dir():
                continue
            files = [path for path in directory.rglob("*") if path.is_file()]
            if not files:
                continue
            findings.append(
                StaleStageOutput(
                    stage=stage,
                    directory=f"{subdir}/{stage}",
                    n_files=len(files),
                    total_bytes=sum(path.stat().st_size for path in files),
                )
            )
    return sorted(findings, key=lambda finding: finding.total_bytes, reverse=True)


def format_stale_outputs(findings: list[StaleStageOutput], *, root: Path) -> str:
    """
    Render findings as an error message that says what to do about them.

    Args:
        findings: The offending directories.
        root: The run's output root, quoted so the remedy is copy-pasteable.

    Returns:
        A multi-line message.
    """

    lines = [
        f"This run directory holds outputs of {len(findings)} stage(s) that the config "
        "DISABLES, so those files cannot have come from this run:",
        "",
    ]
    for finding in findings:
        lines.append(
            f"  {finding.directory:32s} {finding.n_files:4d} file(s), "
            f"{finding.total_bytes / 1e6:8.1f} MB   (stage '{finding.stage}' is disabled)"
        )
    lines += [
        "",
        "A run directory is the replot source and the provenance a manuscript cites, so "
        "leftovers from a stage that did not run are indistinguishable on disk from "
        "results that did. Remove or move them aside, then re-run:",
        "",
    ]
    lines += [f"  rm -rf {root / finding.directory}" for finding in findings]
    lines += [
        "",
        "If you meant to keep them, move the whole run directory instead of writing a "
        "second run's results on top of the first.",
    ]
    return "\n".join(lines)


def assert_output_dir_matches_config(root: Path, *, disabled_stages: list[str]) -> None:
    """
    Halt if the run directory contradicts the config's disabled stages.

    Args:
        root: The run's output root.
        disabled_stages: Stage names the plan marked disabled.

    Raises:
        StaleOutputError: If any disabled stage has outputs on disk.
    """

    findings = find_disabled_stage_outputs(root, disabled_stages=disabled_stages)
    if findings:
        raise StaleOutputError(format_stale_outputs(findings, root=root))


@dataclass(frozen=True)
class InheritedArtifact:
    """
    A file in the output tree that this run did not write.

    Args:
        path: Location relative to the run root, so the record survives a moved
            or renamed run directory.
        size_bytes: File size.
        modified_utc: Last-modified time, ISO-8601. Kept because it is the only
            evidence available for *which* earlier run left the file: a cluster of
            identical timestamps is one superseded run, a scatter is several.
    """

    path: str
    size_bytes: int
    modified_utc: str


def find_inherited_artifacts(
    root: Path,
    *,
    run_started_at: datetime,
    subdirs: tuple[str, ...] = STAGE_OUTPUT_SUBDIRS,
    grace_seconds: float = 60.0,
) -> list[InheritedArtifact]:
    """
    Inventory output files older than this run.

    A run that rewrites its whole output tree returns nothing here. Anything returned
    is a file some *earlier* process put there, which on a reused directory means an
    artifact for an entity this run no longer produces -- a skipped group, a cluster
    that merged away, a pathway that fell below threshold.

    The grace window absorbs the ordinary skew between the recorded run-start
    timestamp and filesystem mtimes (clock granularity, a stage that began writing
    moments before the recorded start). It is generous on purpose: this function
    exists to surface leftovers from *hours*-old runs, so a minute of slack costs no
    real sensitivity and removes a whole class of false positives.

    Args:
        root: The run's output root.
        run_started_at: When this run began. Must be timezone-aware; a naive
            datetime is rejected rather than silently interpreted as local time,
            which would shift the cutoff by the UTC offset and either hide
            leftovers or invent them.
        subdirs: Output subtrees to inventory.
        grace_seconds: Slack subtracted from the cutoff.

    Returns:
        Findings ordered by size descending. Empty when the run wrote everything
        in its tree.

    Raises:
        ValueError: If ``run_started_at`` has no timezone.
    """

    if run_started_at.tzinfo is None:
        raise ValueError(
            "run_started_at must be timezone-aware; a naive timestamp would shift "
            "the cutoff by the local UTC offset and silently mis-classify files."
        )

    cutoff = run_started_at.timestamp() - grace_seconds
    findings: list[InheritedArtifact] = []
    for subdir in subdirs:
        base = root / subdir
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            if stat.st_mtime >= cutoff:
                continue
            findings.append(
                InheritedArtifact(
                    path=str(path.relative_to(root)),
                    size_bytes=stat.st_size,
                    modified_utc=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                )
            )
    return sorted(findings, key=lambda finding: finding.size_bytes, reverse=True)


def format_inherited_artifacts(findings: list[InheritedArtifact], *, max_listed: int = 10) -> str:
    """
    Render an inherited-artifact inventory as a warning for the run summary.

    Args:
        findings: The inventory.
        max_listed: How many paths to name before summarising the remainder. A
            resume can inherit thousands of files legitimately, and a warning long
            enough to scroll the run summary off screen is a warning nobody reads.

    Returns:
        A one-or-few-line message, or the empty string when there is nothing to say.
    """

    if not findings:
        return ""

    total = sum(finding.size_bytes for finding in findings)
    lines = [
        f"{len(findings)} file(s) in results/ and figures/ predate this run "
        f"({total / 1e6:.1f} MB). This run did not write them. If the directory was "
        "reused, each one is an artifact for an entity this run no longer produces "
        "(a skipped group, a merged cluster, a pathway below threshold) and is "
        "indistinguishable on disk from a current result. Inventory: "
        "provenance/inherited_artifacts.csv",
    ]
    for finding in findings[:max_listed]:
        lines.append(f"    {finding.path}  ({finding.size_bytes / 1e6:.1f} MB)")
    if len(findings) > max_listed:
        lines.append(f"    ... and {len(findings) - max_listed} more")
    return "\n".join(lines)


__all__ = [
    "STAGE_OUTPUT_SUBDIRS",
    "InheritedArtifact",
    "StaleOutputError",
    "StaleStageOutput",
    "assert_output_dir_matches_config",
    "find_disabled_stage_outputs",
    "find_inherited_artifacts",
    "format_inherited_artifacts",
    "format_stale_outputs",
]
