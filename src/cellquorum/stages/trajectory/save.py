"""House-style artifact writers for the trajectory stage (skip-not-crash)."""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad

from cellquorum.core.h5ad_io import H5adWriteError, write_h5ad
from cellquorum.core.stage import StageArtifact


def safe_name(group: str) -> str:
    """Filesystem-safe token: any non-alphanumeric run → single underscores."""
    return re.sub(r"[^0-9A-Za-z]", "_", str(group))


def _write(adata: ad.AnnData, path: Path) -> tuple[str | None, str]:
    """Write one object; return ``(error_message | None, suffix)``.

    Every writer here is skip-not-crash, which is right — a missing artifact must
    not destroy a long run — but it also means a writer that trips over something
    serializable-with-effort silently produces nothing. That is not hypothetical:
    one mixed-type obs column upstream once cost this stage all of its h5ads, and
    with them CellRank's velocity kernel and the CytoTRACE kernel, both of which
    consume files this module writes. Going through the shared writer means the
    fixable cases are fixed and reported rather than skipped.

    The suffix carries any coercions into the caller's note, so a reader can see
    that an object was adjusted to be writable instead of guessing.
    """
    try:
        notes = write_h5ad(adata, path)
    except H5adWriteError as exc:
        return str(exc), ""
    return None, f" ({'; '.join(notes)})" if notes else ""


def record_write(
    outcome: tuple[StageArtifact | None, str],
    *,
    notes: list[str],
    warnings: list[str],
) -> StageArtifact | None:
    """File one writer's outcome: a note when it wrote, a WARNING when it did not.

    Skip-not-crash is right, but it only works if the skip is visible. Every one
    of these writers used to report failure as a note, so a run whose h5ad writes
    all failed still showed ``status = success`` with an empty warnings list — the
    one place a reader looks. The classification is mechanical (no artifact means
    no file), so it belongs here rather than at each of the five call sites.

    Args:
        outcome: The ``(artifact | None, note)`` pair a writer returns.
        notes: Collector for the stage's notes; appended on success.
        warnings: Collector for the stage's warnings; appended on failure.

    Returns:
        The artifact, or None when the write failed.
    """
    artifact, note = outcome
    (notes if artifact is not None else warnings).append(note)
    return artifact


def write_velocity_h5ad(
    sub_adata: ad.AnnData, results_dir: Path | str, group: str, stem: str | None = None
) -> tuple[StageArtifact | None, str]:
    """Write a per-group velocity ``.h5ad``; return (artifact | None, note).

    Args:
        sub_adata: The per-group AnnData to write.
        results_dir: Directory to write the h5ad file into.
        group: The raw group name (for metrics/description).
        stem: Optional collision-free filename stem; defaults to safe_name(group).
    """
    file_stem = stem if stem is not None else safe_name(group)
    path = Path(results_dir) / f"{file_stem}.h5ad"
    error, suffix = _write(sub_adata, path)
    if error is not None:
        return None, f"velocity h5ad write failed for '{group}': {error}"
    artifact = StageArtifact(
        name=f"velocity_{file_stem}",
        path=path,
        kind="h5ad",
        description=f"RNA velocity object for group '{group}'",
    )
    return artifact, f"wrote velocity h5ad for '{group}'{suffix}"


def write_whole_object_velocity_h5ad(
    adata: ad.AnnData, results_dir: Path | str
) -> tuple[StageArtifact | None, str]:
    """Write the WHOLE-object velocity ``.h5ad`` (Ms + velocity layers).

    This is the object CellRank's VelocityKernel consumes: velocity computed once
    on the full atlas rather than per group. Written to a fixed
    ``whole_object.h5ad`` stem so consumers can resolve it by convention.

    Args:
        adata: The whole-object velocity AnnData (carries Ms + velocity layers).
        results_dir: Directory to write the h5ad file into.

    Returns (artifact | None, note). Never raises (skip-not-crash).
    """
    path = Path(results_dir) / "whole_object.h5ad"
    error, suffix = _write(adata, path)
    if error is not None:
        return None, f"whole-object velocity h5ad write failed: {error}"
    artifact = StageArtifact(
        name="velocity_whole_object",
        path=path,
        kind="h5ad",
        description="Whole-object RNA velocity for CellRank's VelocityKernel",
    )
    return artifact, f"wrote whole-object velocity h5ad{suffix}"


def write_cellrank_h5ad(
    adata: ad.AnnData, results_dir: Path | str, subsampled: bool = False
) -> tuple[StageArtifact | None, str]:
    """Write the whole-object CellRank fate-mapping ``.h5ad``.

    Args:
        adata: The fate-mapping AnnData to write.
        results_dir: Directory to write the h5ad file into.
        subsampled: Whether ``adata`` is a seeded subsample (drives the label).

    Returns (artifact | None, note). Never raises (skip-not-crash).
    """
    path = Path(results_dir) / "fate_mapping.h5ad"
    error, suffix = _write(adata, path)
    if error is not None:
        return None, f"cellrank h5ad write failed: {error}"
    scope = "subsampled" if subsampled else "whole atlas"
    artifact = StageArtifact(
        name="cellrank_fate_mapping",
        path=path,
        kind="h5ad",
        description=f"CellRank GPCCA fate-mapping object ({scope})",
    )
    return artifact, f"wrote cellrank fate-mapping h5ad{suffix}"


def write_pseudotime_h5ad(
    adata: ad.AnnData, results_dir: Path | str, tool: str, subset: bool = False
) -> tuple[StageArtifact | None, str]:
    """Write the pseudotime ``.h5ad`` for ``tool`` (dpt|palantir).

    Args:
        adata: The pseudotime AnnData to write.
        results_dir: Directory to write the h5ad file into.
        tool: Producer name (drives filename + description).
        subset: Whether ``adata`` is a subset (subsampled / outliers excluded);
            drives the honest scope label.

    Returns (artifact | None, note). Never raises (skip-not-crash).
    """
    stem = safe_name(tool)
    path = Path(results_dir) / f"{stem}_pseudotime.h5ad"
    error, suffix = _write(adata, path)
    if error is not None:
        return None, f"{tool} pseudotime h5ad write failed: {error}"
    scope = "subset" if subset else "whole object"
    artifact = StageArtifact(
        name=f"{stem}_pseudotime",
        path=path,
        kind="h5ad",
        description=f"{tool} pseudotime object ({scope})",
    )
    return artifact, f"wrote {tool} pseudotime h5ad{suffix}"


__all__ = [
    "record_write",
    "safe_name",
    "write_velocity_h5ad",
    "write_whole_object_velocity_h5ad",
    "write_cellrank_h5ad",
    "write_pseudotime_h5ad",
]
