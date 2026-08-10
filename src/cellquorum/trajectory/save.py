"""House-style artifact writers for the trajectory stage (skip-not-crash)."""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad

from cellquorum.core.stage import StageArtifact


def safe_name(group: str) -> str:
    """Filesystem-safe token: any non-alphanumeric run → single underscores."""
    return re.sub(r"[^0-9A-Za-z]", "_", str(group))


def write_velocity_h5ad(
    sub_adata: ad.AnnData, results_dir: Path | str, group: str
) -> tuple[StageArtifact | None, str]:
    """Write a per-group velocity ``.h5ad``; return (artifact | None, note)."""
    path = Path(results_dir) / f"{safe_name(group)}.h5ad"
    try:
        sub_adata.write_h5ad(path)
    except Exception as exc:  # noqa: BLE001 — skip-not-crash
        return None, f"velocity h5ad write failed for '{group}': {exc}"
    artifact = StageArtifact(
        name=f"velocity_{safe_name(group)}",
        path=path,
        kind="h5ad",
        description=f"RNA velocity object for group '{group}'",
    )
    return artifact, f"wrote velocity h5ad for '{group}'"


__all__ = ["safe_name", "write_velocity_h5ad"]
