"""Shared, IO-only CSV discovery for the enrichment-viz methods."""

from __future__ import annotations

from pathlib import Path


def collections_from_glob(results_dir: Path, prefix: str, suffix: str = ".csv") -> list[str]:
    """Return collection/resource names parsed from files matching prefix*suffix."""
    names = []
    for path in sorted(results_dir.glob(f"{prefix}*{suffix}")):
        names.append(path.name[len(prefix) : -len(suffix)])
    return names


__all__ = ["collections_from_glob"]
