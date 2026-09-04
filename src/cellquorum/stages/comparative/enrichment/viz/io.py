"""I/O helpers for enrichment visualization: CSV discovery and figure saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

from cellquorum.visualization.figio import figure_artifacts, save_figure
from cellquorum.visualization.figstyle import apply_cellquorum_theme


def collections_from_glob(results_dir: Path, prefix: str, suffix: str = ".csv") -> list[str]:
    """Return collection/resource names parsed from files matching prefix*suffix."""
    names = []
    for path in sorted(results_dir.glob(f"{prefix}*{suffix}")):
        names.append(path.name[len(prefix) : -len(suffix)])
    return names


def apply_theme() -> None:
    """Apply the house theme plus enrichment-viz vector-font overrides."""
    apply_cellquorum_theme()
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


# save_figure/figure_artifacts are re-exported, not redefined: the local copy was
# a bare savefig loop that left truncated files behind and abandoned the remaining
# formats when one raised mid-write. See visualization.figio.


__all__ = ["collections_from_glob", "apply_theme", "save_figure", "figure_artifacts"]
