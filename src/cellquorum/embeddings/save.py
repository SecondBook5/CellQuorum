"""House-style figure saving for the embeddings stage (mirrors enrichment_viz)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cellquorum.core.stage import StageArtifact
from cellquorum.visualization.style import apply_cellquorum_theme


def apply_theme() -> None:
    """Apply the house theme plus vector-font overrides."""
    apply_cellquorum_theme()
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def save_figure(
    fig: Figure,
    out_dir: str | Path,
    stem: str,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Write ``fig`` to ``out_dir/stem.<fmt>`` for each format, then close it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def figure_artifacts(paths: list[Path], *, name: str, description: str) -> list[StageArtifact]:
    """Wrap saved figure paths as ``kind='figure'`` stage artifacts."""
    return [
        StageArtifact(name=name, path=path, kind="figure", description=description)
        for path in paths
    ]


__all__ = ["apply_theme", "save_figure", "figure_artifacts"]
