"""The one place a stage writes a figure and wraps it as an artifact.

Four viz stage families — enrichment_viz, embeddings, trajectory_viz and ccc_viz
— each carried a byte-identical private copy of ``save_figure`` and
``figure_artifacts``. The copies were a bare loop over ``fig.savefig``, and that
cost a real figure: on the LEC arm the velocity stream raised "Can only output
finite numbers in PDF" *partway through writing*, so the run ended up with a
38 KB truncated ``velocity_stream.pdf`` in the figures directory — a file that
looks rendered and will not open — and no ``velocity_stream.png`` at all,
because the exception left the format loop on its first iteration. Meanwhile
:func:`cellquorum.visualization.figstyle.save_figure` had already been hardened
against exactly that, and three of the four copies never saw the fix.

So the write itself lives once, in ``figstyle`` (atomic per format, every format
attempted, figure always closed), and this module is the stage-facing surface
that pairs it with the ``StageArtifact`` wrapper. The four stage modules re-export
from here, so their existing call sites and ``__all__`` are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellquorum.core.stage import StageArtifact
from cellquorum.visualization.figstyle import save_figure

if TYPE_CHECKING:
    from pathlib import Path


def figure_artifacts(
    paths: list[Path],
    *,
    name: str,
    description: str,
) -> list[StageArtifact]:
    """Wrap saved figure paths as ``kind="figure"`` stage artifacts.

    Takes whatever :func:`save_figure` actually wrote, which is why it accepts a
    list rather than a stem plus formats: a format that failed contributes no
    path, and the artifact list must not claim a file that is not on disk.
    """
    return [
        StageArtifact(name=name, path=path, kind="figure", description=description)
        for path in paths
    ]


__all__ = ["save_figure", "figure_artifacts"]
