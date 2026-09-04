"""Measure whether a categorical palette's colours are actually distinguishable.

CellQuorum's rule is that colour-vision safety is measured, never eyeballed. That
rule was previously enforced by a script outside this repo
(``skills/dataviz/scripts/validate_palette.py``), which meant the shipped palette
carried a comment asserting it had passed gates nobody could re-run. Auditing the
palette in-tree turns that assertion into a test.

Two distances matter, and conflating them is how a broken palette passes review:

**Adjacent separation** is the distance between colours that sit next to each other
in the palette order. It governs the case a reader meets most often -- two series in
a legend, or a stacked bar's neighbouring bands -- and a long palette can satisfy it
while still containing look-alikes far apart in the order.

**All-pairs separation** is the worst distance between ANY two colours. It is what
"you can tell the categories apart by colour" actually requires, because nothing
stops category 3 and category 11 from being the two a reader must compare. It falls
fast with palette length: perceptual space has room for only so many well-separated
hues, so past roughly eight categories, identity has to come from direct labels or
position rather than from hue alone.

Distances are computed in CAM02-UCS, where Euclidean distance approximates perceived
difference, under normal vision and under full-severity deuteranomaly, protanomaly
and tritanomaly. There is no single blessed threshold in the literature; the numbers
this module reports are meant to be compared against a floor a project states
explicitly, which is what :mod:`tests.test_palette_audit` does.
"""

from __future__ import annotations

# Import dataclass for the frozen audit result.
from dataclasses import dataclass

# Import combinations for the all-pairs sweep.
from itertools import combinations

# Import numpy for the distance arithmetic.
import numpy as np

# Import the CVD simulation and CAM02-UCS conversion.
from colorspacious import cspace_convert

# Import matplotlib's colour parser so any matplotlib-legal colour spec works.
from matplotlib.colors import to_rgb

# The light chart surface every CellQuorum figure renders on. A colour that is
# separated from every other colour but not from the paper it sits on is still
# invisible, so the surface is audited as if it were another palette entry.
CHART_SURFACE: str = "#fcfcfb"

# The colour-vision deficiencies to simulate, at full severity. Anomalous trichromacy
# at severity 100 is the standard stand-in for dichromacy and is the harsher test, so
# passing here covers the milder anomalous cases.
CVD_VIEWS: dict[str, object] = {
    "normal": "sRGB1",
    "deuteranomaly": {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100},
    "protanomaly": {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 100},
    "tritanomaly": {"name": "sRGB1+CVD", "cvd_type": "tritanomaly", "severity": 100},
}


class PaletteAuditError(ValueError):
    """Raised when a palette cannot be audited at all."""


@dataclass(frozen=True)
class ViewSeparation:
    """
    Separation measured under one way of seeing the palette.

    Args:
        view: Name of the simulated vision (e.g. ``deuteranomaly``).
        min_adjacent: Smallest CAM02-UCS distance between consecutive colours.
        min_all_pairs: Smallest distance between any two colours.
        worst_pair: 1-based slot numbers of the closest pair, worst first.
    """

    view: str
    min_adjacent: float
    min_all_pairs: float
    worst_pair: tuple[int, int]


@dataclass(frozen=True)
class PaletteAudit:
    """
    Everything measured about one palette.

    Args:
        n: Number of colours audited.
        views: Separation per simulated vision, keyed by view name.
        min_vs_surface: Smallest distance from any colour to the chart surface.
        lightness_span: Range of CAM02-UCS lightness across the palette. A wide
            span is what keeps the palette legible in greyscale print, where hue is
            gone entirely.
    """

    n: int
    views: dict[str, ViewSeparation]
    min_vs_surface: float
    lightness_span: float

    @property
    def worst_adjacent(self) -> float:
        """Smallest adjacent separation across every simulated vision."""

        return min(view.min_adjacent for view in self.views.values())

    @property
    def worst_all_pairs(self) -> float:
        """Smallest all-pairs separation across every simulated vision."""

        return min(view.min_all_pairs for view in self.views.values())

    @property
    def worst_all_pairs_view(self) -> ViewSeparation:
        """The vision under which the palette does worst on all-pairs."""

        return min(self.views.values(), key=lambda view: view.min_all_pairs)


def _perceptual(colors: list[str], view: str) -> np.ndarray:
    """
    Convert colours into CAM02-UCS coordinates as seen under one vision.

    Args:
        colors: Matplotlib-legal colour specs.
        view: Key of :data:`CVD_VIEWS`.

    Returns:
        Array of shape ``(len(colors), 3)`` in CAM02-UCS.

    Raises:
        PaletteAuditError: If the view is unknown.
    """

    if view not in CVD_VIEWS:
        raise PaletteAuditError(f"unknown vision {view!r}; expected one of {sorted(CVD_VIEWS)}")
    rgb = np.asarray([to_rgb(color) for color in colors], dtype=float)
    return np.asarray(cspace_convert(rgb, CVD_VIEWS[view], "CAM02-UCS"), dtype=float)


def _separation(colors: list[str], view: str) -> ViewSeparation:
    """Measure adjacent and all-pairs separation under one vision."""

    points = _perceptual(colors, view)
    distances = {
        (i, j): float(np.linalg.norm(points[i] - points[j]))
        for i, j in combinations(range(len(colors)), 2)
    }
    worst = min(distances, key=lambda pair: distances[pair])
    adjacent = [distances[(i, i + 1)] for i in range(len(colors) - 1)]
    return ViewSeparation(
        view=view,
        min_adjacent=min(adjacent),
        min_all_pairs=distances[worst],
        worst_pair=(worst[0] + 1, worst[1] + 1),
    )


def audit_palette(colors: list[str], *, surface: str = CHART_SURFACE) -> PaletteAudit:
    """
    Measure a categorical palette's separation under normal and CVD vision.

    Args:
        colors: The palette, in the order it will be assigned to categories.
        surface: The chart background the palette is drawn on.

    Returns:
        The measured audit.

    Raises:
        PaletteAuditError: If fewer than two colours are given, since separation is
            a property of a pair.
    """

    if len(colors) < 2:
        raise PaletteAuditError(
            f"a palette needs at least 2 colours to have any separation; got {len(colors)}"
        )

    views = {view: _separation(colors, view) for view in CVD_VIEWS}

    # The surface is audited under normal vision only: a colour that vanishes into
    # the paper does so because of its lightness, which CVD simulation preserves.
    points = _perceptual([*colors, surface], "normal")
    min_vs_surface = float(min(np.linalg.norm(point - points[-1]) for point in points[:-1]))
    lightness = points[:-1, 0]

    return PaletteAudit(
        n=len(colors),
        views=views,
        min_vs_surface=min_vs_surface,
        lightness_span=float(lightness.max() - lightness.min()),
    )


def longest_safe_prefix(colors: list[str], *, threshold: float) -> int:
    """
    Return how many leading colours stay all-pairs separated under every vision.

    A palette is used by taking the first ``n`` entries for ``n`` categories, so the
    honest question is not whether the whole list is safe but how far down the list
    a figure can go before colour alone stops carrying identity. Past that point a
    figure needs direct labels or position, and saying so is more useful than a
    pass/fail verdict on the full palette.

    Args:
        colors: The palette in assignment order.
        threshold: Minimum acceptable CAM02-UCS all-pairs distance.

    Returns:
        The largest ``n`` (at least 0) whose prefix clears the threshold in every
        simulated vision. 0 means even the first two colours collide.
    """

    safe = 0
    for n in range(2, len(colors) + 1):
        audit = audit_palette(colors[:n])
        if audit.worst_all_pairs < threshold:
            break
        safe = n
    return safe


def format_audit(audit: PaletteAudit) -> str:
    """
    Render an audit as a short table for logs and review notes.

    Args:
        audit: The measured audit.

    Returns:
        A multi-line string; the caller decides where it goes.
    """

    lines = [
        f"palette of {audit.n} colours, CAM02-UCS separation",
        f"{'view':<16}{'min adjacent':>14}{'min all-pairs':>15}  worst pair",
    ]
    for view in audit.views.values():
        lines.append(
            f"{view.view:<16}{view.min_adjacent:>14.1f}{view.min_all_pairs:>15.1f}"
            f"  slots {view.worst_pair[0]} & {view.worst_pair[1]}"
        )
    lines.append(f"lightness span {audit.lightness_span:.1f} (greyscale legibility)")
    lines.append(f"min distance to surface {audit.min_vs_surface:.1f}")
    return "\n".join(lines)


__all__ = [
    "CHART_SURFACE",
    "CVD_VIEWS",
    "PaletteAudit",
    "PaletteAuditError",
    "ViewSeparation",
    "audit_palette",
    "format_audit",
    "longest_safe_prefix",
]
