"""Tests for the categorical palette audit, and for the shipped palette's contract.

The point of these tests is not to admire the palette. It is to make its guarantee a
thing that can fail. Before this, ``figstyle.CATEGORICAL_PALETTE`` carried a comment
saying it had been validated by a script that lives outside this repository, so the
claim could not be checked and -- as the last test in this file records -- part of it
was not true.
"""

from __future__ import annotations

import pytest

from cellquorum.visualization.figstyle import CATEGORICAL_PALETTE
from cellquorum.visualization.palette_audit import (
    PaletteAuditError,
    _perceptual,
    audit_palette,
    format_audit,
    longest_safe_prefix,
)

# The number of colours the palette promises to keep separated from EACH OTHER, not
# merely from their neighbours in the order. Eight is where the dataviz reference
# theme ends and the overflow tier begins.
CORE_SIZE = 8

# Floor the core eight clear under every simulated vision. This is the MEASURED value
# (6.97, orange vs red under tritanomaly), floored rather than rounded up: raising it to
# a prettier 7 or an aspirational 10 would make this test a wish instead of a
# description of the palette that ships. The number is also the honest ceiling on what
# colour alone can do here -- it means two categories are distinguishable side by side,
# not that a lone mark is identifiable, which is why every categorical figure in the
# engine also carries a legend or direct labels.
CORE_ALL_PAIRS_FLOOR = 6.9

# Floor for CONSECUTIVE colours across the whole 18, which is the case a stacked bar
# or an ordered legend actually presents.
ADJACENT_FLOOR = 8.0


def test_identical_colours_have_no_separation() -> None:
    """
    Verify the metric reports zero for a duplicated colour.

    A palette audit that cannot detect an outright duplicate would certify anything.
    """

    audit = audit_palette(["#2a78d6", "#2a78d6"])

    assert audit.worst_all_pairs == pytest.approx(0.0, abs=1e-6)


def test_a_classic_red_green_pair_fails_under_deuteranomaly_only() -> None:
    """
    Verify CVD simulation is actually applied, not merely named.

    Red and green are the canonical pair that looks fine to a normal-vision reviewer
    and collapses for roughly one man in twelve. If the simulated views returned the
    same numbers as normal vision, every test here would pass vacuously.
    """

    audit = audit_palette(["#008300", "#c1121f"])

    assert audit.views["normal"].min_all_pairs > 20
    assert audit.views["deuteranomaly"].min_all_pairs < 5
    assert audit.worst_all_pairs_view.view == "deuteranomaly"


def test_a_single_colour_cannot_be_audited() -> None:
    """
    Verify a one-colour palette is refused rather than scored.

    Separation is a property of a pair; returning something like infinity would let a
    caller "pass" the gate by shrinking the palette.
    """

    with pytest.raises(PaletteAuditError, match="at least 2 colours"):
        audit_palette(["#2a78d6"])


def test_an_unknown_vision_is_refused() -> None:
    """
    Verify a typo'd view name fails instead of silently measuring normal vision.

    Reaches into the conversion helper on purpose: it is the single place a vision
    name is honoured, and quietly falling back to normal vision there is how a palette
    would get certified CVD-safe without ever having been simulated.
    """

    with pytest.raises(PaletteAuditError, match="unknown vision"):
        _perceptual(["#2a78d6", "#eb6834"], "deuteranopia")  # the real key is -anomaly


def test_longest_safe_prefix_stops_at_the_first_collision() -> None:
    """
    Verify the prefix search reports where colour alone stops carrying identity.

    A palette is consumed as "the first n entries", so this is the number a figure
    author needs; a pass/fail verdict on the whole list hides it.
    """

    # Third colour is a near-duplicate of the first, so only the leading pair is safe.
    palette = ["#2a78d6", "#eb6834", "#2b79d7", "#008300"]

    assert longest_safe_prefix(palette, threshold=10.0) == 2


def test_the_core_eight_keep_every_pair_apart() -> None:
    """
    Verify the dataviz reference theme is all-pairs separated under every vision.

    This is the palette's real load-bearing promise: up to eight categories can be
    told apart by colour, in any pairing, including by a reader with any of the three
    simulated deficiencies.
    """

    audit = audit_palette(list(CATEGORICAL_PALETTE[:CORE_SIZE]))

    assert audit.worst_all_pairs >= CORE_ALL_PAIRS_FLOOR, format_audit(audit)
    assert longest_safe_prefix(list(CATEGORICAL_PALETTE), threshold=CORE_ALL_PAIRS_FLOOR) >= (
        CORE_SIZE
    )


def test_the_full_palette_keeps_consecutive_colours_apart() -> None:
    """
    Verify all 18 slots are separated from their NEIGHBOURS under every vision.

    This is the guarantee the overflow tier was ordered to satisfy, and it is what
    makes an 18-band stacked bar or an ordered legend readable.
    """

    audit = audit_palette(list(CATEGORICAL_PALETTE))

    assert audit.worst_adjacent >= ADJACENT_FLOOR, format_audit(audit)


def test_the_full_palette_is_not_all_pairs_safe_and_that_is_recorded() -> None:
    """
    Pin the measured fact that the overflow tier contains look-alike pairs.

    ``figstyle`` claims the whole 18 clears the CVD-separation gate. Measured, it does
    not: the closest pair is 0.7 apart under deuteranomaly (slots 6 and 12, the
    textbook green/crimson collision) and 3.5 apart under NORMAL vision (slots 3 and
    11, aqua and emerald -- a near-duplicate every reader sees). This test exists so
    that fact is a checked statement rather than a comment, and so the day someone
    repairs the tier this test fails and forces the docs to be updated with it. A
    repaired tier reaching 7.0 has been demonstrated, but changing these hues repaints
    every published figure, so it is a deliberate decision and not a silent fix.

    Perceptual space simply does not hold 18 mutually distinct hues at usable
    lightness; past about eight categories, identity has to come from direct labels or
    position, which is why every categorical figure in the engine carries them.
    """

    audit = audit_palette(list(CATEGORICAL_PALETTE))

    assert audit.worst_all_pairs < CORE_ALL_PAIRS_FLOOR, format_audit(audit)
    assert audit.worst_all_pairs_view.view == "deuteranomaly"
    assert audit.worst_all_pairs_view.worst_pair == (6, 12)


def test_no_palette_colour_disappears_into_the_chart_surface() -> None:
    """
    Verify every colour is separated from the paper it is drawn on.

    A palette can be internally perfect and still produce an invisible series: the
    surface is effectively a nineteenth category that every mark must clear.
    """

    audit = audit_palette(list(CATEGORICAL_PALETTE))

    assert audit.min_vs_surface >= 20.0, format_audit(audit)


def test_the_palette_spans_enough_lightness_for_greyscale_print() -> None:
    """
    Verify the palette keeps a wide lightness range.

    Journals still print in greyscale, where hue is gone entirely and lightness is
    the only channel left. A palette tuned purely for hue separation can collapse to
    a single grey.
    """

    audit = audit_palette(list(CATEGORICAL_PALETTE))

    assert audit.lightness_span >= 30.0, format_audit(audit)
