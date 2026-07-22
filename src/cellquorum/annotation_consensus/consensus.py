"""Pure consensus-reconciliation functions (no AnnData; fully unit-testable)."""

from __future__ import annotations

from collections import Counter


def normalize_label(label: object, aliases: dict[str, str]) -> str | None:
    """
    Map a raw per-method label onto its canonical backbone label.

    Args:
        label: Raw label value (string, None, or NA-like).
        aliases: Mapping from raw label to canonical label.

    Returns:
        The canonical label, the original label when no alias applies, or None
        when the input is missing.
    """

    # Treat None and NA-like values (including pandas NaN) as missing.
    if label is None:
        return None
    text = str(label)
    if text in {"", "nan", "NaN", "None", "NA", "<NA>"}:
        return None

    # Apply the alias map; pass through labels with no alias.
    return aliases.get(text, text)


def reconcile_votes(
    votes: list[str | None],
    *,
    min_agree_fraction: float,
    high_confidence_all: bool,
) -> tuple[str | None, str, bool]:
    """
    Reconcile one cell's per-method votes into a label, tier, and review flag.

    Args:
        votes: Per-method labels for one cell (already normalized); None = the
            method produced no label for this cell (e.g. it skipped).
        min_agree_fraction: Fraction of non-missing votes that must back the
            winning label to call a (non-unanimous) majority.
        high_confidence_all: Whether unanimous agreement (≥2 votes) is 'high'.

    Returns:
        (consensus_label, confidence_tier, needs_review). Tiers are
        'high' | 'medium' | 'low'; needs_review is True iff tier == 'low'.
    """

    # Drop missing votes; count the rest.
    present = [v for v in votes if v is not None]
    n = len(present)

    # No information: low confidence, needs review, no label.
    if n == 0:
        return None, "low", True

    # Fewer than two votes cannot be corroborated: low confidence.
    if n < 2:
        return present[0], "low", True

    # Tally and take the most common label.
    counts = Counter(present)
    winner, winner_count = counts.most_common(1)[0]
    fraction = winner_count / n

    # Unanimous across all non-missing votes -> high.
    if high_confidence_all and winner_count == n:
        return winner, "high", False

    # A qualifying majority -> medium.
    if fraction >= min_agree_fraction and _is_unique_leader(counts, winner_count):
        return winner, "medium", False

    # Otherwise the cell is contested -> low, needs review.
    return winner, "low", True


def _is_unique_leader(counts: Counter, winner_count: int) -> bool:
    """
    Return whether exactly one label holds the top count (no tie for first).

    Args:
        counts: Vote tally.
        winner_count: The top count.

    Returns:
        True when a single label is the strict leader.
    """

    # A tie for the top count is not a majority we can trust.
    return sum(1 for c in counts.values() if c == winner_count) == 1


__all__ = ["normalize_label", "reconcile_votes"]
