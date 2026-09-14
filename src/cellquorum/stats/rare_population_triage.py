"""Triage rare populations: real-but-rare biology vs. noise vs. single-donor artifact.

A raw cell count alone cannot tell these apart: a population of 100 cells
recurring across 9 donors is a real, small population; a population of 100
cells where 90 came from one donor is that one donor's private artifact
wearing the same-sized costume; a population of 2 cells anywhere is neither
-- it is one or two barcodes, not a population. Reading rarity off cell
count alone conflates all three.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def triage_populations(
    obs: pd.DataFrame,
    *,
    label_col: str,
    donor_col: str,
    condition_col: str,
    min_cells: int = 20,
    min_donors: int = 3,
    dominant_donor_veto: float = 0.8,
    reference_condition: str | None = None,
) -> pd.DataFrame:
    """
    Classify every label in ``obs[label_col]`` as noise, single-donor artifact, or candidate.

    Args:
        obs: Obs DataFrame carrying ``label_col``, ``donor_col``, ``condition_col``.
        label_col: Column identifying each cell's population (fine or coarse type).
        donor_col: Column identifying each cell's donor.
        condition_col: Column identifying each cell's condition/treatment arm.
        min_cells: Below this many cells, a label is ``"noise"`` regardless
            of donor spread -- too few barcodes to be a population claim.
        min_donors: A label clearing ``min_cells`` but detected in fewer than
            this many donors is still ``"noise"``: it could be one sample's
            doublets or contamination, not (yet) a replicated claim.
        dominant_donor_veto: A label clearing both floors above is
            ``"single_donor_artifact"`` when one donor supplies at least this
            fraction of its cells -- donor RECURRENCE (``min_donors``) does
            not rule out donor DOMINANCE by a different single donor.
        reference_condition: Which condition to measure enrichment FOR (e.g.
            the disease arm). None auto-picks the cohort's own overall
            majority condition -- a floating reference is rarely what a real
            two-arm study wants (it silently flips which direction "enriched"
            means if the cohort mix isn't 50/50), so pass this explicitly
            whenever there is a specific condition of interest.

    Returns:
        DataFrame with one row per label: ``n_cells``, ``n_donors``,
        ``dominant_donor_frac``, ``condition_enrichment`` (this label's
        fraction in ``reference_condition`` minus the cohort's own overall
        fraction in that condition -- positive means enriched, negative
        means depleted), and ``category`` (``"noise"`` /
        ``"single_donor_artifact"`` / ``"candidate"``). Sorted by category,
        then ascending ``n_cells`` within each category.
    """
    import pandas as pd

    total_n = len(obs)
    condition_counts_overall = obs[condition_col].value_counts()
    reference = (
        reference_condition
        if reference_condition is not None
        else condition_counts_overall.idxmax()
    )
    overall_reference_frac = float(condition_counts_overall.get(reference, 0) / total_n)

    rows: list[dict[str, object]] = []
    for label, group in obs.groupby(label_col, observed=True):
        n_cells = len(group)
        donor_counts = group[donor_col].value_counts()
        n_donors = int((donor_counts > 0).sum())
        dominant_donor_frac = float(donor_counts.max() / n_cells) if n_cells else 0.0
        label_reference_frac = (
            float((group[condition_col] == reference).sum() / n_cells) if n_cells else 0.0
        )

        if n_cells < min_cells or n_donors < min_donors:
            category = "noise"
        elif dominant_donor_frac >= dominant_donor_veto:
            category = "single_donor_artifact"
        else:
            category = "candidate"

        rows.append(
            {
                "label": label,
                "n_cells": n_cells,
                "n_donors": n_donors,
                "dominant_donor_frac": dominant_donor_frac,
                "condition_enrichment": label_reference_frac - overall_reference_frac,
                "category": category,
            }
        )

    result = pd.DataFrame(rows)
    category_order = {"noise": 0, "single_donor_artifact": 1, "candidate": 2}
    result["_category_order"] = result["category"].map(category_order)
    result = result.sort_values(["_category_order", "n_cells"]).drop(columns="_category_order")
    return result.reset_index(drop=True)


__all__ = ["triage_populations"]
