"""TDD tests for cellquorum.stats.ambient_specificity.

Cross-compartment ambient contamination (SoupX runs per-library, before any cell-type
label exists, so it cannot catch a gene that peaks in one cell type but leaks into
another at low, non-specific levels — e.g. epithelial keratins showing up in
fibroblasts). ``ambient_gene_mask`` flags a gene as ambient for a target group only
when it BOTH peaks in a foreign group AND is expressed there at a low fraction of its
peak, so genuinely shared biology (a marker that peaks in one closely related cell
type but is real in another) survives via an optional coarser ``group_map``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cellquorum.stats.ambient_specificity import ambient_gene_mask


def _profile() -> pd.DataFrame:
    """genes (rows) x groups (cols), mean expression.

    KRT10: peaks in Keratinocytes (5.0); Fibroblasts=0.5 -> ratio 0.10 (foreign, weak).
    ACTA2: peaks in Pericyte_SMC (4.0); Fibroblasts=3.0 -> ratio 0.75 (foreign, strong).
    UNDETECTED: zero everywhere -> no peak to assess.
    BORDERLINE: peaks in Keratinocytes (1.0); Fibroblasts=0.9 -> ratio 0.90 (foreign, strong).
    SPECIFIC_ELSEWHERE: peaks in Keratinocytes (4.0); Fibroblasts=1.0 -> ratio 0.25 (== threshold).
    """
    return pd.DataFrame(
        {
            "Keratinocytes": [5.0, 0.2, 0.0, 1.0, 4.0],
            "Fibroblasts": [0.5, 3.0, 0.0, 0.9, 1.0],
            "Pericyte_SMC": [0.1, 4.0, 0.0, 0.1, 0.5],
        },
        index=["KRT10", "ACTA2", "UNDETECTED", "BORDERLINE", "SPECIFIC_ELSEWHERE"],
    )


def test_drops_a_foreign_low_specificity_gene():
    mask = ambient_gene_mask(_profile(), target_group="Fibroblasts", spec_threshold=0.25)
    assert mask["KRT10"] == False  # noqa: E712


def test_keeps_a_foreign_but_high_specificity_gene():
    """ACTA2 peaks in Pericyte_SMC (foreign to Fibroblasts) but at 75% of its peak —
    the specificity gate alone protects real shared biology even with no group_map."""
    mask = ambient_gene_mask(_profile(), target_group="Fibroblasts", spec_threshold=0.25)
    assert mask["ACTA2"]


def test_group_map_reclassifies_a_foreign_group_as_the_same_group():
    """Mapping Fibroblasts and Pericyte_SMC to the same coarse group makes ACTA2's
    peak group non-foreign outright, regardless of its specificity ratio."""
    group_map = {
        "Fibroblasts": "mesenchymal",
        "Pericyte_SMC": "mesenchymal",
        "Keratinocytes": "epithelial",
    }
    mask = ambient_gene_mask(
        _profile(), target_group="Fibroblasts", spec_threshold=0.25, group_map=group_map
    )
    assert mask["ACTA2"]


def test_threshold_is_exclusive_at_the_boundary():
    """ratio == spec_threshold exactly is NOT below it -> kept, not ambient."""
    mask = ambient_gene_mask(_profile(), target_group="Fibroblasts", spec_threshold=0.25)
    assert mask["SPECIFIC_ELSEWHERE"]


def test_undetected_gene_everywhere_is_kept_not_flagged_ambient():
    mask = ambient_gene_mask(_profile(), target_group="Fibroblasts", spec_threshold=0.25)
    assert mask["UNDETECTED"]


def test_gene_peaking_in_the_target_group_itself_is_never_ambient():
    profile = _profile()
    mask = ambient_gene_mask(profile, target_group="Keratinocytes", spec_threshold=0.25)
    assert mask["KRT10"]  # KRT10 peaks IN Keratinocytes, the target itself


def test_raises_for_unknown_target_group():
    with pytest.raises(ValueError, match="target_group"):
        ambient_gene_mask(_profile(), target_group="NotAGroup")


def test_unmapped_group_falls_back_to_identity_in_group_map():
    """A group_map naming only SOME groups leaves the rest as their own group."""
    group_map = {"Fibroblasts": "mesenchymal"}  # Keratinocytes left unmapped
    mask = ambient_gene_mask(
        _profile(), target_group="Fibroblasts", spec_threshold=0.25, group_map=group_map
    )
    # Keratinocytes (unmapped -> identity) is still foreign to 'mesenchymal', and
    # KRT10 is still weak there -> still ambient.
    assert not mask["KRT10"]


def test_mask_is_a_boolean_series_indexed_like_the_profile():
    mask = ambient_gene_mask(_profile(), target_group="Fibroblasts")
    assert list(mask.index) == list(_profile().index)
    assert mask.dtype == bool
