"""TDD tests for cellquorum.stats.reference_identity.

Generalizes a real forensic diagnostic from the Mehrara analysis
(umap_blob_diagnostic.py) into reusable CellQuorum code: which cells does the
reference-mapping classifier itself have low confidence in (an "ambiguous
identity" blob, independent of the three-way consensus check), and which
cells does it think belong to some OTHER coarse group than the one they were
assigned -- generalized across every coarse category, not hardcoded to one
cell type the way the original script's KC-only "reassembly sweep" was.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cellquorum.stats.reference_identity import (
    ambiguous_identity_mask,
    blob_composition,
    blob_qc_comparison,
    coarse_reassembly_candidates,
    top_reference_probabilities,
)


def test_top_reference_probabilities_returns_top_and_second_per_cell():
    probs = pd.DataFrame(
        {
            "Fibroblast_A": [0.7, 0.2, 0.34],
            "Fibroblast_B": [0.2, 0.3, 0.33],
            "DC_1": [0.1, 0.5, 0.33],
        },
        index=["c1", "c2", "c3"],
    )

    result = top_reference_probabilities(probs)

    assert list(result.columns) == ["top_label", "top_prob", "second_label", "second_prob"]
    assert result.loc["c1", "top_label"] == "Fibroblast_A"
    assert result.loc["c1", "top_prob"] == pytest.approx(0.7)
    assert result.loc["c1", "second_label"] == "Fibroblast_B"
    assert result.loc["c1", "second_prob"] == pytest.approx(0.2)
    assert result.loc["c2", "top_label"] == "DC_1"


def test_ambiguous_identity_mask_flags_low_top_confidence():
    top_prob = pd.Series([0.9, 0.4, 0.5, 0.49], index=["a", "b", "c", "d"])

    mask = ambiguous_identity_mask(top_prob, threshold=0.5)

    assert mask.tolist() == [False, True, False, True]


def test_blob_composition_reports_cell_type_fractions_within_the_blob():
    obs = pd.DataFrame(
        {"cell_type": ["Fibroblasts", "Fibroblasts", "DC", "T/NK"]},
        index=["a", "b", "c", "d"],
    )
    blob = pd.Series([True, True, True, False], index=obs.index)

    result = blob_composition(obs, blob, "cell_type")

    assert result["Fibroblasts"] == pytest.approx(2 / 3)
    assert result["DC"] == pytest.approx(1 / 3)
    assert "T/NK" not in result.index


def test_blob_qc_comparison_reports_blob_vs_rest_means():
    obs = pd.DataFrame(
        {
            "doublet_score": [0.8, 0.9, 0.1, 0.2],
            "n_genes": [500, 600, 3000, 3200],
        },
        index=["a", "b", "c", "d"],
    )
    blob = pd.Series([True, True, False, False], index=obs.index)

    result = blob_qc_comparison(obs, blob, ["doublet_score", "n_genes"])

    assert result.loc["doublet_score", "blob_mean"] == pytest.approx(0.85)
    assert result.loc["doublet_score", "rest_mean"] == pytest.approx(0.15)
    assert result.loc["n_genes", "blob_mean"] == pytest.approx(550.0)
    assert result.loc["n_genes", "rest_mean"] == pytest.approx(3100.0)


def test_coarse_reassembly_candidates_is_generalized_not_hardcoded_to_one_type():
    """The original script only ever checked Keratinocytes. This must work for
    every coarse group the hierarchy defines, from the SAME call."""
    fine_labels = ["Fibroblast_A", "Fibroblast_B", "DC_1", "DC_2"]
    probs = pd.DataFrame(
        {
            "Fibroblast_A": [0.6, 0.1, 0.05, 0.1],
            "Fibroblast_B": [0.1, 0.1, 0.05, 0.1],
            "DC_1": [0.2, 0.7, 0.1, 0.3],
            "DC_2": [0.1, 0.1, 0.1, 0.2],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    fine_to_coarse = {
        "Fibroblast_A": "Fibroblasts",
        "Fibroblast_B": "Fibroblasts",
        "DC_1": "DC",
        "DC_2": "DC",
    }
    # c1 is assigned DC but the classifier's Fibroblast mass (0.7) says otherwise.
    # c2 is assigned Fibroblasts but the classifier's DC mass (0.8) says otherwise.
    # c3/c4 have no group above threshold -> not candidates for anything.
    assigned_coarse = pd.Series(["DC", "Fibroblasts", "DC", "Fibroblasts"], index=probs.index)

    result = coarse_reassembly_candidates(
        probs, fine_labels, fine_to_coarse, assigned_coarse, threshold=0.5
    )

    assert set(result["candidate_group"]) == {"Fibroblasts", "DC"}
    fib_candidates = result[result["candidate_group"] == "Fibroblasts"]
    assert list(fib_candidates["cell"]) == ["c1"]
    dc_candidates = result[result["candidate_group"] == "DC"]
    assert list(dc_candidates["cell"]) == ["c2"]


def test_coarse_reassembly_candidates_empty_when_nothing_qualifies():
    probs = pd.DataFrame({"A": [0.3], "B": [0.3]}, index=["c1"])
    fine_to_coarse = {"A": "GroupA", "B": "GroupB"}
    assigned_coarse = pd.Series(["GroupA"], index=["c1"])

    result = coarse_reassembly_candidates(
        probs, ["A", "B"], fine_to_coarse, assigned_coarse, threshold=0.5
    )

    assert result.empty
    assert list(result.columns) == ["cell", "candidate_group", "candidate_prob", "assigned_group"]
