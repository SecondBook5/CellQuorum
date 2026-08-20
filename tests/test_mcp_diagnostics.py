"""Pure-math diagnostics for multicellular programs."""

from __future__ import annotations

import pandas as pd

from cellquorum.multicellular_programs.diagnostics import (
    donor_support,
    match_program_loadings,
    program_stability,
)


def test_donor_support_counts_distinct_active_donors():
    # MCP1 active (score>mean) in cells of donors d1,d2; MCP2 active only in d1.
    scores = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c", "d", "e", "f"],
            "program": ["MCP1"] * 3 + ["MCP2"] * 3,
            "score": [1.0, 1.0, -2.0, 1.0, -1.0, -1.0],
        }
    )
    donor_map = {"a": "d1", "b": "d2", "c": "d1", "d": "d1", "e": "d2", "f": "d3"}
    out = donor_support(scores, donor_map, donor_support_min=2)
    out = out.set_index("program")
    assert out.loc["MCP1", "n_donors"] == 2 and bool(out.loc["MCP1", "supported"]) is True
    assert out.loc["MCP2", "n_donors"] == 1 and bool(out.loc["MCP2", "supported"]) is False


def test_donor_support_matches_numeric_cell_ids_to_str_donor_map():
    # Regression: scores read back from CSV carry int cell_ids when barcodes are
    # purely numeric, while donor_map is str-keyed. The join must coerce, not
    # silently report 0 donors (the wrong-but-plausible failure mode).
    scores = pd.DataFrame(
        {
            "cell_id": [0, 1, 2],  # int, as pandas infers from numeric barcodes
            "program": ["MCP1"] * 3,
            "score": [1.0, 1.0, -2.0],
        }
    )
    donor_map = {"0": "d1", "1": "d2", "2": "d1"}
    out = donor_support(scores, donor_map, donor_support_min=2).set_index("program")
    assert out.loc["MCP1", "n_donors"] == 2 and bool(out.loc["MCP1", "supported"]) is True


def test_match_program_loadings_ignores_sign_flip():
    # MCP loading sign is arbitrary between independent runs; a sign-flipped copy
    # of the same program must still score as fully reproducible (|r| == 1), not 0.
    full = pd.DataFrame(
        {
            "program": ["MCP1"] * 3,
            "cell_type": ["A", "A", "B"],
            "gene": ["G1", "G2", "G1"],
            "loading": [0.9, 0.1, 0.5],
        }
    )
    flipped = full.copy()
    flipped["loading"] = -flipped["loading"]
    matches = match_program_loadings(full, flipped)
    assert abs(matches["MCP1"] - 1.0) < 1e-9


def test_match_and_stability_identical_is_one():
    full = pd.DataFrame(
        {
            "program": ["MCP1"] * 3,
            "cell_type": ["A", "A", "B"],
            "gene": ["G1", "G2", "G1"],
            "loading": [0.9, 0.1, 0.5],
        }
    )
    matches = match_program_loadings(full, full.copy())
    assert abs(matches["MCP1"] - 1.0) < 1e-9
    stab = program_stability([matches, matches]).set_index("program")
    assert abs(stab.loc["MCP1", "mean_stability"] - 1.0) < 1e-9
    assert int(stab.loc["MCP1", "n_resamples"]) == 2
