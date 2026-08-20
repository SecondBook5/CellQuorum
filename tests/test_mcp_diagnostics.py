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
