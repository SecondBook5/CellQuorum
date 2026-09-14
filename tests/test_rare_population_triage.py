"""TDD tests for cellquorum.stats.rare_population_triage.

Distinguishes real-but-rare biology from (a) statistical noise (too few
cells to mean anything) and (b) single-patient artifacts (recurs in only
one donor, so "rare" could just mean "this one sample"), and flags
condition skew for whatever remains -- the actual triage a reader needs
before treating a small population as novel biology.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cellquorum.stats.rare_population_triage import triage_populations


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": (
                ["Noise"]  # 1 cell, noise
                + ["Artifact"] * 30  # 3 donors, but one dominates
                + ["Real"] * 20  # recurrent across donors, condition-skewed
            ),
            "donor_id": (
                ["d1"] + ["d1"] * 25 + ["d2"] * 3 + ["d3"] * 2 + [f"d{i}" for i in range(1, 6)] * 4
            ),
            "condition": (["Normal"] + ["Normal"] * 30 + ["Lymphedema"] * 18 + ["Normal"] * 2),
        }
    )


def test_noise_below_min_cells_is_excluded_as_noise():
    result = triage_populations(
        _obs(),
        label_col="label",
        donor_col="donor_id",
        condition_col="condition",
        min_cells=20,
        min_donors=3,
        dominant_donor_veto=0.8,
    )
    assert result.loc[result["label"] == "Noise", "category"].iloc[0] == "noise"


def test_single_donor_dominated_population_is_flagged_as_artifact():
    result = triage_populations(
        _obs(),
        label_col="label",
        donor_col="donor_id",
        condition_col="condition",
        min_cells=20,
        min_donors=3,
        dominant_donor_veto=0.8,
    )
    row = result.loc[result["label"] == "Artifact"].iloc[0]
    assert row["category"] == "single_donor_artifact"
    assert row["n_donors"] == 3
    assert row["dominant_donor_frac"] == pytest.approx(25 / 30)


def test_donor_recurrent_population_is_a_real_candidate_with_condition_enrichment():
    result = triage_populations(
        _obs(),
        label_col="label",
        donor_col="donor_id",
        condition_col="condition",
        min_cells=20,
        min_donors=3,
        dominant_donor_veto=0.8,
        reference_condition="Lymphedema",
    )
    row = result.loc[result["label"] == "Real"].iloc[0]
    assert row["category"] == "candidate"
    assert row["n_donors"] == 5
    assert row["n_cells"] == 20
    # "Real" is 18/20 = 90% Lymphedema; the cohort overall is 18/51 = 35.3%
    # Lymphedema -- so "Real" is enriched for the disease condition.
    assert row["condition_enrichment"] == pytest.approx(0.9 - 18 / 51)


def test_result_is_sorted_ascending_by_n_cells_within_category():
    obs = pd.DataFrame(
        {
            "label": ["A"] * 25 + ["B"] * 50,
            "donor_id": (["d1", "d2", "d3", "d4", "d5"] * 5)
            + (["d1", "d2", "d3", "d4", "d5"] * 10),
            "condition": ["Normal"] * 75,
        }
    )
    result = triage_populations(
        obs,
        label_col="label",
        donor_col="donor_id",
        condition_col="condition",
        min_cells=20,
        min_donors=3,
        dominant_donor_veto=0.8,
    )
    candidates = result[result["category"] == "candidate"]
    assert list(candidates["n_cells"]) == sorted(candidates["n_cells"])
