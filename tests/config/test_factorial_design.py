"""Runtime validation of a factorial / repeated-measures design against specimens.

All analysis is on a one-row-per-specimen table. The reference fixture is a balanced
6-subject / 12-specimen split-plot: ``arm`` varies between subjects, ``site`` within.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cellquorum.config.design import FactorialDesignConfig, build_design_matrix
from cellquorum.config.factorial_design import (
    analyze_factorial_design,
    required_specimen_columns,
    validate_factorial_design,
)
from cellquorum.core.exceptions import CellQuorumConfigError, CellQuorumDataError
from cellquorum.stages.comparative.differential_expression.pseudobulk import (
    build_specimen_universe,
)

SUBJECTS = {"C1": "ctl", "C2": "ctl", "C3": "ctl", "P2": "trt", "P3": "trt", "P4": "trt"}
SITES = {"colon": "C", "jej": "J"}


def _specimens() -> pd.DataFrame:
    rows = [
        {"specimen": f"{s}{code}", "subject": s, "arm": arm, "site": site}
        for s, arm in SUBJECTS.items()
        for site, code in SITES.items()
    ]
    return pd.DataFrame(rows).set_index("specimen")


def _design(**overrides) -> FactorialDesignConfig:
    base = {
        "subject_col": "subject",
        "factors": {
            "arm": {"reference": "ctl", "scope": "between_subject"},
            "site": {"reference": "colon", "scope": "within_subject"},
        },
        "interactions": [["arm", "site"]],
        "random_effects": ["subject"],
    }
    base.update(overrides)
    return FactorialDesignConfig.model_validate(base)


def _analyze(table: pd.DataFrame, design: FactorialDesignConfig | None = None):
    return analyze_factorial_design(
        table, design=design or _design(), sample_col="specimen", subject_col="subject"
    )


def _validate(table: pd.DataFrame, design: FactorialDesignConfig | None = None):
    return validate_factorial_design(
        table, design=design or _design(), sample_col="specimen", subject_col="subject"
    )


# --- the balanced reference design ---------------------------------------------------


def test_balanced_split_plot_report():
    rep = _validate(_specimens())

    assert rep.ok and not rep.errors and not rep.warnings
    assert (rep.n_samples, rep.n_subjects) == (12, 6)

    arm = rep.factor("arm")
    assert arm.observed_levels == ["ctl", "trt"]
    assert arm.n_samples_by_level == {"ctl": 6, "trt": 6}
    assert arm.n_subjects_by_level == {"ctl": 3, "trt": 3}
    assert arm.constant_within_subject is True

    site = rep.factor("site")
    assert site.n_samples_by_level == {"colon": 6, "jej": 6}
    assert site.n_subjects_by_level == {"colon": 6, "jej": 6}
    assert site.n_subjects_varying == 6
    assert sorted(site.complete_subjects) == sorted(SUBJECTS)
    assert site.incomplete_subjects == {}

    (inter,) = rep.interactions
    assert inter.factors == ("arm", "site")
    assert len(inter.cells) == 4 and inter.empty_cells == []
    assert all(c["n_samples"] == 3 and c["n_subjects"] == 3 for c in inter.cells)

    (re,) = rep.random_effects
    assert (re.column, re.n_groups, re.n_groups_with_repeats) == ("subject", 6, 6)
    assert (re.min_specimens, re.max_specimens) == (2, 2)

    # Fixed effects only: intercept, arm, site, arm:site. Subject is NOT a fixed term,
    # so the between-subject factor is not aliased by it.
    assert rep.full_rank and rep.fixed_effect_rank == 4
    assert rep.fixed_effect_columns == [
        "Intercept",
        "arm_trt",
        "site_jej",
        "arm_trt:site_jej",
    ]


def test_fixed_subject_term_would_alias_the_between_subject_factor():
    # The reason random effects stay out of the fixed matrix: modelling subject as a
    # fixed factor next to a between-subject factor is rank-deficient.
    design = _design(
        factors={
            "subject": {"reference": "C1"},
            "arm": {"reference": "ctl", "scope": "between_subject"},
            "site": {"reference": "colon", "scope": "within_subject"},
        },
        interactions=[],
    )
    rep = _analyze(_specimens(), design)
    assert not rep.full_rank
    assert any("rank-deficient" in e for e in rep.errors)


def test_report_is_json_safe_and_round_trips():
    rep = _validate(_specimens())
    payload = json.loads(json.dumps(rep.to_dict()))
    assert payload["n_samples"] == 12 and payload["n_subjects"] == 6
    assert payload["ok"] is True and payload["full_rank"] is True
    assert payload["factors"][0]["name"] == "arm"
    assert payload["interactions"][0]["empty_cells"] == []


def test_stage2_universe_feeds_the_analyzer():
    # Cell-level obs -> Stage 2 universe -> design report; 20 cells/specimen never
    # inflate the specimen count.
    table = _specimens().reset_index()
    obs = table.loc[table.index.repeat(20)].reset_index(drop=True)
    design = _design()
    cols = required_specimen_columns(design, sample_col="specimen", subject_col="subject")
    universe = build_specimen_universe(obs, sample_col="specimen", metadata_cols=cols)
    rep = analyze_factorial_design(
        universe, design=design, sample_col="specimen", subject_col="subject"
    )
    assert (rep.n_samples, rep.n_subjects) == (12, 6)


# --- factor roles ---------------------------------------------------------------------


def test_between_subject_violation_is_a_hard_error_naming_factor_subject_levels():
    table = _specimens()
    table.loc["C1J", "arm"] = "trt"
    with pytest.raises(CellQuorumConfigError) as exc:
        _validate(table)
    msg = str(exc.value)
    assert "'arm'" in msg and "between_subject" in msg
    assert "'C1'" in msg and "ctl" in msg and "trt" in msg
    # A semantic mismatch, reported as such -- not (only) as rank deficiency.
    assert "varies within a subject" in msg


def test_between_subject_factor_need_not_vary_even_with_repeated_specimens():
    rep = _validate(_specimens())
    assert rep.factor("arm").n_subjects_varying == 0


def test_within_subject_factor_that_never_varies_is_a_hard_error():
    # Every subject contributes one site only (site is really between-subject).
    table = _specimens()
    table = table[table.index.str.endswith("C")].copy()
    with pytest.raises(CellQuorumConfigError, match="constant within every subject"):
        _validate(table, _design(interactions=[]))


def test_incomplete_repeated_design_is_reported_not_fatal():
    table = _specimens().drop(index="C1J")
    rep = _validate(table)  # must not raise
    site = rep.factor("site")
    assert site.incomplete_subjects == {"C1": ["jej"]}
    assert "C1" not in site.complete_subjects and len(site.complete_subjects) == 5
    assert any("C1" in w and "jej" in w for w in rep.warnings)
    assert rep.n_samples == 11 and rep.n_subjects == 6


def test_sample_level_factor_makes_no_subject_claim():
    table = _specimens()
    table["batch"] = ["b1", "b2", "b2", "b1"] * 3  # crosses both arm and site
    design = _design(
        factors={
            "arm": {"reference": "ctl", "scope": "between_subject"},
            "site": {"reference": "colon", "scope": "within_subject"},
            "batch": {"reference": "b1"},
        }
    )
    rep = _validate(table, design)
    assert rep.factor("batch").scope == "sample_level"
    assert rep.full_rank


def test_scope_claim_without_subject_column_is_an_error():
    rep = analyze_factorial_design(
        _specimens(), design=_design(random_effects=[]), sample_col="specimen", subject_col=None
    )
    assert any("no subject column" in e for e in rep.errors)


# --- references ------------------------------------------------------------------------


def test_reference_level_absent_from_data_is_a_hard_error_listing_observed_levels():
    design = _design(
        factors={
            "arm": {"reference": "WT", "scope": "between_subject"},
            "site": {"reference": "colon", "scope": "within_subject"},
        }
    )
    with pytest.raises(CellQuorumConfigError) as exc:
        _validate(_specimens(), design)
    assert "'WT'" in str(exc.value) and "['ctl', 'trt']" in str(exc.value)


def test_declared_reference_controls_the_coding():
    design = _design(
        factors={
            "arm": {"reference": "trt", "scope": "between_subject"},
            "site": {"reference": "jej", "scope": "within_subject"},
        }
    )
    rep = _validate(_specimens(), design)
    assert rep.fixed_effect_columns == [
        "Intercept",
        "arm_ctl",
        "site_colon",
        "arm_ctl:site_colon",
    ]


# --- integrity ---------------------------------------------------------------------------


def test_sample_mapped_to_two_subjects_is_a_hard_error():
    table = pd.concat([_specimens(), _specimens().loc[["C1C"]].assign(subject="C2")])
    with pytest.raises(CellQuorumDataError, match="more than one subject"):
        _analyze(table)


def test_duplicate_specimen_rows_are_rejected():
    table = pd.concat([_specimens(), _specimens().loc[["C1C"]]])
    with pytest.raises(CellQuorumDataError, match="Duplicate specimen rows"):
        _analyze(table)


def test_missing_subject_identifier_is_a_hard_error():
    table = _specimens()
    table["subject"] = table["subject"].astype(object)
    table.loc["P3J", "subject"] = None
    with pytest.raises(CellQuorumDataError, match="subject"):
        _analyze(table)


def test_missing_sample_identifier_is_a_hard_error():
    table = _specimens()
    table.index = [None, *table.index[1:]]
    with pytest.raises(CellQuorumDataError, match="missing specimen ids"):
        _analyze(table)


def test_missing_factor_value_and_absent_factor_column_are_hard_errors():
    table = _specimens()
    table["site"] = table["site"].astype(object)
    table.loc["P3J", "site"] = None
    with pytest.raises(CellQuorumDataError, match="site"):
        _analyze(table)
    with pytest.raises(CellQuorumConfigError, match="absent"):
        _analyze(_specimens().drop(columns="site"))


# --- random effects -----------------------------------------------------------------------


def test_random_effect_column_absent_is_a_hard_error():
    with pytest.raises(CellQuorumConfigError, match="litter"):
        _analyze(_specimens(), _design(random_effects=["litter"]))


def test_random_effect_with_a_single_level_is_a_hard_error():
    table = _specimens()
    table["cage"] = "one"
    with pytest.raises(CellQuorumConfigError, match="'cage' has 1 group"):
        _validate(table, _design(random_effects=["cage"]))


def test_random_effect_with_one_specimen_per_group_is_a_hard_error():
    table = _specimens()
    table["unit"] = [f"u{i}" for i in range(len(table))]
    with pytest.raises(CellQuorumConfigError, match="one specimen per group"):
        _validate(table, _design(random_effects=["unit"]))


def test_random_effect_on_the_specimen_id_itself_is_a_hard_error():
    with pytest.raises(CellQuorumConfigError, match="one specimen per group"):
        _validate(_specimens(), _design(random_effects=["specimen"]))


def test_within_subject_without_subject_random_effect_warns():
    rep = _validate(_specimens(), _design(random_effects=[]))
    assert any("not a random effect" in w for w in rep.warnings)


# --- fixed-effect estimability ---------------------------------------------------------------


def test_factor_perfectly_confounded_with_another_declared_factor_is_rejected():
    table = _specimens()
    table["batch"] = table["arm"].map({"ctl": "b1", "trt": "b2"})
    design = _design(
        factors={
            "arm": {"reference": "ctl", "scope": "between_subject"},
            "batch": {"reference": "b1"},
        },
        interactions=[],
    )
    with pytest.raises(CellQuorumConfigError) as exc:
        _validate(table, design)
    assert "rank-deficient" in str(exc.value) and "arm ~ batch" in str(exc.value)


def test_undeclared_metadata_column_is_never_a_model_factor():
    table = _specimens()
    table["batch"] = table["arm"].map({"ctl": "b1", "trt": "b2"})  # aliased, but undeclared
    rep = _validate(table)
    assert rep.full_rank and "batch" not in " ".join(rep.fixed_effect_columns)


def test_empty_factorial_cell_for_a_requested_interaction_fails_validation():
    table = _specimens()
    table = table.drop(index=[i for i in table.index if i.startswith("P") and i.endswith("J")])
    rep = _analyze(table)  # analyzer reports
    (inter,) = rep.interactions
    assert inter.empty_cells == [
        {"levels": {"arm": "trt", "site": "jej"}, "n_samples": 0, "n_subjects": 0}
    ]
    with pytest.raises(CellQuorumConfigError, match="Interaction arm:site is not estimable"):
        _validate(table)


def test_replication_is_described_never_gated():
    # One subject per arm: scientifically thin, structurally describable.
    table = _specimens()
    table = table[table["subject"].isin(["C1", "P2"])]
    rep = _validate(table)
    assert rep.n_subjects == 2
    assert any("single independent subject" in w for w in rep.warnings)


def test_single_level_factor_is_not_estimable():
    table = _specimens()
    table["arm"] = "ctl"
    with pytest.raises(CellQuorumConfigError, match="single observed level"):
        _validate(table, _design(interactions=[]))


# --- reference-aware coding in the shared design-matrix layer ------------------------------------


def test_build_design_matrix_default_coding_is_unchanged_and_reference_is_opt_in():
    meta = _specimens()
    default = build_design_matrix(meta, ["arm", "site"])
    assert list(default.columns) == ["Intercept", "arm_trt", "site_jej"]
    flipped = build_design_matrix(meta, ["arm", "site"], references={"arm": "trt"})
    assert list(flipped.columns) == ["Intercept", "arm_ctl", "site_jej"]
    with pytest.raises(CellQuorumConfigError, match="not observed"):
        build_design_matrix(meta, ["arm"], references={"arm": "nope"})
