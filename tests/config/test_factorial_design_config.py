"""Schema-level tests for the factorial / repeated-measures design declaration.

These need no data: they pin what ``FactorialDesignConfig`` accepts and rejects at
config time, that legacy ``DesignConfig`` YAML is untouched, and the single
precedence rule resolving the specimen and subject columns.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cellquorum.config.design import DesignConfig, FactorialDesignConfig, FactorSpec
from cellquorum.config.factorial_design import (
    required_specimen_columns,
    resolve_factorial_columns,
)
from cellquorum.core.exceptions import CellQuorumConfigError


def _factorial(**overrides) -> dict:
    base = {
        "subject_col": "subject",
        "factors": {
            "arm": {"reference": "ctl", "scope": "between_subject"},
            "site": {"reference": "a", "scope": "within_subject"},
        },
        "interactions": [["arm", "site"]],
        "random_effects": ["subject"],
    }
    base.update(overrides)
    return base


# --- legacy compatibility ----------------------------------------------------------


def test_legacy_design_config_is_unchanged():
    design = DesignConfig.model_validate(
        {"donor_col": "d", "condition_col": "c", "case": "x", "control": "y", "paired": True}
    )
    assert design.factorial is None
    assert (design.donor_col, design.condition_col, design.case, design.control) == (
        "d",
        "c",
        "x",
        "y",
    )
    assert design.paired is True
    assert DesignConfig().factorial is None


def test_legacy_design_still_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        DesignConfig.model_validate({"donor_col": "d", "bogus": 1})


# --- valid declaration ---------------------------------------------------------------


def test_factorial_block_parses_from_plain_mapping():
    design = DesignConfig.model_validate({"factorial": _factorial()})
    fac = design.factorial
    assert fac is not None
    assert fac.factors["arm"] == FactorSpec(reference="ctl", scope="between_subject")
    assert fac.interactions == [("arm", "site")]
    assert fac.random_effects == ["subject"]


def test_factor_scope_defaults_to_sample_level():
    fac = FactorialDesignConfig.model_validate({"factors": {"batch": {"reference": "b1"}}})
    assert fac.factors["batch"].scope == "sample_level"


def test_subject_and_random_effect_are_independent_concepts():
    # A subject column with no random effect, and a random effect that is not the subject.
    fac = FactorialDesignConfig.model_validate(
        _factorial(random_effects=["litter"], interactions=[])
    )
    assert fac.subject_col == "subject"
    assert fac.random_effects == ["litter"]


# --- rejections ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"factors": {}}, "at least one factor"),
        ({"interactions": [["arm", "nope"]]}, "not a declared factor"),
        ({"interactions": [["arm", "arm"]]}, "with itself"),
        ({"interactions": [["arm", "site"], ["arm", "site"]]}, "declared twice"),
        ({"interactions": [["arm", "site"], ["site", "arm"]]}, "declared twice"),
        ({"random_effects": ["subject", "subject"]}, "duplicates"),
        ({"random_effects": [""]}, "non-empty"),
        ({"sample_col": "subject"}, "distinct"),
    ],
)
def test_structurally_invalid_declarations_fail_early(overrides, match):
    with pytest.raises(ValidationError, match=match):
        FactorialDesignConfig.model_validate(_factorial(**overrides))


def test_empty_reference_unknown_scope_and_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        FactorSpec.model_validate({"reference": ""})
    with pytest.raises(ValidationError):
        FactorSpec.model_validate({"reference": "a", "scope": "unspecified"})
    with pytest.raises(ValidationError):
        FactorSpec.model_validate({"reference": "a", "levels": ["a"]})
    with pytest.raises(ValidationError):
        FactorialDesignConfig.model_validate(_factorial(unknown=1))


# --- column resolution ------------------------------------------------------------------


def _fac(**kw) -> FactorialDesignConfig:
    return FactorialDesignConfig.model_validate(_factorial(interactions=[], **kw))


def test_columns_fall_back_to_cohort_keys():
    config = SimpleNamespace(cohort=SimpleNamespace(sample_key="lib", donor_key="animal"))
    fac = FactorialDesignConfig.model_validate({"factors": {"arm": {"reference": "ctl"}}})
    resolved = resolve_factorial_columns(config, fac)
    assert (resolved.sample_col, resolved.subject_col) == ("lib", "animal")


def test_explicit_design_columns_beat_cohort_keys():
    config = SimpleNamespace(cohort=SimpleNamespace(sample_key="lib", donor_key="animal"))
    fac = _fac(sample_col="specimen")
    resolved = resolve_factorial_columns(config, fac)
    assert (resolved.sample_col, resolved.subject_col) == ("specimen", "subject")


def test_unresolvable_specimen_column_is_an_error():
    with pytest.raises(CellQuorumConfigError, match="specimen column"):
        resolve_factorial_columns(None, _fac())


def test_subject_may_remain_unresolved():
    fac = FactorialDesignConfig.model_validate({"factors": {"arm": {"reference": "ctl"}}})
    resolved = resolve_factorial_columns(
        SimpleNamespace(cohort=SimpleNamespace(sample_key="lib", donor_key=None)), fac
    )
    assert resolved.subject_col is None


def test_cohort_keys_that_coincide_are_rejected():
    config = SimpleNamespace(cohort=SimpleNamespace(sample_key="x", donor_key="x"))
    fac = FactorialDesignConfig.model_validate({"factors": {"arm": {"reference": "ctl"}}})
    with pytest.raises(CellQuorumConfigError, match="same column"):
        resolve_factorial_columns(config, fac)


def test_required_specimen_columns_lists_subject_factors_random_effects_once():
    fac = _fac(random_effects=["subject", "litter", "specimen"])
    cols = required_specimen_columns(fac, sample_col="specimen", subject_col="subject")
    assert cols == ["subject", "arm", "site", "litter"]
