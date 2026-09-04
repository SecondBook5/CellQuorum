"""Tests for the central cohort schema and its resolver/overlay."""

from __future__ import annotations

from pathlib import Path

import pytest
from _external_data import stub_config_env

from cellquorum.config.cohort import (
    CohortConfig,
    resolve_cohort_key,
    validate_cohort_against_obs,
)
from cellquorum.config.loader import load_config
from cellquorum.config.models import CellQuorumConfig


def test_cohort_defaults_are_empty_and_optional() -> None:
    """A default cohort declares nothing, so existing configs are unaffected."""

    cfg = CellQuorumConfig.model_validate({"project": {"name": "t"}})
    assert cfg.cohort.sample_key is None
    assert cfg.cohort.donor_key is None
    assert cfg.cohort.condition_key is None
    assert cfg.cohort.batch_key is None
    assert cfg.cohort.condition_levels == []
    assert cfg.cohort.focus.labels == []


def test_resolve_cohort_key_prefers_cohort_over_stage_value() -> None:
    """A set cohort key wins; an unset one falls back to the stage value."""

    cfg = CellQuorumConfig.model_validate(
        {"project": {"name": "t"}, "cohort": {"batch_key": "sequencing_run"}}
    )
    # Cohort set -> cohort wins.
    assert resolve_cohort_key(cfg, attr="batch_key", stage_value="batch") == "sequencing_run"
    # Cohort unset -> stage value used.
    assert resolve_cohort_key(cfg, attr="donor_key", stage_value="patient_id") == "patient_id"


def test_resolve_cohort_key_handles_missing_cohort() -> None:
    """A config without a cohort attribute falls back to the stage value."""

    assert resolve_cohort_key(object(), attr="batch_key", stage_value="batch") == "batch"
    assert resolve_cohort_key(None, attr="batch_key", stage_value="batch") == "batch"


def test_validate_cohort_against_obs_warns_on_missing_keys() -> None:
    """Declared-but-absent cohort keys produce warnings, not errors."""

    cohort = CohortConfig(sample_key="sample_id", donor_key="missing_donor")
    warnings = validate_cohort_against_obs(["sample_id", "condition"], cohort=cohort)
    assert any("missing_donor" in w for w in warnings)
    assert not any("sample_id" in w for w in warnings)


def test_cohort_overlay_reaches_dispatch_stage_config() -> None:
    """The MethodDispatchStage cohort overlay injects same-named structural keys."""

    from cellquorum.methods.stage_base import _apply_cohort_overlay

    cfg = CellQuorumConfig.model_validate(
        {"project": {"name": "t"}, "cohort": {"batch_key": "sequencing_run"}}
    )
    context = type("Ctx", (), {"config": cfg})()

    # A stage config that understands batch_key gets the cohort value overlaid.
    overlaid = _apply_cohort_overlay(context, {"batch_key": "batch", "method": "harmony"})
    assert overlaid["batch_key"] == "sequencing_run"
    assert overlaid["method"] == "harmony"

    # A key the stage does not declare is not injected.
    assert "donor_key" not in _apply_cohort_overlay(context, {"method": "harmony"})


def test_le_kc_config_still_validates(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing le_kc.yaml must still validate with the cohort field added."""

    le_kc_path = Path("configs/le_kc.yaml")
    if not le_kc_path.exists():
        return  # Config not present in this checkout; nothing to assert.

    # Load through load_config rather than yaml.safe_load + model_validate. The config
    # names its external inputs with ${oc.env:...} instead of hardcoding one machine's
    # paths, and only load_config resolves those interpolations — a raw safe_load leaves
    # the literal "${oc.env:...}" string, which then fails the .h5ad suffix validator for
    # a reason that has nothing to do with this test. Going through the real loader also
    # means this exercises the path a user actually takes.
    stub_config_env(monkeypatch, tmp_path)

    cfg = load_config(le_kc_path)
    assert cfg.project.name
