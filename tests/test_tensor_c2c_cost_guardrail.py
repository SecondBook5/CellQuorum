"""Tensor-cell2cell decomposition cost guardrail (#140).

The non-negative CP factorization cost scales with ``runs x prod(tensor.shape)``;
the tensor's sender/receiver axes are the cell-type group count, so a fine-grained
(many-subcluster) tensor at the default ``robust`` setting (100 runs) can silently
take many hours. These tests cover the pure run-count decision that turns that
silent footgun into either an explicit budget-scaled run count (``auto``) or a loud
over-budget warning, and the config surface that admits the new knobs.
"""

from __future__ import annotations

import pytest

from cellquorum.cell_cell_communication.config import CellCellCommunicationConfig
from cellquorum.cell_cell_communication.tensor_c2c_method import (
    resolve_factorization_runs,
)
from cellquorum.core.exceptions import CellQuorumConfigError

# --------------------------------------------------------------------------- #
# No budget set -> byte-identical to the pre-guardrail behavior                #
# --------------------------------------------------------------------------- #


def test_robust_without_budget_is_100_runs():
    runs, note = resolve_factorization_runs(
        tf_optimization="robust", tensor_elements=1_000_000, max_cost=None
    )
    assert runs == 100
    assert note is None


def test_regular_without_budget_is_1_run():
    runs, note = resolve_factorization_runs(
        tf_optimization="regular", tensor_elements=1_000_000, max_cost=None
    )
    assert runs == 1
    assert note is None


def test_unknown_tensor_size_falls_back_to_base_runs():
    # An unbuilt tensor exposes shape () -> tensor_elements is None. Without a
    # size estimate the guardrail must not block: keep the requested run count.
    runs, note = resolve_factorization_runs(
        tf_optimization="robust", tensor_elements=None, max_cost=10
    )
    assert runs == 100
    assert note is None


# --------------------------------------------------------------------------- #
# 'auto' -> scale runs to fit the cost budget                                  #
# --------------------------------------------------------------------------- #


def test_auto_scales_runs_down_to_budget():
    # budget // elements = 10000 // 1000 = 10 runs (< 100), so it scales and notes.
    runs, note = resolve_factorization_runs(
        tf_optimization="auto", tensor_elements=1000, max_cost=10_000
    )
    assert runs == 10
    assert note is not None
    assert "auto-scaled" in note


def test_auto_never_drops_below_one_run():
    # A tensor larger than the whole budget still gets one run (never zero).
    runs, note = resolve_factorization_runs(
        tf_optimization="auto", tensor_elements=1000, max_cost=500
    )
    assert runs == 1
    assert note is not None


def test_auto_uses_full_robust_runs_when_budget_is_generous():
    # A generous budget leaves auto at the robust ceiling and emits no scaling note.
    runs, note = resolve_factorization_runs(
        tf_optimization="auto", tensor_elements=10, max_cost=10_000_000
    )
    assert runs == 100
    assert note is None


# --------------------------------------------------------------------------- #
# Explicit robust/regular over budget -> honor the choice but warn loudly      #
# --------------------------------------------------------------------------- #


def test_explicit_robust_over_budget_warns_but_keeps_runs():
    runs, note = resolve_factorization_runs(
        tf_optimization="robust", tensor_elements=1_000_000, max_cost=1000
    )
    assert runs == 100  # explicit choice honored, not silently reduced
    assert note is not None
    assert "exceeds" in note
    assert "auto" in note  # points the user at the auto escape hatch


def test_explicit_robust_under_budget_is_quiet():
    runs, note = resolve_factorization_runs(
        tf_optimization="robust", tensor_elements=10, max_cost=10_000
    )
    assert runs == 100
    assert note is None


# --------------------------------------------------------------------------- #
# Fail-loud on an unknown optimization level                                   #
# --------------------------------------------------------------------------- #


def test_unknown_tf_optimization_raises():
    with pytest.raises(CellQuorumConfigError, match="tf_optimization"):
        resolve_factorization_runs(
            tf_optimization="turbo", tensor_elements=100, max_cost=None
        )


# --------------------------------------------------------------------------- #
# Config surface                                                               #
# --------------------------------------------------------------------------- #


def test_config_accepts_auto_and_budget():
    cfg = CellCellCommunicationConfig(
        tf_optimization="auto", max_decomposition_cost=5_000_000
    )
    assert cfg.tf_optimization == "auto"
    assert cfg.max_decomposition_cost == 5_000_000


def test_config_budget_defaults_to_none():
    cfg = CellCellCommunicationConfig()
    assert cfg.max_decomposition_cost is None
    assert cfg.tf_optimization == "robust"


def test_config_rejects_unknown_tf_optimization():
    with pytest.raises(ValueError):
        CellCellCommunicationConfig(tf_optimization="bogus")
