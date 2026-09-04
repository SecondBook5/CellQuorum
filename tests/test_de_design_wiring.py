"""The DE stage bridges config.design → the method's config keys.

Without this bridge the pseudobulk_edger method never receives case/control
(they live in config.design, not in DifferentialExpressionConfig), so DE always
returns a "case/control not set" skip in production. These tests pin the
declare-the-question-once bridge: cohort keys win, design fills the rest, and an
absent design still degrades to a clean skip (never a crash).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.config.cohort import CohortConfig
from cellquorum.config.design import DesignConfig
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.stages.comparative.differential_expression.stage import DifferentialExpressionStage


def _adata():
    X = sp.csr_matrix(np.random.default_rng(0).poisson(5, size=(20, 10)).astype(float))
    obs = pd.DataFrame(
        {
            "patient_id": (["d1"] * 5 + ["d2"] * 5) * 2,
            "condition": ["Normal"] * 10 + ["LE"] * 10,
        }
    )
    a = ad.AnnData(X=X, obs=obs)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(10)]
    set_layer_tag(a, "counts", kind="counts")
    return a


class _Ctx:
    """Minimal pipeline context wrapping a real (or fake) config object."""

    def __init__(self, config):
        self._adata = _adata()
        self.config = config
        self.backend_registry = None

    def require_adata(self):
        return self._adata


def test_augment_config_pulls_case_control_from_design():
    """A populated config.design supplies case/control/paired to the method config."""

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal", paired=True))
    stage = DifferentialExpressionStage()

    augmented = stage._augment_config(_Ctx(config), {"method": "pseudobulk_edger"})

    assert augmented["case"] == "LE"
    assert augmented["control"] == "Normal"
    assert augmented["paired"] is True
    # Design column defaults flow through so the method's contract lines up with obs.
    assert augmented["condition_col"] == "condition"
    assert augmented["donor_col"] == "patient_id"


def test_augment_config_cohort_keys_win_over_design():
    """Cohort donor/condition keys take precedence over the design defaults."""

    config = CellQuorumConfig(
        cohort=CohortConfig(donor_key="subject", condition_key="group"),
        design=DesignConfig(
            donor_col="patient_id",
            condition_col="condition",
            case="LE",
            control="Normal",
        ),
    )
    stage = DifferentialExpressionStage()

    augmented = stage._augment_config(_Ctx(config), {"method": "pseudobulk_edger"})

    # Cohort is the single source of truth for structural keys.
    assert augmented["donor_col"] == "subject"
    assert augmented["condition_col"] == "group"
    # case/control still come from design.
    assert augmented["case"] == "LE"
    assert augmented["control"] == "Normal"


def test_augment_config_stage_value_wins_over_design():
    """An explicit stage-config value is never overwritten by the design block."""

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal"))
    stage = DifferentialExpressionStage()

    augmented = stage._augment_config(
        _Ctx(config),
        {"method": "pseudobulk_edger", "case": "OVERRIDE", "condition_col": "cond_override"},
    )

    assert augmented["case"] == "OVERRIDE"
    assert augmented["condition_col"] == "cond_override"
    # Unset keys still fall back to design.
    assert augmented["control"] == "Normal"


def test_augment_config_absent_design_leaves_case_control_unset():
    """With no case/control declared anywhere, the keys stay unset (clean skip, no crash)."""

    config = CellQuorumConfig(design=DesignConfig())  # case/control default to None
    stage = DifferentialExpressionStage()

    augmented = stage._augment_config(_Ctx(config), {"method": "pseudobulk_edger"})

    assert not augmented.get("case")
    assert not augmented.get("control")


def test_stage_runs_instead_of_skipping_when_design_declares_case_control():
    """End-to-end through run(): a populated design makes DE attempt the fit.

    Backend is absent here, so the method reaches a *downstream* skip (rscript
    backend unavailable) rather than the upstream "case/control not set" skip.
    Reaching the backend guard proves the design bridge delivered case/control.
    """

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal"))
    result = DifferentialExpressionStage().run(_Ctx(config))

    assert result.metrics.get("skipped") is True
    reason = result.metrics.get("reason", "")
    assert "case/control labels not set" not in reason
