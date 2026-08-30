"""The DA stage bridges config.design → the method's config keys.

Without this bridge DA methods never receive case/control (they live in
config.design, not in DifferentialAbundanceConfig), so DA always returns a
"case/control not set" skip in production. These tests pin the
declare-the-question-once bridge: cohort keys win, design fills the rest, and an
absent design still degrades to a clean skip (never a crash).

Additionally tests the default methods-list injection: a bare config (no methods,
no scalar method) receives the 4-method default list; explicit methods/method are
left untouched.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.stages.comparative.differential_abundance.stage import DifferentialAbundanceStage
from cellquorum.config.cohort import CohortConfig
from cellquorum.config.design import DesignConfig
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.contracts.layer_tags import set_layer_tag


def _adata():
    X = sp.csr_matrix(np.random.default_rng(0).poisson(5, size=(20, 10)).astype(float))
    obs = pd.DataFrame(
        {
            "patient_id": (["d1"] * 5 + ["d2"] * 5) * 2,
            "condition": ["Normal"] * 10 + ["LE"] * 10,
            "cell_type": ["TypeA"] * 10 + ["TypeB"] * 10,
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
        # Mock paths for methods that write artifacts
        root = Path("/tmp/da_test")
        self.paths = type("obj", (object,), {"root": root, "results": root / "results"})

    def require_adata(self):
        return self._adata


def test_augment_config_pulls_case_control_from_design():
    """A populated config.design supplies case/control/paired to the method config."""

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal", paired=True))
    stage = DifferentialAbundanceStage()

    augmented = stage._augment_config(_Ctx(config), {"method": "milo"})

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
    stage = DifferentialAbundanceStage()

    augmented = stage._augment_config(_Ctx(config), {"method": "milo"})

    # Cohort is the single source of truth for structural keys.
    assert augmented["donor_col"] == "subject"
    assert augmented["condition_col"] == "group"
    # case/control still come from design.
    assert augmented["case"] == "LE"
    assert augmented["control"] == "Normal"


def test_augment_config_stage_value_wins_over_design():
    """An explicit stage-config value is never overwritten by the design block."""

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal"))
    stage = DifferentialAbundanceStage()

    augmented = stage._augment_config(
        _Ctx(config),
        {"method": "milo", "case": "OVERRIDE", "condition_col": "cond_override"},
    )

    assert augmented["case"] == "OVERRIDE"
    assert augmented["condition_col"] == "cond_override"
    # Unset keys still fall back to design.
    assert augmented["control"] == "Normal"


def test_augment_config_absent_design_leaves_case_control_unset():
    """With no case/control declared anywhere, the keys stay unset (clean skip, no crash)."""

    config = CellQuorumConfig(design=DesignConfig())  # case/control default to None
    stage = DifferentialAbundanceStage()

    augmented = stage._augment_config(_Ctx(config), {"method": "milo"})

    assert not augmented.get("case")
    assert not augmented.get("control")


def test_augment_config_injects_default_methods_list_when_bare():
    """A bare config (no methods, no scalar method) receives the 4-method default."""

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal"))
    stage = DifferentialAbundanceStage()

    augmented = stage._augment_config(_Ctx(config), {})

    assert "methods" in augmented
    methods = augmented["methods"]
    assert len(methods) == 4
    method_names = [m["method"] for m in methods]
    assert method_names == ["milo", "sccoda", "propeller", "proportion_ttest"]


def test_augment_config_respects_explicit_methods_list():
    """An explicit methods list is never overwritten."""

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal"))
    stage = DifferentialAbundanceStage()

    explicit = [{"method": "propeller"}, {"method": "proportion_ttest"}]
    augmented = stage._augment_config(_Ctx(config), {"methods": explicit})

    assert augmented["methods"] == explicit


def test_augment_config_respects_scalar_method():
    """An explicit scalar method key suppresses default methods-list injection."""

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal"))
    stage = DifferentialAbundanceStage()

    augmented = stage._augment_config(_Ctx(config), {"method": "sccoda"})

    # No methods list should be injected.
    assert "methods" not in augmented
    assert augmented["method"] == "sccoda"


def test_augment_config_injects_methods_even_when_design_is_none():
    """The methods-list injection happens even when design is absent."""

    config = CellQuorumConfig()  # No design block
    stage = DifferentialAbundanceStage()

    augmented = stage._augment_config(_Ctx(config), {})

    assert "methods" in augmented
    assert len(augmented["methods"]) == 4


def test_stage_runs_instead_of_skipping_when_design_declares_case_control():
    """End-to-end through run(): a populated design makes DA attempt the fit.

    The stage will dispatch to all 4 methods. In the test environment without
    backends, some methods may skip (e.g. milo/sccoda/propeller for missing
    backends), but proportion_ttest is pure-python and should run or skip for
    a data reason — NOT for "case/control not set". Reaching any downstream
    skip/success proves the design bridge delivered case/control.
    """

    config = CellQuorumConfig(design=DesignConfig(case="LE", control="Normal"))
    result = DifferentialAbundanceStage().run(_Ctx(config))

    # The stage should not skip entirely (methods list is non-empty).
    # Check that per_method results exist.
    metrics = result.metrics
    assert "per_method" in metrics

    # Verify that no method skipped due to "case/control not set".
    for method_result in metrics["per_method"]:
        if method_result.get("skipped"):
            reason = method_result.get("reason", "")
            assert "case/control labels not set" not in reason
