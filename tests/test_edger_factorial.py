"""Factorial / interaction differential-expression tests.

These cover the DE engine's factorial capability: building the edgeR design
right-hand side for a crossed design, and testing a two-way *interaction*
(difference-of-differences) rather than the case-vs-control main effect. The
interaction path reuses the multi-factor estimability layer in
``config.design`` so a non-estimable factorial (an empty grid cell) halts loudly
before the fit.
"""

from __future__ import annotations

import shutil
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.backends.registry import build_default_backend_registry
from cellquorum.comparative.differential_expression.pseudobulk_edger_method import (
    build_edger_design_rhs,
)
from cellquorum.comparative.differential_expression.stage import DifferentialExpressionStage
from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.core.exceptions import CellQuorumConfigError


def _edger_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status=ifelse(requireNamespace('edgeR', quietly=TRUE),0,1))",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


# --------------------------------------------------------------------------- #
# Pure design-RHS builder                                                     #
# --------------------------------------------------------------------------- #


def test_design_rhs_without_interactions_tests_condition():
    # No interactions -> additive RHS and the default (empty) test-coef token,
    # which the R script resolves to the case-vs-control condition coefficient.
    rhs, test_coef = build_edger_design_rhs(
        covariates=["batch"],
        paired=True,
        interactions=[],
        condition_col="condition",
        donor_col="patient_id",
    )
    assert rhs == "batch + donor + condition"
    assert test_coef == ""


def test_design_rhs_with_interaction_aliases_condition_and_flags_interaction():
    # The R meta CSV renames the condition column to the fixed name 'condition';
    # an interaction referencing the condition column must use that alias, and
    # the tested effect becomes the interaction (':interaction' token).
    rhs, test_coef = build_edger_design_rhs(
        covariates=["batch"],
        paired=False,
        interactions=[("status", "batch")],
        condition_col="status",
        donor_col="patient_id",
    )
    assert rhs == "batch + condition + condition:batch"
    assert test_coef == ":interaction"


# --------------------------------------------------------------------------- #
# End-to-end interaction test (requires edgeR)                                #
# --------------------------------------------------------------------------- #


def _factorial_adata(*, empty_le_b1: bool = False) -> ad.AnnData:
    """A crossed condition x batch design with distinct donors per arm.

    Distinct donors per condition arm keep the design unpaired (no donor-pairing
    auto-promotion), so ``batch`` stays crossed with ``condition`` at the donor
    level. Gene G2 carries a real interaction (the LE effect appears only in
    batch b1); gene G5 carries a pure main effect (the same LE shift in both
    batches, i.e. zero interaction).
    """

    rng = np.random.default_rng(7)
    spec = [
        ("d1", "Normal", "b0"),
        ("d2", "Normal", "b0"),
        ("d3", "Normal", "b1"),
        ("d4", "Normal", "b1"),
        ("d5", "LE", "b0"),
        ("d6", "LE", "b0"),
        # With empty_le_b1 the (LE, b1) grid cell is absent -> the interaction is
        # inestimable and the multi-factor gate must halt before the fit.
        ("d7", "LE", "b0" if empty_le_b1 else "b1"),
        ("d8", "LE", "b0" if empty_le_b1 else "b1"),
    ]
    blocks, obs_rows = [], []
    for donor, cond, batch in spec:
        for _ in range(10):
            base = rng.poisson(30, size=15).astype(float)
            if cond == "LE" and batch == "b1":
                base[2] += 200  # G2: interaction (LE effect only in b1)
            if cond == "LE":
                base[5] += 200  # G5: main effect (LE effect in both batches)
            blocks.append(base)
            obs_rows.append({"patient_id": donor, "condition": cond, "batch": batch})
    a = ad.AnnData(X=sp.csr_matrix(np.vstack(blocks)), obs=pd.DataFrame(obs_rows))
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(15)]
    set_layer_tag(a, "counts", kind="counts")
    return a


class _CfgInteraction:
    differential_expression = {
        "enabled": True,
        "method": "pseudobulk_edger",
        "layer": "counts",
        "condition_col": "condition",
        "donor_col": "patient_id",
        "case": "LE",
        "control": "Normal",
        "covariates": ["batch"],
        "interactions": [["condition", "batch"]],
        "paired": False,
    }
    cohort = None


@pytest.mark.skipif(not _edger_available(), reason="edgeR not installed")
def test_interaction_test_flags_interaction_gene_not_main_effect(tmp_path):
    # The interaction test must isolate the difference-of-differences: G2 (a real
    # condition x batch interaction) is significant, while G5 (a pure main effect
    # with zero interaction) is not, even though G5 has a large main-effect shift.
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    ctx = PipelineContext(
        config=_CfgInteraction(),
        paths=paths,
        adata=_factorial_adata(),
        backend_registry=build_default_backend_registry(),
    )
    result = DifferentialExpressionStage().run(ctx)

    de_paths = [a.path for a in result.artifacts if a.name == "de_results"]
    assert de_paths, result.metrics
    de = pd.read_csv(de_paths[0]).set_index("gene")
    assert de.loc["G2", "FDR"] < 0.05
    assert de.loc["G5", "FDR"] > 0.05


# --------------------------------------------------------------------------- #
# Fail-loud guards (no R required — gate runs before the backend)             #
# --------------------------------------------------------------------------- #


def test_interaction_halts_on_empty_factorial_cell(tmp_path):
    # (LE, b1) is entirely absent, so the interaction coefficient is a zero column
    # and the design is rank-deficient. The stage must halt loudly before edgeR.
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    ctx = PipelineContext(
        config=_CfgInteraction(),
        paths=paths,
        adata=_factorial_adata(empty_le_b1=True),
        backend_registry=build_default_backend_registry(),
    )
    with pytest.raises(CellQuorumConfigError, match="estimable|Empty|rank"):
        DifferentialExpressionStage().run(ctx)


class _CfgBadInteractionMember:
    differential_expression = {
        "enabled": True,
        "method": "pseudobulk_edger",
        "layer": "counts",
        "condition_col": "condition",
        "donor_col": "patient_id",
        "case": "LE",
        "control": "Normal",
        "covariates": ["batch"],
        # 'timepoint' is neither the condition column nor a declared covariate.
        "interactions": [["condition", "timepoint"]],
        "paired": False,
    }
    cohort = None


def test_interaction_member_must_be_condition_or_covariate(tmp_path):
    # An interaction term may only reference the condition column or a declared
    # covariate — otherwise the column would never be aggregated into pseudobulk.
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    ctx = PipelineContext(
        config=_CfgBadInteractionMember(),
        paths=paths,
        adata=_factorial_adata(),
        backend_registry=build_default_backend_registry(),
    )
    with pytest.raises(CellQuorumConfigError, match="interaction|covariate|condition"):
        DifferentialExpressionStage().run(ctx)
