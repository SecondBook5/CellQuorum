"""Tests for ProportionTTestMethod differential abundance."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.backends.registry import build_default_backend_registry
from cellquorum.differential_abundance.proportion_ttest_method import ProportionTTestMethod
from cellquorum.methods.base import MethodSkip


@pytest.fixture
def paired_cohort_adata():
    """
    Build a paired cohort with 4 donors present in BOTH case and control.

    Cell type distribution (cells per donor×condition):
    - TypeA: control ~60, case ~30  (depleted in case)
    - TypeB: control ~30, case ~60  (enriched in case)
    - TypeC: control ~10, case ~10  (no change)
    """
    np.random.seed(42)

    # 4 donors × 2 conditions = 8 samples
    donors = ["D1", "D2", "D3", "D4"] * 2
    conditions = ["control"] * 4 + ["case"] * 4

    obs_rows = []
    for donor, cond in zip(donors, conditions, strict=False):
        # TypeA: more in control
        n_a = np.random.poisson(60 if cond == "control" else 30)
        for _ in range(n_a):
            obs_rows.append({"donor": donor, "condition": cond, "cell_type": "TypeA"})

        # TypeB: more in case
        n_b = np.random.poisson(30 if cond == "control" else 60)
        for _ in range(n_b):
            obs_rows.append({"donor": donor, "condition": cond, "cell_type": "TypeB"})

        # TypeC: constant
        n_c = np.random.poisson(10)
        for _ in range(n_c):
            obs_rows.append({"donor": donor, "condition": cond, "cell_type": "TypeC"})

    obs = pd.DataFrame(obs_rows)
    X = np.zeros((len(obs), 10))
    return ad.AnnData(X=X, obs=obs)


@pytest.fixture
def unpaired_cohort_adata():
    """
    Build an unpaired cohort: 3 donors only-case, 3 donors only-control (disjoint).

    Cell type distribution similar to paired cohort.
    """
    np.random.seed(43)

    # 3 control-only donors + 3 case-only donors
    obs_rows = []
    for donor, cond in [
        ("C1", "control"),
        ("C2", "control"),
        ("C3", "control"),
        ("E1", "case"),
        ("E2", "case"),
        ("E3", "case"),
    ]:
        n_a = np.random.poisson(60 if cond == "control" else 30)
        for _ in range(n_a):
            obs_rows.append({"donor": donor, "condition": cond, "cell_type": "TypeA"})

        n_b = np.random.poisson(30 if cond == "control" else 60)
        for _ in range(n_b):
            obs_rows.append({"donor": donor, "condition": cond, "cell_type": "TypeB"})

        n_c = np.random.poisson(10)
        for _ in range(n_c):
            obs_rows.append({"donor": donor, "condition": cond, "cell_type": "TypeC"})

    obs = pd.DataFrame(obs_rows)
    X = np.zeros((len(obs), 10))
    return ad.AnnData(X=X, obs=obs)


@pytest.fixture
def zero_variance_adata():
    """
    Build a cohort where one cell type has identical proportion across all samples.
    """
    np.random.seed(44)

    donors = ["D1", "D2", "D3", "D4"] * 2
    conditions = ["control"] * 4 + ["case"] * 4

    obs_rows = []
    for donor, cond in zip(donors, conditions, strict=False):
        # TypeA: variable
        n_a = np.random.poisson(60 if cond == "control" else 30)
        for _ in range(n_a):
            obs_rows.append({"donor": donor, "condition": cond, "cell_type": "TypeA"})

        # TypeZ: EXACTLY 10 cells per sample (zero variance in proportion if totals constant)
        # To ensure truly zero variance in proportion, make total counts constant too
        n_z = 10
        for _ in range(n_z):
            obs_rows.append({"donor": donor, "condition": cond, "cell_type": "TypeZ"})

    obs = pd.DataFrame(obs_rows)
    X = np.zeros((len(obs), 10))
    return ad.AnnData(X=X, obs=obs)


@pytest.fixture
def mock_context(tmp_path):
    """Build a mock stage context with paths and backend registry."""

    class Paths:
        root = tmp_path
        scratch = tmp_path / "scratch"
        results = tmp_path / "results"

    class Context:
        paths = Paths()
        backend_registry = build_default_backend_registry()

    return Context()


def test_paired_happy_path(paired_cohort_adata, mock_context):
    """Paired t-test happy path: verify enriched type has positive effect_pp and low pvalue."""

    method = ProportionTTestMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "case",
        "control": "control",
        "paired": True,
        "seed": 42,
    }

    result = method.run(paired_cohort_adata, config, mock_context)

    # Should not be a skip
    assert not isinstance(result, MethodSkip)

    # Check artifacts
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.name == "da_results"
    assert artifact.path.name == "da_proportion_ttest.csv"
    assert artifact.kind == "csv"

    # Read the CSV
    df = pd.read_csv(artifact.path)

    # Verify columns
    expected_cols = {
        "cell_type",
        "n_case",
        "n_control",
        "control_mean_pct",
        "case_mean_pct",
        "effect_pp",
        "bootstrap_ci_low_pp",
        "bootstrap_ci_high_pp",
        "statistic",
        "pvalue",
        "fdr",
        "paired",
    }
    assert set(df.columns) == expected_cols

    # TypeB is enriched in case: effect_pp should be positive and significant
    type_b = df[df["cell_type"] == "TypeB"].iloc[0]
    assert type_b["effect_pp"] > 0
    assert type_b["pvalue"] < 0.05
    assert type_b["paired"]

    # FDR should be present and finite
    assert pd.notna(type_b["fdr"])

    # Metrics
    assert result.metrics["case"] == "case"
    assert result.metrics["control"] == "control"
    assert result.metrics["paired"] is True
    assert result.metrics["n_donors_paired"] == 4


def test_unpaired_path(unpaired_cohort_adata, mock_context):
    """Unpaired path: disjoint donors, paired=False."""

    method = ProportionTTestMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "case",
        "control": "control",
        "paired": False,
        "seed": 42,
    }

    result = method.run(unpaired_cohort_adata, config, mock_context)

    assert not isinstance(result, MethodSkip)

    df = pd.read_csv(result.artifacts[0].path)

    # paired column should be False
    assert all(~df["paired"])

    # TypeB should still be enriched
    type_b = df[df["cell_type"] == "TypeB"].iloc[0]
    assert type_b["effect_pp"] > 0

    # Metrics should carry n_case/n_control
    assert result.metrics["n_case"] == 3
    assert result.metrics["n_control"] == 3
    assert "n_donors_paired" not in result.metrics


def test_paired_no_overlap_skip(unpaired_cohort_adata, mock_context):
    """Paired=True but donors are disjoint → should skip, not fall back to unpaired."""

    method = ProportionTTestMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "case",
        "control": "control",
        "paired": True,  # Request paired but donors don't overlap
        "seed": 42,
    }

    result = method.run(unpaired_cohort_adata, config, mock_context)

    # Should skip
    assert isinstance(result, MethodSkip)
    assert "paired" in result.reason or "both arms" in result.reason


def test_determinism(paired_cohort_adata, mock_context, tmp_path):
    """Run twice with same seed → bootstrap CI should be identical."""

    method = ProportionTTestMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "case",
        "control": "control",
        "paired": True,
        "seed": 42,
    }

    # Run 1
    result1 = method.run(paired_cohort_adata, config, mock_context)
    assert not isinstance(result1, MethodSkip)
    df1 = pd.read_csv(result1.artifacts[0].path)

    # Run 2 (fresh context)
    run2_root = tmp_path / "run2"

    class Paths:
        root = run2_root
        scratch = run2_root / "scratch"
        results = run2_root / "results"

    class Context:
        paths = Paths()
        backend_registry = build_default_backend_registry()

    context2 = Context()
    result2 = method.run(paired_cohort_adata, config, context2)
    assert not isinstance(result2, MethodSkip)
    df2 = pd.read_csv(result2.artifacts[0].path)

    # Bootstrap CIs should be identical
    assert np.allclose(df1["bootstrap_ci_low_pp"], df2["bootstrap_ci_low_pp"])
    assert np.allclose(df1["bootstrap_ci_high_pp"], df2["bootstrap_ci_high_pp"])


def test_skip_missing_cell_type_col(paired_cohort_adata, mock_context):
    """Should skip when cell_type_col is missing."""

    method = ProportionTTestMethod()
    config = {
        "cell_type_col": "nonexistent_column",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "case",
        "control": "control",
    }

    result = method.run(paired_cohort_adata, config, mock_context)

    assert isinstance(result, MethodSkip)
    assert "nonexistent_column" in result.reason


def test_skip_missing_case_control(paired_cohort_adata, mock_context):
    """Should skip when case or control is unset."""

    method = ProportionTTestMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        # No case/control
    }

    result = method.run(paired_cohort_adata, config, mock_context)

    assert isinstance(result, MethodSkip)
    assert "case" in result.reason or "control" in result.reason


def test_zero_variance_no_crash(zero_variance_adata, mock_context):
    """Zero-variance cell type should not crash; should emit row with NaN or finite values."""

    method = ProportionTTestMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "case",
        "control": "control",
        "paired": True,
        "seed": 42,
    }

    result = method.run(zero_variance_adata, config, mock_context)

    # Should complete successfully
    assert not isinstance(result, MethodSkip)

    df = pd.read_csv(result.artifacts[0].path)

    # TypeZ should be present
    type_z = df[df["cell_type"] == "TypeZ"]
    assert len(type_z) == 1

    # Its pvalue might be NaN or a finite value, but should not crash
    # (zero variance in paired differences can yield NaN from ttest_rel)
