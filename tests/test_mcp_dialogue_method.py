"""Precondition-skip tests for MulticellularProgramsMethod (skip-not-crash)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.comparative.multicellular_programs.dialogue_method import (
    MulticellularProgramsMethod,
)
from cellquorum.methods.base import MethodSkip


@pytest.fixture
def mock_context():
    class MockBackend:
        def __init__(self, has_pkg=True):
            self._has_pkg = has_pkg

        def _rscript_available(self) -> bool:
            return True

        def _r_package_available(self, package: str) -> bool:
            return self._has_pkg

    class MockRegistry:
        def __init__(self, backend):
            self._b = backend

        def get(self, name):
            if name == "rscript":
                return self._b
            raise ValueError(name)

    class MockPaths:
        def __init__(self, tmp_path):
            self.scratch = tmp_path / "scratch"
            self.results = tmp_path / "results"
            self.scratch.mkdir(parents=True, exist_ok=True)
            self.results.mkdir(parents=True, exist_ok=True)

    class MockContext:
        def __init__(self, tmp_path, has_pkg=True):
            self.paths = MockPaths(tmp_path)
            self.backend_registry = MockRegistry(MockBackend(has_pkg))

    return MockContext


def _adata(n=300, n_types=2, n_samples=6, with_rep=True):
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(n, 40)).astype(float)
    obs = pd.DataFrame(
        {
            "cell_type": [f"T{i % n_types}" for i in range(n)],
            "sample_id": [f"s{i % n_samples}" for i in range(n)],
            "condition": ["case" if (i % n_samples) % 2 else "ctrl" for i in range(n)],
            "donor": [f"d{i % n_samples}" for i in range(n)],
        },
        index=[f"c{i}" for i in range(n)],
    )
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = [f"G{i}" for i in range(40)]
    if with_rep:
        a.obsm["X_pca"] = rng.normal(size=(n, 8))
    return a


def _cfg(**over):
    cfg = {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "donor_col": "donor",
        "condition_col": "condition",
        "use_rep": "X_pca",
        "n_pcs": 5,
        "n_programs": 2,
        "min_cells_per_type": 20,
        "min_cell_types": 2,
        "min_samples": 4,
        "stability_resamples": 0,
    }
    cfg.update(over)
    return cfg


def test_skip_missing_rep(tmp_path, mock_context):
    res = MulticellularProgramsMethod()._run(_adata(with_rep=False), _cfg(), mock_context(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "rep" in res.reason.lower()


def test_skip_too_few_cell_types(tmp_path, mock_context):
    res = MulticellularProgramsMethod()._run(
        _adata(n_types=1), _cfg(min_cell_types=2), mock_context(tmp_path)
    )
    assert isinstance(res, MethodSkip)
    assert "cell type" in res.reason.lower()


def test_skip_too_few_samples(tmp_path, mock_context):
    res = MulticellularProgramsMethod()._run(
        _adata(n_samples=2), _cfg(min_samples=4), mock_context(tmp_path)
    )
    assert isinstance(res, MethodSkip)
    assert "sample" in res.reason.lower()


def test_skip_when_package_unavailable(tmp_path, mock_context):
    res = MulticellularProgramsMethod()._run(
        _adata(), _cfg(), mock_context(tmp_path, has_pkg=False)
    )
    assert isinstance(res, MethodSkip)
    assert "dialogue" in res.reason.lower()


def test_skip_on_missing_confounder(tmp_path, mock_context):
    # Ruling R3: a confounder that is not an obs column must skip loudly, not be silently dropped.
    res = MulticellularProgramsMethod()._run(
        _adata(), _cfg(confounders=["not_a_col"]), mock_context(tmp_path)
    )
    assert isinstance(res, MethodSkip)
    assert "confounder" in res.reason.lower()


def test_skip_on_non_numeric_confounder(tmp_path, mock_context):
    # Ruling R3: a present-but-categorical confounder must skip loudly (covar needs numeric).
    res = MulticellularProgramsMethod()._run(
        _adata(), _cfg(confounders=["cell_type"]), mock_context(tmp_path)
    )
    assert isinstance(res, MethodSkip)
    assert "confounder" in res.reason.lower()


def test_skip_on_missing_condition_col(tmp_path, mock_context):
    # An absent condition_col would KeyError inside export_dialogue_inputs -> skip loudly.
    res = MulticellularProgramsMethod()._run(
        _adata(), _cfg(condition_col="NOT_A_COLUMN"), mock_context(tmp_path)
    )
    assert isinstance(res, MethodSkip)
    assert "condition_col" in res.reason


def test_skip_on_missing_quality_col(tmp_path, mock_context):
    # An absent quality_col would KeyError inside export_dialogue_inputs -> skip loudly.
    res = MulticellularProgramsMethod()._run(
        _adata(), _cfg(quality_col="NOT_A_QUAL_COL"), mock_context(tmp_path)
    )
    assert isinstance(res, MethodSkip)
    assert "quality_col" in res.reason
