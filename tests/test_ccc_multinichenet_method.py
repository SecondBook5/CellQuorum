from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.cell_cell_communication.multinichenet_method import MultiNicheNetMethod
from cellquorum.methods.base import MethodSkip


@pytest.fixture
def mock_context():
    class MockBackend:
        def __init__(self, has_pkg=True):
            self._has_pkg = has_pkg

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


def _toy_adata():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(20, 8)).astype(float))
    obs = pd.DataFrame(
        {
            "cell_type": (["A", "B"] * 10),
            "sample_id": ([f"s{i%4}" for i in range(20)]),
            "condition": (["case", "ctrl"] * 10),
            "patient_id": ([f"p{i%4}" for i in range(20)]),
        },
        index=[f"c{i}" for i in range(20)],
    )
    var = pd.DataFrame(index=[f"G{i}" for i in range(8)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_multinichenet_skips_without_contrast(tmp_path, mock_context):
    adata = _toy_adata()
    config = {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "condition_col": "condition",
    }  # no case/control
    res = MultiNicheNetMethod()._run(adata, config, mock_context(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "contrast" in res.reason.lower()


def test_multinichenet_skips_without_prior_models(tmp_path, mock_context):
    adata = _toy_adata()
    config = {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "condition_col": "condition",
        "case": "case",
        "control": "ctrl",
    }
    # prior model paths unset
    res = MultiNicheNetMethod()._run(adata, config, mock_context(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "prior" in res.reason.lower() or "model" in res.reason.lower()


def test_multinichenet_skips_without_r_package(tmp_path, mock_context, monkeypatch):
    monkeypatch.setattr(
        "cellquorum.cell_cell_communication.multinichenet_method.shutil.which",
        lambda name: "/usr/bin/Rscript",
    )
    adata = _toy_adata()
    lt = tmp_path / "lt.rds"
    lt.write_text("x")
    lr = tmp_path / "lr.rds"
    lr.write_text("x")
    config = {
        "cell_type_col": "cell_type",
        "sample_col": "sample_id",
        "condition_col": "condition",
        "case": "case",
        "control": "ctrl",
        "nichenet_ligand_target_matrix": str(lt),
        "nichenet_lr_network": str(lr),
    }
    res = MultiNicheNetMethod()._run(adata, config, mock_context(tmp_path, has_pkg=False))
    assert isinstance(res, MethodSkip)
    assert "package" in res.reason.lower() or "multinichenetr" in res.reason.lower()
