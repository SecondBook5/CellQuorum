from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.cell_cell_communication.nichenet_method import NicheNetMethod
from cellquorum.methods.base import MethodSkip


@pytest.fixture
def mock_context():
    class MockBackend:
        def __init__(self, has_pkg=True):
            self._has_pkg = has_pkg

        def _rscript_available(self) -> bool:
            # Rscript is present; the test varies R-package availability below.
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
            self.root = tmp_path
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
    X = sp.csr_matrix(rng.poisson(1.0, size=(8, 12)).astype(float))
    obs = pd.DataFrame(
        {
            "cell_type": (["LEC", "Fib"] * 4),
            "sample_id": ([f"s{i % 4}" for i in range(8)]),
            "condition": (["case", "ctrl"] * 4),
        },
        index=[f"c{i}" for i in range(8)],
    )
    var = pd.DataFrame(index=[f"G{i}" for i in range(12)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_nichenet_skips_without_sender_receiver(tmp_path, mock_context):
    adata = _toy_adata()
    config = {"cell_type_col": "cell_type"}  # no sender/receiver
    res = NicheNetMethod()._run(adata, config, mock_context(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "sender" in res.reason.lower() or "receiver" in res.reason.lower()


def test_nichenet_skips_without_de_csv(tmp_path, mock_context):
    adata = _toy_adata()
    config = {
        "cell_type_col": "cell_type",
        "nichenet_sender": "LEC",
        "nichenet_receiver": "Fib",
    }  # DE csv missing
    res = NicheNetMethod()._run(adata, config, mock_context(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "de" in res.reason.lower() or "geneset" in res.reason.lower()


def test_nichenet_skips_without_prior_models(tmp_path, mock_context):
    adata = _toy_adata()
    de = tmp_path / "de.csv"
    pd.DataFrame(
        {
            "gene": ["G1", "G2"],
            "logFC": [3.0, 1.0],
            "logCPM": [1, 1],
            "F": [1, 1],
            "PValue": [0.001, 0.001],
            "FDR": [0.01, 0.02],
        }
    ).to_csv(de, index=False)
    config = {
        "cell_type_col": "cell_type",
        "nichenet_sender": "LEC",
        "nichenet_receiver": "Fib",
        "nichenet_de_csv": str(de),
    }
    # prior models unset
    res = NicheNetMethod()._run(adata, config, mock_context(tmp_path))
    assert isinstance(res, MethodSkip)
    assert "prior" in res.reason.lower() or "model" in res.reason.lower()


def test_nichenet_skips_unknown_sender(tmp_path, mock_context):
    adata = _toy_adata()
    de = tmp_path / "de.csv"
    pd.DataFrame(
        {"gene": ["G1"], "logFC": [3.0], "logCPM": [1], "F": [1], "PValue": [0.001], "FDR": [0.01]}
    ).to_csv(de, index=False)
    config = {
        "cell_type_col": "cell_type",
        "nichenet_sender": "GHOST",
        "nichenet_receiver": "Fib",
        "nichenet_de_csv": str(de),
    }
    res = NicheNetMethod()._run(adata, config, mock_context(tmp_path))
    assert isinstance(res, MethodSkip)
