from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.cell_cell_communication.nichenet_method import NicheNetMethod


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


# --- multi-sender resolution -------------------------------------------------------------
#
# "which cell types signal to this one" is the question NicheNet is usually asked, and it has
# no single-sender answer. These tests pin the three ways a caller names senders and the one
# thing that must never reach a downstream network: a joined "A, B" label in a source column.


def _multi_adata():
    """Four cell types, so an ``all`` sentinel has something to exclude and something to keep."""
    rng = np.random.default_rng(1)
    X = sp.csr_matrix(rng.poisson(1.0, size=(12, 12)).astype(float))
    obs = pd.DataFrame(
        {"cell_type": (["LEC", "Fib", "Mac", "T"] * 3)},
        index=[f"c{i}" for i in range(12)],
    )
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(12)]))


def _de_csv(tmp_path):
    de = tmp_path / "de.csv"
    pd.DataFrame(
        {
            "gene": [f"G{i}" for i in range(6)],
            "logFC": [3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
            "logCPM": [1] * 6,
            "F": [1] * 6,
            "PValue": [0.001] * 6,
            "FDR": [0.01] * 6,
        }
    ).to_csv(de, index=False)
    return de


def _priors(tmp_path):
    paths = {}
    for key, fname in (
        ("nichenet_ligand_target_matrix", "lt.rds"),
        ("nichenet_lr_network", "lr.rds"),
        ("nichenet_weighted_networks", "wn.rds"),
    ):
        p = tmp_path / fname
        p.write_bytes(b"0")
        paths[key] = str(p)
    return paths


class _CapturingContext:
    """A context whose R backend records argv and writes the outputs R would write."""

    def __init__(self, tmp_path, mock_context, *, expression=None):
        self.inner = mock_context(tmp_path)
        self.paths = self.inner.paths
        self.args = None
        self._expression = expression
        outer = self

        class Backend:
            def _rscript_available(self):
                return True

            def _r_package_available(self, package):
                return True

            def run_script(self, script, args, timeout=None):
                outer.args = list(args)
                pd.DataFrame(
                    {"test_ligand": ["TGFB1", "IL1B"], "aupr_corrected": [0.3, 0.2], "rank": [1, 2]}
                ).to_csv(args[6], index=False)
                pd.DataFrame(
                    {
                        "ligand": ["TGFB1", "IL1B"],
                        "receptor": ["TGFBR1", "IL1R1"],
                        "aupr_corrected": [0.3, 0.2],
                    }
                ).to_csv(args[7], index=False)
                if outer._expression is not None and len(args) >= 20:
                    outer._expression.to_csv(args[19], index=False)

                class Proc:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Proc()

        class Registry:
            def get(self, name):
                if name == "rscript":
                    return Backend()
                raise ValueError(name)

        self.backend_registry = Registry()


def _config(tmp_path, sender):
    config = {
        "cell_type_col": "cell_type",
        "nichenet_sender": sender,
        "nichenet_receiver": "LEC",
        "nichenet_de_csv": str(_de_csv(tmp_path)),
    }
    config.update(_priors(tmp_path))
    return config


def test_all_sentinel_sends_every_other_cell_type(tmp_path, mock_context):
    ctx = _CapturingContext(tmp_path, mock_context)
    res = NicheNetMethod()._run(_multi_adata(), _config(tmp_path, "all"), ctx)
    assert not isinstance(res, MethodSkip), getattr(res, "reason", "")
    # argv position 10 (index 9) is the sender list; the receiver must not be in it.
    assert ctx.args[9] == "Fib,Mac,T"
    assert res.metrics["n_senders"] == 3


def test_a_sender_list_is_passed_comma_joined(tmp_path, mock_context):
    ctx = _CapturingContext(tmp_path, mock_context)
    res = NicheNetMethod()._run(_multi_adata(), _config(tmp_path, ["Fib", "Mac"]), ctx)
    assert not isinstance(res, MethodSkip), getattr(res, "reason", "")
    assert ctx.args[9] == "Fib,Mac"


def test_one_unknown_sender_in_a_list_is_refused_and_named(tmp_path, mock_context):
    ctx = _CapturingContext(tmp_path, mock_context)
    res = NicheNetMethod()._run(_multi_adata(), _config(tmp_path, ["Fib", "GHOST"]), ctx)
    assert isinstance(res, MethodSkip)
    # Naming the offender matters: "sender absent" alone leaves the caller guessing which.
    assert "GHOST" in str(res.details)


def test_all_sentinel_refused_when_the_receiver_is_the_only_cell_type(tmp_path, mock_context):
    adata = _multi_adata()
    adata.obs["cell_type"] = "LEC"
    ctx = _CapturingContext(tmp_path, mock_context)
    res = NicheNetMethod()._run(adata, _config(tmp_path, "all"), ctx)
    assert isinstance(res, MethodSkip)


def test_the_two_new_output_paths_are_passed_and_registered(tmp_path, mock_context):
    expression = pd.DataFrame(
        {
            "sender": ["Fib", "Mac", "Fib", "Mac"],
            "ligand": ["TGFB1", "TGFB1", "IL1B", "IL1B"],
            "n_cells": [3, 3, 3, 3],
            "fraction_expressing": [0.9, 0.5, 0.0, 0.8],
            "mean_expression": [1.2, 0.6, 0.0, 0.9],
            "expressed": ["TRUE", "TRUE", "FALSE", "TRUE"],
        }
    )
    ctx = _CapturingContext(tmp_path, mock_context, expression=expression)
    res = NicheNetMethod()._run(_multi_adata(), _config(tmp_path, ["Fib", "Mac"]), ctx)
    assert not isinstance(res, MethodSkip), getattr(res, "reason", "")
    assert len(ctx.args) == 20
    assert ctx.args[18].endswith("nichenet_ligand_target_weights.csv")
    assert ctx.args[19].endswith("nichenet_sender_expression.csv")
    names = {a.name for a in res.artifacts}
    assert "nichenet_sender_expression" in names
    # The weights file was not written by the mock, so it must not be claimed as an artifact.
    assert "nichenet_ligand_target_weights" not in names


def test_the_canonical_table_names_senders_not_a_joined_label(tmp_path, mock_context):
    expression = pd.DataFrame(
        {
            "sender": ["Fib", "Mac", "Fib", "Mac"],
            "ligand": ["TGFB1", "TGFB1", "IL1B", "IL1B"],
            "fraction_expressing": [0.9, 0.5, 0.0, 0.8],
            "expressed": ["TRUE", "TRUE", "FALSE", "TRUE"],
        }
    )
    ctx = _CapturingContext(tmp_path, mock_context, expression=expression)
    res = NicheNetMethod()._run(_multi_adata(), _config(tmp_path, ["Fib", "Mac"]), ctx)
    canonical = pd.read_csv(tmp_path / "results" / "nichenet_canonical_lr.csv")
    assert set(canonical["source"]) == {"Fib", "Mac"}
    assert "Fib, Mac" not in set(canonical["source"])
    # TGFB1 from both senders, IL1B from Mac only: three edges, not the two ligands.
    assert len(canonical) == 3
    assert set(canonical.loc[canonical["ligand"] == "IL1B", "source"]) == {"Mac"}
    assert res.metrics["n_canonical_edges"] == 3


def test_a_single_sender_still_labels_itself(tmp_path, mock_context):
    ctx = _CapturingContext(tmp_path, mock_context)
    res = NicheNetMethod()._run(_multi_adata(), _config(tmp_path, "Fib"), ctx)
    assert not isinstance(res, MethodSkip), getattr(res, "reason", "")
    canonical = pd.read_csv(tmp_path / "results" / "nichenet_canonical_lr.csv")
    assert set(canonical["source"]) == {"Fib"}
