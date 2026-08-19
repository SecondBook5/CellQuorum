"""Regression: a subprocess timeout must skip the method, never crash the stage.

The subprocess-backed DA methods (milo, propeller via Rscript; sccoda via the
sccoda_env helper) pass a ``timeout`` to the backend, which calls
``subprocess.run(..., timeout=...)`` — that raises ``subprocess.TimeoutExpired``.
Before this fix the methods caught only ``FileNotFoundError``, so a configured
``timeout_seconds`` firing would propagate out of the method, crash
``DifferentialAbundanceStage.run``, and abort every sibling method still queued
after it — including the trusted ``proportion_ttest`` anchor. Each method must
instead record a MethodSkip (skip-not-crash invariant).
"""

from __future__ import annotations

import subprocess

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.differential_abundance.milo_method import MiloMethod
from cellquorum.differential_abundance.propeller_method import PropellerMethod
from cellquorum.differential_abundance.sccoda_method import SccodaMethod
from cellquorum.methods.base import MethodSkip


def _cohort_adata():
    """Small eligible cohort: 3 donors/arm, a rep for milo, a cell_type column."""
    rng = np.random.default_rng(0)
    n_genes = 20
    donors = ["d1", "d2", "d3", "d4", "d5", "d6"]
    blocks, obs_rows = [], []
    for i, donor in enumerate(donors):
        condition = "Normal" if i < 3 else "Disease"
        for _ in range(15):
            blocks.append(rng.poisson(5, size=n_genes).astype(float))
            obs_rows.append({"patient_id": donor, "condition": condition, "cell_type": "TypeA"})
        for _ in range(10):
            blocks.append(rng.poisson(5, size=n_genes).astype(float))
            obs_rows.append({"patient_id": donor, "condition": condition, "cell_type": "TypeB"})
    X = np.vstack(blocks)
    a = ad.AnnData(X=X, obs=pd.DataFrame(obs_rows))
    a.var_names = [f"G{i}" for i in range(n_genes)]
    # A reduced-dim rep so milo passes its rep guard and reaches the subprocess call.
    a.obsm["X_pca"] = np.asarray(X[:, :10])
    return a


class _TimingOutBackend:
    """A backend whose script/helper invocation always times out."""

    def _rscript_available(self) -> bool:
        # Rscript is present; the timeout occurs later in run_script/run_helper.
        return True

    def _r_package_available(self, pkg: str) -> bool:
        # Pretend the R package is present so the method reaches run_script.
        return True

    def run_script(self, script, args, timeout=None):
        raise subprocess.TimeoutExpired(cmd="Rscript", timeout=timeout or 1)

    def run_helper(self, script, args, timeout=None):
        raise subprocess.TimeoutExpired(cmd="micromamba", timeout=timeout or 1)

    def status(self):
        class _S:
            available = True

        return _S()


class _Registry:
    def __init__(self, backend):
        self._backend = backend

    def get(self, name):
        return self._backend


class _Paths:
    def __init__(self, tmp_path):
        self.results = tmp_path / "results"
        self.scratch = tmp_path / "scratch"
        self.results.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp_path):
        self.paths = _Paths(tmp_path)
        self.backend_registry = _Registry(_TimingOutBackend())


_BASE_CONFIG = {
    "condition_col": "condition",
    "donor_col": "patient_id",
    "cell_type_col": "cell_type",
    "case": "Disease",
    "control": "Normal",
    "timeout_seconds": 1,
}


def test_milo_skips_on_subprocess_timeout(tmp_path):
    """Milo records a skip (not a crash) when the R subprocess times out."""
    out = MiloMethod()._run(_cohort_adata(), dict(_BASE_CONFIG), _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)
    assert "timed out" in out.reason.lower()


def test_propeller_skips_on_subprocess_timeout(tmp_path):
    """Propeller records a skip (not a crash) when the R subprocess times out."""
    out = PropellerMethod()._run(_cohort_adata(), dict(_BASE_CONFIG), _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)
    assert "timed out" in out.reason.lower()


def test_sccoda_skips_on_subprocess_timeout(tmp_path):
    """scCODA records a skip (not a crash) when the env helper times out."""
    out = SccodaMethod()._run(_cohort_adata(), dict(_BASE_CONFIG), _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)
    assert "timed out" in out.reason.lower()
