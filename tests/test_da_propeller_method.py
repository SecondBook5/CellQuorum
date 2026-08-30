# tests/test_da_propeller_method.py
import shutil
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.stages.comparative.differential_abundance.propeller_method import PropellerMethod
from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.methods.base import MethodSkip


def _propeller_available() -> bool:
    """Check if Rscript and speckle package are available."""
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status=ifelse(requireNamespace('speckle', quietly=TRUE),0,1))",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _cohort_adata():
    """Build a small cohort with ≥3 donors/arm, one cell type enriched in case."""
    rng = np.random.default_rng(42)
    n_genes = 20
    donors = ["d1", "d2", "d3", "d4", "d5", "d6"]
    blocks, obs_rows = [], []

    for i, donor in enumerate(donors):
        # First 3 donors are control, last 3 are case
        condition = "Normal" if i < 3 else "Disease"

        # TypeA: enriched in Disease (more cells in case)
        n_typeA = 5 if condition == "Normal" else 20
        for _ in range(n_typeA):
            counts = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(counts)
            obs_rows.append({"patient_id": donor, "condition": condition, "cell_type": "TypeA"})

        # TypeB: balanced
        for _ in range(10):
            counts = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(counts)
            obs_rows.append({"patient_id": donor, "condition": condition, "cell_type": "TypeB"})

    X = sp.csr_matrix(np.vstack(blocks))
    obs = pd.DataFrame(obs_rows)
    a = ad.AnnData(X=X, obs=obs)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(n_genes)]
    set_layer_tag(a, "counts", kind="counts")
    return a


class _Paths:
    def __init__(self, tmp_path):
        self.results = tmp_path / "results"
        self.scratch = tmp_path / "scratch"
        self.results.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)


class _Registry:
    def get(self, name):
        from cellquorum.backends.rscript import RscriptBackend

        if name == "rscript":
            return RscriptBackend()
        raise KeyError(name)


class _Ctx:
    def __init__(self, tmp_path):
        self.paths = _Paths(tmp_path)
        self.backend_registry = _Registry()


def test_contract_requires_design_obs():
    """Verify the method's input contract requires design obs columns."""
    method = PropellerMethod()
    contract = method.input_contract(
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            "case": "Disease",
            "control": "Normal",
        }
    )
    assert "condition" in contract.required_obs
    assert "patient_id" in contract.required_obs
    assert "cell_type" in contract.required_obs


def test_skips_when_rscript_absent(monkeypatch, tmp_path):
    """Verify the method skips when Rscript is unavailable."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    method = PropellerMethod()
    out = method._run(
        _cohort_adata(),
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            "case": "Disease",
            "control": "Normal",
            "transform": "asin",
        },
        _Ctx(tmp_path),
    )
    assert isinstance(out, MethodSkip)
    assert "rscript" in out.reason.lower() or "propeller" in out.reason.lower()


def test_skips_when_case_control_absent(tmp_path):
    """Verify the method skips when case/control labels are not set."""
    method = PropellerMethod()
    out = method._run(
        _cohort_adata(),
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            # case and control missing
        },
        _Ctx(tmp_path),
    )
    assert isinstance(out, MethodSkip)
    assert "case" in out.reason.lower() or "control" in out.reason.lower()


def test_skips_when_cell_type_col_absent(tmp_path):
    """Verify the method skips (not crashes) when cell_type column is missing.

    An ineligible stage must record a skip, not raise. This test pins the
    requires_obs() guard behavior so cohorts missing cell-type annotations
    skip cleanly at the executor level.
    """
    # Build an AnnData WITHOUT cell_type column.
    rng = np.random.default_rng(42)
    X = sp.csr_matrix(rng.poisson(5, size=(50, 20)).astype(float))
    obs = pd.DataFrame(
        {
            "patient_id": ["d1"] * 25 + ["d2"] * 25,
            "condition": ["Normal"] * 25 + ["Disease"] * 25,
        }
    )
    a = ad.AnnData(X=X, obs=obs)
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(20)]
    set_layer_tag(a, "counts", kind="counts")

    # Call the real .run() entrypoint (NOT ._run) so requires_obs is exercised.
    method = PropellerMethod()
    result = method.run(
        a,
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            "case": "Disease",
            "control": "Normal",
        },
        _Ctx(tmp_path),
    )

    # Must return a MethodSkip whose reason mentions the missing obs column.
    assert isinstance(result, MethodSkip)
    assert "cell_type" in result.reason or "obs" in result.reason


@pytest.mark.skipif(not _propeller_available(), reason="Rscript+speckle not available")
def test_propeller_detects_enriched_celltype(tmp_path):
    """Verify propeller detects a cell type enriched in case."""
    method = PropellerMethod()
    result = method._run(
        _cohort_adata(),
        {
            "condition_col": "condition",
            "donor_col": "patient_id",
            "cell_type_col": "cell_type",
            "case": "Disease",
            "control": "Normal",
            "transform": "asin",
        },
        _Ctx(tmp_path),
    )
    # A StageResult with a da_results artifact.
    artifact_paths = [a.path for a in result.artifacts if a.name == "da_results"]
    assert artifact_paths, "expected a da_results artifact"
    da = pd.read_csv(artifact_paths[0])

    # Verify the CSV has expected columns.
    assert "cell_type" in da.columns
    assert "PropRatio" in da.columns
    assert "PValue" in da.columns
    assert "FDR" in da.columns

    # TypeA should be detected as enriched (low p-value).
    typeA = da.loc[da["cell_type"] == "TypeA"].iloc[0]
    assert typeA["PValue"] < 0.05, "TypeA should be significantly enriched"

    # Check metrics.
    assert result.metrics["case"] == "Disease"
    assert result.metrics["control"] == "Normal"
    assert result.metrics["transform"] == "asin"
    assert result.metrics["n_samples"] == 6  # 6 donors
    assert result.metrics["n_celltypes"] == 2  # TypeA, TypeB
