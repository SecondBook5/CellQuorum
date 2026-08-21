# tests/test_de_edger_method.py
import shutil
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.comparative.differential_expression.pseudobulk_edger_method import (
    PseudobulkEdgeRMethod,
)
from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.methods.base import MethodSkip


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


def _paired_adata():
    rng = np.random.default_rng(1)
    n_genes = 20
    donors = ["d1", "d2", "d3"]
    blocks, obs_rows = [], []
    for donor in donors:
        for cond in ["Normal", "LE"]:
            for _ in range(10):  # 10 cells per donor x condition
                base = rng.poisson(5, size=n_genes).astype(float)
                if cond == "LE":
                    base[1] += 40  # G1 up in LE
                blocks.append(base)
                obs_rows.append({"patient_id": donor, "condition": cond})
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


def test_contract_requires_counts_and_design_cols():
    method = PseudobulkEdgeRMethod()
    contract = method.input_contract(
        {
            "layer": "counts",
            "condition_col": "condition",
            "donor_col": "patient_id",
            "case": "LE",
            "control": "Normal",
            "covariates": [],
        }
    )
    assert "counts" in contract.required_layers
    assert "condition" in contract.required_obs
    assert "patient_id" in contract.required_obs
    assert contract.expected_kind == "counts"


def test_skips_when_rscript_absent(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: None)
    method = PseudobulkEdgeRMethod()
    out = method._run(
        _paired_adata(),
        {
            "layer": "counts",
            "condition_col": "condition",
            "donor_col": "patient_id",
            "case": "LE",
            "control": "Normal",
            "covariates": [],
            "paired": True,
        },
        _Ctx(tmp_path),
    )
    assert isinstance(out, MethodSkip)
    assert "rscript" in out.reason.lower() or "edger" in out.reason.lower()


def test_skips_when_design_obs_absent(tmp_path):
    """Verify the method skips (not crashes) when design obs columns are missing.

    An ineligible stage must record a skip, not raise. This test pins the
    requires_obs() guard behavior so generic cohorts (no patient_id/condition)
    skip cleanly at the executor level.
    """
    # Build an AnnData with a counts layer but WITHOUT the design obs columns.
    rng = np.random.default_rng(42)
    X = sp.csr_matrix(rng.poisson(5, size=(50, 20)).astype(float))
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"cell_{i}" for i in range(50)]))
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(20)]
    set_layer_tag(a, "counts", kind="counts")

    # Call the real .run() entrypoint (NOT ._run) so requires_obs is exercised.
    method = PseudobulkEdgeRMethod()
    result = method.run(
        a,
        {
            "layer": "counts",
            "condition_col": "condition",
            "donor_col": "patient_id",
            "case": "LE",
            "control": "Normal",
            "covariates": [],
            "paired": True,
        },
        _Ctx(tmp_path),
    )

    # Must return a MethodSkip whose reason mentions the missing obs column(s).
    assert isinstance(result, MethodSkip)
    assert "condition" in result.reason or "patient_id" in result.reason or "obs" in result.reason


def _unreplicated_adata():
    """One donor per condition arm — a non-estimable comparison."""
    rng = np.random.default_rng(3)
    n_genes = 20
    blocks, obs_rows = [], []
    # d1 contributes only Normal; d2 contributes only LE -> 1 donor per arm.
    for donor, cond in [("d1", "Normal"), ("d2", "LE")]:
        for _ in range(10):
            base = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(base)
            obs_rows.append({"patient_id": donor, "condition": cond})
    a = ad.AnnData(X=sp.csr_matrix(np.vstack(blocks)), obs=pd.DataFrame(obs_rows))
    a.layers["counts"] = a.X.copy()
    a.var_names = [f"G{i}" for i in range(n_genes)]
    set_layer_tag(a, "counts", kind="counts")
    return a


def test_unreplicated_arm_halts_loudly(tmp_path):
    """A comparison with a single donor per arm must HALT, not silently run.

    Regression for the silent-wrong class where DE ran on an unreplicated arm
    and returned confident-but-meaningless statistics. The pre-flight design
    guard raises before any R invocation, so this needs no edgeR.
    """
    from cellquorum.core.exceptions import CellQuorumConfigError

    method = PseudobulkEdgeRMethod()
    with pytest.raises(CellQuorumConfigError, match="donor replication"):
        method._run(
            _unreplicated_adata(),
            {
                "layer": "counts",
                "condition_col": "condition",
                "donor_col": "patient_id",
                "case": "LE",
                "control": "Normal",
                "covariates": [],
                "paired": False,
            },
            _Ctx(tmp_path),
        )


def test_min_donors_per_arm_override_allows_pilot(tmp_path, monkeypatch):
    """An explicit min_donors_per_arm=1 lets a deliberate pilot past the gate.

    With the floor lowered the estimability gate no longer raises; the method
    proceeds to the downstream Rscript-availability guard (skip when R absent),
    proving the gate itself did not halt.
    """
    monkeypatch.setattr("shutil.which", lambda name: None)
    method = PseudobulkEdgeRMethod()
    out = method._run(
        _unreplicated_adata(),
        {
            "layer": "counts",
            "condition_col": "condition",
            "donor_col": "patient_id",
            "case": "LE",
            "control": "Normal",
            "covariates": [],
            "paired": False,
            "min_donors_per_arm": 1,
        },
        _Ctx(tmp_path),
    )
    # Past the gate -> reaches the Rscript guard (which we forced unavailable).
    assert isinstance(out, MethodSkip)
    assert "rscript" in out.reason.lower() or "edger" in out.reason.lower()


@pytest.mark.skipif(not _edger_available(), reason="Rscript+edgeR not available")
def test_paired_de_detects_upregulated_gene(tmp_path):
    method = PseudobulkEdgeRMethod()
    result = method._run(
        _paired_adata(),
        {
            "layer": "counts",
            "condition_col": "condition",
            "donor_col": "patient_id",
            "case": "LE",
            "control": "Normal",
            "covariates": [],
            "paired": True,
        },
        _Ctx(tmp_path),
    )
    # A StageResult with a de_results artifact.
    artifact_paths = [a.path for a in result.artifacts if a.name == "de_results"]
    assert artifact_paths, "expected a de_results artifact"
    de = pd.read_csv(artifact_paths[0])
    g1 = de.loc[de["gene"] == "G1"].iloc[0]
    assert g1["logFC"] > 1 and g1["FDR"] < 0.05
    assert result.metrics["design_rhs"] == "donor + condition"
    assert result.metrics["n_pseudosamples"] == 6


@pytest.mark.skipif(not _edger_available(), reason="Rscript+edgeR not available")
def test_auto_promote_matched_cohort_and_contrast_sign(tmp_path):
    """A matched cohort passed as unpaired auto-promotes; the contrast sign is correct.

    Regression for the unpaired-by-default silent-wrong: the fixture is fully
    matched (every donor has both arms), so leaving ``paired=False`` must
    auto-promote to a donor-blocked fit (``donor + condition``). G1 is spiked up
    in LE (the case arm), so a correct contrast reports logFC > 0 for it.
    """
    method = PseudobulkEdgeRMethod()
    result = method._run(
        _paired_adata(),
        {
            "layer": "counts",
            "condition_col": "condition",
            "donor_col": "patient_id",
            "case": "LE",
            "control": "Normal",
            "covariates": [],
            "paired": False,  # NOT declared paired -> must auto-promote.
        },
        _Ctx(tmp_path),
    )
    # Auto-promotion: donor-blocked design + recorded note + paired metric.
    assert result.metrics["paired"] is True
    assert result.metrics["design_rhs"] == "donor + condition"
    assert any("Auto-promoted" in note for note in result.notes)
    # Contrast sign: G1 is up in LE (case), so logFC must be positive.
    de = pd.read_csv([a.path for a in result.artifacts if a.name == "de_results"][0])
    g1 = de.loc[de["gene"] == "G1"].iloc[0]
    assert g1["logFC"] > 1 and g1["FDR"] < 0.05
