"""A cohort key that is declared but absent from the data must be visible to the user.

A misspelled condition column does not stop the run (stages that need it validate it
themselves), so the only protection is that it is reported where the user will look: the
run summary and the report. It used to land only in a provenance JSON file, while the report
said "No warnings were emitted."
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from cellquorum import run_pipeline
from cellquorum.backends.base import BaseBackend
from cellquorum.backends.registry import BackendRegistry
from cellquorum.config.models import StageSelectionConfig


def _registry() -> BackendRegistry:
    backends = BackendRegistry()
    backends.register(BaseBackend(name="python", kind="python"))
    return backends


def _config(source: Path, condition_key: str) -> dict:
    return {
        "project": {"name": "cohort_mismatch"},
        "input": {"h5ad": str(source), "counts_layer": "counts"},
        "compute": {"backend": "cpu", "prefer_gpu": False},
        "r": {"enabled": False},
        "run": {"random_seed": 1, "verbose": False},
        "cohort": {"sample_key": "sample", "donor_key": "donor", "condition_key": condition_key},
        "stages": {name: name == "qc_evidence" for name in StageSelectionConfig.model_fields},
        "qc_evidence": {
            "metrics": {"layer": "counts", "percent_top": [20]},
            "floors": {"min_genes_per_cell": 20, "min_cells_per_gene": 1},
            "doublets": {"enabled": False},
            "graded": {"archetype_audit": False},
            "outputs": {"write_figures": False},
        },
        "report": {"enabled": True, "html": True, "markdown": True, "fail_on_report_error": True},
    }


def _write_input(path: Path) -> None:
    rng = np.random.default_rng(3)
    counts = rng.poisson(3, size=(60, 40)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "sample": np.repeat(["s0", "s1", "s2"], 20),
            "donor": np.repeat(["d1", "d2", "d3"], 20),
            "condition": np.tile(["control", "treated"], 30),
        },
        index=[f"cell{i}" for i in range(60)],
    )
    data = ad.AnnData(
        sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(40)])
    )
    data.layers["counts"] = data.X.copy()
    data.write_h5ad(path)


def test_a_misspelled_cohort_key_reaches_the_summary_and_the_report(tmp_path: Path) -> None:
    source = tmp_path / "input.h5ad"
    _write_input(source)
    result = run_pipeline(
        _config(source, condition_key="treatment_typo"),
        output_dir=tmp_path / "out",
        backend_registry=_registry(),
        quiet=True,
    )
    assert any("treatment_typo" in warning for warning in result.plan.warnings)
    report = (tmp_path / "out" / "reports" / "report.md").read_text()
    assert "treatment_typo" in report
    assert "No warnings were emitted" not in report


def test_a_correct_cohort_key_adds_no_warning(tmp_path: Path) -> None:
    source = tmp_path / "input.h5ad"
    _write_input(source)
    result = run_pipeline(
        _config(source, condition_key="condition"),
        output_dir=tmp_path / "out",
        backend_registry=_registry(),
        quiet=True,
    )
    assert not any("Cohort" in warning for warning in result.plan.warnings)
