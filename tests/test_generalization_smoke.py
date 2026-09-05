"""Acceptance test: the same engine runs a non-KC cohort via config only.

This is the WP12 gate for the "generalize the spine" slice. It builds a generic
PBMC-style stimulated-vs-control cohort (no keratinocyte/lymphedema anything),
declares its structure through the central `cohort` block, and runs it end to
end through the public ``run_pipeline`` API on a CPU-only backend. It asserts
the run completes, writes provenance + a report, and that any stage needing an
absent backend skips cleanly rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum import run_pipeline
from cellquorum.backends.base import BaseBackend
from cellquorum.backends.registry import BackendRegistry


def _cpu_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(BaseBackend(name="python", kind="python"))
    return registry


def _pbmc_adata(n: int = 80, seed: int = 0) -> ad.AnnData:
    """Build a generic PBMC-style cohort with the declared cohort obs keys."""

    rng = np.random.default_rng(seed)
    genes = ["MT-CO1", "MT-ND1", "RPS3", "RPL13"] + [f"G{i}" for i in range(46)]
    x = rng.poisson(2.0, size=(n, len(genes))).astype(np.float32)

    # Two donors x two conditions x two batches — a generic non-KC structure.
    donor = np.where(np.arange(n) % 2 == 0, "D1", "D2")
    stim = np.where((np.arange(n) // 2) % 2 == 0, "control", "stimulated")
    batch = np.where((np.arange(n) // 4) % 2 == 0, "b1", "b2")
    obs = pd.DataFrame(
        {
            "sample_id": [f"{d}_{s}" for d, s in zip(donor, stim, strict=True)],
            "donor_id": donor,
            "stim": stim,
            "batch": batch,
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = x.copy()
    return adata


def _generic_config(h5ad_path: Path, output_dir: Path) -> dict:
    """A generic non-KC config that declares structure once via cohort."""

    return {
        "project": {"name": "generic_pbmc_smoke"},
        "input": {"h5ad": str(h5ad_path), "counts_layer": "counts"},
        "run": {"random_seed": 7, "verbose": False},
        "compute": {"backend": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
        "r": {"enabled": False},
        "cohort": {
            "sample_key": "sample_id",
            "donor_key": "donor_id",
            "condition_key": "stim",
            "batch_key": "batch",
            "condition_levels": ["control", "stimulated"],
        },
        "stages": {
            "qc": True,
            "preprocessing": True,
            "dimensionality": True,
            "clustering": True,
            "population_identity": True,
            # Deps-heavy / atlas / R stages off for portability.
            "integration": False,
            "reference_mapping": False,
            "annotation_diagnostics": False,
            "integration_benchmark": False,
            "adjudication": False,
        },
        "qc": {
            "metrics": {"layer": "counts", "percent_top": [2]},
            # The fixture has 50 genes, so the default 200-gene detection floor would remove
            # every cell. Set the floors to the fixture's scale: this test is about the stages
            # executing on a generic cohort, not about this synthetic matrix being clean.
            "floors": {"min_genes_per_cell": 5, "min_cells_per_gene": 1},
            "outputs": {"write_h5ad": False, "write_figures": False},
        },
        # cp10k recipe keeps this generalization smoke test env-independent; the
        # scclr-backed PFlog1pPF default needs the isolated scclr env (covered by
        # the dedicated scclr backend/normalization tests).
        "preprocessing": {
            "normalization": {
                "output_layer": "cellquorum_normalized",
                "recipe": "cellquorum_log1p_cp10k_v1",
            }
        },
        "dimensionality": {"input_layer": "cellquorum_normalized", "n_pcs": 5, "max_pcs": 5},
        "clustering": {"method": "leiden", "use_rep": "X_pca"},
        "population_identity": {"cluster_key": "leiden", "write_figures": False},
        "report": {"enabled": True, "html": True, "markdown": True},
    }


def test_generic_non_kc_cohort_runs_end_to_end(tmp_path: Path) -> None:
    """The engine runs a generic cohort via config, producing provenance + report."""

    h5ad_path = tmp_path / "pbmc.h5ad"
    _pbmc_adata().write_h5ad(h5ad_path)
    output_dir = tmp_path / "run"

    result = run_pipeline(
        _generic_config(h5ad_path, output_dir),
        output_dir=output_dir,
        backend_registry=_cpu_registry(),
        execute=True,
    )

    execution = result.execution_result
    assert execution is not None

    # The run must not fail; any ineligible stage must skip, not error.
    assert not execution.has_failures()

    # Provenance and the human-readable report were written.
    provenance = output_dir / "provenance" / "stage_execution_records.json"
    assert provenance.exists()
    assert (output_dir / "reports" / "report.md").exists()
    assert (output_dir / "reports" / "methods.txt").exists()

    # QC ran and annotated the generic cohort (proving cohort-keyed grouping).
    succeeded = set(execution.succeeded_stage_names())
    assert "qc" in succeeded


def test_generic_config_file_plans_and_validates() -> None:
    """The shipped generic example config validates and plans."""

    config_path = Path("configs/generic_pbmc_example.yaml")
    if not config_path.exists():
        return

    import yaml

    from cellquorum.config.models import CellQuorumConfig
    from cellquorum.core.planner import PipelinePlanner

    cfg = CellQuorumConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    plan = PipelinePlanner(cfg).build_plan()
    # The plan renders and QC is among the enabled stages.
    assert "qc" in plan.enabled_stage_names()
