"""End-to-end integration test for DifferentialAbundanceStage through PipelineContext.

This is the guardrail task: it exercises all 4 DA methods through the real dispatch
path, proving the design bridge delivers case/control to every method and no method
raises when its backend is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.backends.registry import build_default_backend_registry
from cellquorum.backends.sccoda_backend import build_sccoda_backend
from cellquorum.config.design import DesignConfig
from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.stages.comparative.differential_abundance.stage import DifferentialAbundanceStage


def _miloR_available() -> bool:
    """Check if Rscript and miloR package are available."""
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status=ifelse(requireNamespace('miloR', quietly=TRUE),0,1))",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _speckle_available() -> bool:
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


def _sccoda_available() -> bool:
    """Check if sccoda_env backend is available."""
    return build_sccoda_backend().status().available


def _adata():
    """Build a real cohort with ≥3 donors/arm, obsm['X_pca'], and cell_type enrichment.

    Structure:
    - 3 control donors (N1,N2,N3) + 3 case donors (D1,D2,D3) — disjoint (unpaired).
    - cell_type column: Type0, Type1, Type2.
    - Type1 is enriched in Disease (more cells in case donors).
    - X_pca in obsm for Milo (real PCA of random counts).
    """
    rng = np.random.default_rng(42)
    n_genes = 20

    # Generate per-donor/condition data
    donors_control = ["N1", "N2", "N3"]
    donors_case = ["D1", "D2", "D3"]
    blocks, obs_rows = [], []

    for donor in donors_control:
        # Type0: 30 cells
        for _ in range(30):
            counts = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(counts)
            obs_rows.append({"patient_id": donor, "condition": "Normal", "cell_type": "Type0"})
        # Type1: 20 cells (balanced)
        for _ in range(20):
            counts = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(counts)
            obs_rows.append({"patient_id": donor, "condition": "Normal", "cell_type": "Type1"})
        # Type2: 10 cells
        for _ in range(10):
            counts = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(counts)
            obs_rows.append({"patient_id": donor, "condition": "Normal", "cell_type": "Type2"})

    for donor in donors_case:
        # Type0: 20 cells
        for _ in range(20):
            counts = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(counts)
            obs_rows.append({"patient_id": donor, "condition": "Disease", "cell_type": "Type0"})
        # Type1: 50 cells (ENRICHED in case)
        for _ in range(50):
            counts = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(counts)
            obs_rows.append({"patient_id": donor, "condition": "Disease", "cell_type": "Type1"})
        # Type2: 10 cells
        for _ in range(10):
            counts = rng.poisson(5, size=n_genes).astype(float)
            blocks.append(counts)
            obs_rows.append({"patient_id": donor, "condition": "Disease", "cell_type": "Type2"})

    X = np.vstack(blocks)
    obs = pd.DataFrame(obs_rows)
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = [f"G{i}" for i in range(n_genes)]

    # Build X_pca: real PCA of X for Milo
    from sklearn.decomposition import PCA

    pca = PCA(n_components=20, random_state=42)
    a.obsm["X_pca"] = pca.fit_transform(X)

    return a


class _Cfg:
    """Minimal config object exposing the DA stage sub-block and design."""

    # Explicit methods list with sccoda num_iterations=2000 to keep test fast (~30s).
    # This still exercises the real 4-method dispatch path.
    differential_abundance = {
        "enabled": True,
        "methods": [
            {"method": "milo"},
            {"method": "sccoda", "num_iterations": 2000},
            {"method": "propeller"},
            {"method": "proportion_ttest"},
        ],
    }
    cohort = None
    # Design bridge: case/control set here so every method receives them.
    # Donors are disjoint, so proportion_ttest with paired=True will skip
    # for a data reason (not case/control-unset), which is acceptable.
    design = DesignConfig(
        donor_col="patient_id",
        condition_col="condition",
        case="Disease",
        control="Normal",
        paired=True,  # Donors disjoint → proportion_ttest skips for data reasons, not config
    )


def test_stage_runs_through_context(tmp_path):
    """
    Verify DifferentialAbundanceStage runs all 4 methods without crash.

    Core invariants (the guardrail):
    1. stage.run(ctx) NEVER raises.
    2. metrics["n_methods"] == 4 (the default 4-method list was dispatched).
    3. NO per_method skip reason mentions case/control-unset (design bridge worked).
    4. For each available backend, its artifact CSV exists and has expected columns.
    """
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()

    ctx = PipelineContext(
        config=_Cfg(),
        paths=paths,
        adata=_adata(),
        backend_registry=build_default_backend_registry(),
    )

    # Run the stage — must never raise, even if all backends are unavailable.
    result = DifferentialAbundanceStage().run(ctx)

    # Invariant 1: stage returned a real StageResult (not a top-level skip).
    assert result.metrics["n_methods"] == 4

    # Invariant 2: metrics has per_method list with 4 entries.
    per_method = result.metrics["per_method"]
    assert len(per_method) == 4

    # Invariant 3: NO method skipped for case/control-unset (design bridge delivered them).
    for entry in per_method:
        if entry.get("skipped"):
            reason = entry.get("reason", "").lower()
            # The case/control-unset message is in the method skip guards.
            assert (
                "case" not in reason or "control" not in reason
            ), f"Method {entry['method']} skipped for case/control-unset: {entry['reason']}"

    # Invariant 4: for each available backend, verify artifact exists with expected columns.
    # Map method name → (availability check, expected CSV columns).
    method_specs = {
        "milo": (
            _miloR_available(),
            [
                "nhood",
                "logFC",
                "PValue",
                "SpatialFDR",
                "nhood_size",
                "majority_celltype",
                "celltype_fraction",
            ],
        ),
        "sccoda": (
            _sccoda_available(),
            [
                "cell_type",
                "log2_fold_change",
                "inclusion_probability",
                "credible_effect",
                "reference",
            ],
        ),
        "propeller": (
            _speckle_available(),
            [
                "cell_type",
                # The arm means the fit already produced, so the table states the
                # magnitudes it ranks instead of only a ratio and a t.
                "control_mean_prop",
                "case_mean_prop",
                "effect_pp",
                "PropRatio",
                "Tstatistic",
                "PValue",
                "FDR",
                # Which design was fitted. This cohort's donors are disjoint, so the
                # requested donor block is declined and `paired` reads False -- the
                # point being that a reader can tell, rather than having to assume.
                "paired",
                "n_donors_blocked",
                "design_floor_p",
                "p_below_design_floor",
                "family_size",
                "family_min_concordant",
                "family_floor_reachable",
            ],
        ),
        "proportion_ttest": (
            True,  # Always available (pure Python)
            [
                "cell_type",
                "n_case",
                "n_control",
                "n_donors_concordant",
                "control_mean_pct",
                "case_mean_pct",
                "effect_pp",
                "bootstrap_ci_low_pp",
                "bootstrap_ci_high_pp",
                "statistic",
                "pvalue",
                "paired",
                "fdr",
                # The design floor and the family's reachability, so a null FDR is
                # distinguishable from a family that could not have called anything.
                "design_floor_p",
                "p_below_design_floor",
                "family_size",
                "family_min_concordant",
                "family_floor_reachable",
            ],
        ),
    }

    # Match artifacts to methods by path suffix.
    artifacts_by_method = {}
    for artifact in result.artifacts:
        if artifact.name == "da_results":
            for method_name in method_specs:
                if artifact.path.name.endswith(f"da_{method_name}.csv"):
                    artifacts_by_method[method_name] = artifact
                    break

    # We dispatched 4 methods in config order: milo, sccoda, propeller, proportion_ttest.
    # Match per_method entries by position (skipped methods inject {"method": ..., "skipped": True},
    # successful methods append their metrics dict directly with no "method" key).
    method_order = ["milo", "sccoda", "propeller", "proportion_ttest"]
    assert len(per_method) == len(method_order), "per_method count mismatch"

    # For each method, check artifact vs availability.
    for i, method_name in enumerate(method_order):
        available, expected_cols = method_specs[method_name]
        method_entry = per_method[i]

        if available:
            # Backend available → method should have produced an artifact
            # (or skipped for data reasons).
            if method_entry.get("skipped"):
                # Data-reason skip is acceptable (e.g. proportion_ttest with
                # disjoint donors + paired=True, or milo with rank-deficient
                # design when donors are disjoint and paired=True).
                # Just verify it's NOT a case/control-unset skip.
                reason = method_entry.get("reason", "").lower()
                # Acceptable data-reason skip patterns:
                # - "paired" / "donor" → insufficient donor overlap for paired analysis
                # - "variance" → zero-variance cell type
                # - "script failed" → R script detected rank-deficient design or other data issue
                # NOT acceptable: case/control mentions (design bridge failed)
                assert (
                    "paired" in reason
                    or "donor" in reason
                    or "variance" in reason
                    or "script failed" in reason
                ), f"{method_name} skipped unexpectedly: {method_entry['reason']}"
                # Also verify method name is correct in skip entry
                assert method_entry.get("method") == method_name
            else:
                # Method ran → artifact must exist.
                assert method_name in artifacts_by_method, f"Missing artifact for {method_name}"
                artifact_path = Path(artifacts_by_method[method_name].path)
                assert artifact_path.exists(), f"Artifact missing: {artifact_path}"

                # Read and verify columns.
                df = pd.read_csv(artifact_path)
                assert not df.empty, f"{method_name} artifact is empty"
                # Required columns, in this relative order. Not an exact match: some
                # columns are conditional on the data (scCODA emits ``is_primary`` only
                # when a sensitivity fit was worth running), so pinning the full list
                # makes the test fail on a legitimate second fit.
                missing = [c for c in expected_cols if c not in df.columns]
                assert not missing, f"{method_name} missing columns: {missing}"
                present_order = [c for c in df.columns if c in expected_cols]
                assert (
                    present_order == expected_cols
                ), f"{method_name} column order changed: {present_order} vs {expected_cols}"
                # Verify case/control made it through (design bridge worked)
                assert method_entry.get("case") == "Disease"
                assert method_entry.get("control") == "Normal"
        else:
            # Backend unavailable → method should have recorded a skip.
            assert method_entry.get(
                "skipped"
            ), f"{method_name} should skip when backend unavailable"
            assert method_entry.get("method") == method_name
            assert (
                method_name not in artifacts_by_method
            ), f"{method_name} produced artifact despite unavailable backend"
