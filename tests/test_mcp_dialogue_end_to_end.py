"""End-to-end integration test for multicellular_programs stage via DIALOGUE.

This test proves the full method + diagnostics + figure pipeline runs against the
real installed DIALOGUE R package, using a synthetic fixture with a shared latent
signal across cell types (the minimum structure that converges to programs).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.stage import StageResult
from cellquorum.multicellular_programs.dialogue_method import MulticellularProgramsMethod


def _dialogue_available() -> bool:
    """Check if Rscript and DIALOGUE package are available."""
    if shutil.which("Rscript") is None:
        return False
    p = subprocess.run(
        [
            "Rscript",
            "-e",
            'quit(status = ifelse(requireNamespace("DIALOGUE", quietly=TRUE), 0, 1))',
        ],
        capture_output=True,
        text=True,
    )
    return p.returncode == 0


@pytest.fixture
def mock_context():
    """Mock execution context with paths and backend registry."""

    class MockBackend:
        """Mock Rscript backend that shells to real Rscript."""

        def _rscript_available(self) -> bool:
            """Check if Rscript is present."""
            return shutil.which("Rscript") is not None

        def _r_package_available(self, package: str) -> bool:
            """Check if an R package is available."""
            if package == "DIALOGUE":
                return _dialogue_available()
            return False

        def run_script(self, script: Path, args: list[str], timeout: int):
            """Run Rscript — actually shells to the real binary."""
            result = subprocess.run(
                ["Rscript", str(script), *args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result

    class MockRegistry:
        def get(self, backend_name: str):
            if backend_name == "rscript":
                return MockBackend()
            raise ValueError(f"Unknown backend: {backend_name}")

    class MockPaths:
        def __init__(self, tmp_path: Path):
            self.scratch = tmp_path / "scratch"
            self.results = tmp_path / "results"
            self.scratch.mkdir(parents=True, exist_ok=True)
            self.results.mkdir(parents=True, exist_ok=True)

    class MockContext:
        def __init__(self, tmp_path: Path):
            self.paths = MockPaths(tmp_path)
            self.backend_registry = MockRegistry()

    return MockContext


def _build_dialogue_fixture():
    """Build AnnData with a shared latent signal across cell types.

    This is the verified converging structure: 8 samples, 2 cell types, 240 cells/type,
    SHARED latent identical across both cell types, X_pca carrying 3.0*latent[samp_idx],
    and 100 genes with two 20-gene latent-loaded blocks. Smaller/unstructured fixtures
    return "No programs" because DIALOGUE's ANOVA filter requires per-sample structure
    that is shared across cell types.
    """
    n_samples = 8
    n_cells_per_type = 240  # ~30 cells/sample/type, 2 cell types
    n_genes = 100
    n_pcs = 8

    # Shared latent (8 samples x 8 dims) — the signal that makes programs multicellular
    latent = np.random.default_rng(0).normal(size=(n_samples, n_pcs))

    # Generate cells for each cell type using the SAME latent
    obs_rows = []
    X_blocks = []
    X_pca_blocks = []

    for type_idx, cell_type in enumerate(["TypeA", "TypeB"]):
        # Use distinct noise seeds per cell type but share the latent
        rng = np.random.default_rng(1 + type_idx)

        # Build per-type sample indices (one index per cell)
        samp_idx = np.array([i % n_samples for i in range(n_cells_per_type)])

        # X_pca: noise + 3.0 * latent[sample_idx] for each cell
        pcs = rng.normal(size=(n_cells_per_type, n_pcs)) + 3.0 * latent[samp_idx, :]
        X_pca_blocks.append(pcs)

        # X expression: baseline counts + two latent-loaded gene blocks (factors 0, 1)
        base = rng.poisson(3, size=(n_genes, n_cells_per_type)).astype(float)
        load1 = np.zeros(n_genes)
        load1[:20] = 3.0
        load2 = np.zeros(n_genes)
        load2[20:40] = 3.0
        # signal is (n_genes, n_cells): outer(load, latent_per_cell)
        signal = np.outer(load1, latent[samp_idx, 0]) + np.outer(load2, latent[samp_idx, 1])
        expr = base + np.maximum(np.round(signal), 0.0)  # genes x cells

        # Transpose to cells x genes for AnnData
        X_blocks.append(expr.T)

        # Build obs rows
        for cell_i in range(n_cells_per_type):
            sample_name = f"s{samp_idx[cell_i]}"
            obs_rows.append(
                {
                    "cell_type": cell_type,
                    "sample": sample_name,
                    "donor": sample_name,
                    "condition": "A" if samp_idx[cell_i] % 2 == 0 else "B",
                }
            )

    # Assemble AnnData
    obs = pd.DataFrame(obs_rows)
    X = np.vstack(X_blocks)  # Stack both cell types
    X_pca = np.vstack(X_pca_blocks)

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
    )
    adata.obsm["X_pca"] = X_pca

    return adata


@pytest.mark.skipif(not _dialogue_available(), reason="DIALOGUE R package not available")
def test_dialogue_end_to_end(tmp_path, mock_context):
    """Run MulticellularProgramsMethod._run against real DIALOGUE with converging fixture."""

    adata = _build_dialogue_fixture()
    ctx = mock_context(tmp_path)

    config = {
        "cell_type_col": "cell_type",
        "sample_col": "sample",
        "donor_col": "donor",
        "condition_col": "condition",
        "use_rep": "X_pca",
        "n_pcs": 8,
        "n_programs": 2,
        "n_program_genes": 30,
        "min_cells_per_type": 20,
        "min_cell_types": 2,
        "min_samples": 6,
        "stability_resamples": 2,
        "seed": 1,
    }

    method = MulticellularProgramsMethod()
    result = method._run(adata, config, ctx)

    # Assert: result is a StageResult (not a MethodSkip)
    assert isinstance(result, StageResult), f"Expected StageResult, got {type(result)}"
    assert result.status == "success"

    # Assert: artifacts contain the canonical outputs
    artifact_names = {a.name for a in result.artifacts}
    assert "mcp_gene_programs" in artifact_names
    assert "mcp_scores" in artifact_names

    # Assert: gene programs table has expected columns
    programs_artifact = next(a for a in result.artifacts if a.name == "mcp_gene_programs")
    programs_df = pd.read_csv(programs_artifact.path)
    expected_cols = {"program", "gene", "loading", "cell_type"}
    assert expected_cols.issubset(
        programs_df.columns
    ), f"Missing columns: {expected_cols - set(programs_df.columns)}"
    assert not programs_df.empty, "Expected at least one program from converging fixture"

    # Assert: scores table has expected columns
    scores_artifact = next(a for a in result.artifacts if a.name == "mcp_scores")
    scores_df = pd.read_csv(scores_artifact.path)
    expected_cols = {"program", "sample", "cell_type", "score"}
    assert expected_cols.issubset(
        scores_df.columns
    ), f"Missing columns: {expected_cols - set(scores_df.columns)}"

    # Assert: donor support CSV exists
    donor_support_path = ctx.paths.results / "multicellular_programs" / "program_donor_support.csv"
    assert donor_support_path.exists(), "program_donor_support.csv missing"

    # Assert: stability CSV exists (resamples=2)
    stability_path = ctx.paths.results / "multicellular_programs" / "program_stability.csv"
    assert stability_path.exists(), "program_stability.csv missing"

    # Assert: metrics carry expected keys
    expected_metrics = {"n_programs", "n_cell_types_used", "n_samples"}
    assert expected_metrics.issubset(
        result.metrics.keys()
    ), f"Missing metrics: {expected_metrics - result.metrics.keys()}"
    assert result.metrics["n_programs"] > 0, "Expected at least one program"
