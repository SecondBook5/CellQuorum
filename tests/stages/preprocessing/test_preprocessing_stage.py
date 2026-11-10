"""Tests for preprocessing stage implementation."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.core.stage import StageResult
from cellquorum.stages.preprocessing.config import NormalizationConfig, PreprocessingConfig
from cellquorum.stages.preprocessing.stage import (
    PreprocessingStage,
)


def _cp10k_config() -> CellQuorumConfig:
    """Config using a pure-Python recipe (no scclr env) for stage-plumbing tests.

    These tests exercise stage plumbing (StageResult shape, artifacts, metrics),
    not the transform, so they use the env-independent cp10k recipe rather than
    the scclr-backed PFlog1pPF default.
    """

    cfg = CellQuorumConfig()
    return cfg.model_copy(
        update={
            "preprocessing": PreprocessingConfig(
                normalization=NormalizationConfig(recipe="cellquorum_log1p_cp10k_v1")
            )
        }
    )


def make_test_adata() -> ad.AnnData:
    """Build a deterministic tiny AnnData for testing."""
    matrix = np.array(
        [
            [5, 5, 0, 0],
            [9, 0, 0, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    obs = pd.DataFrame(index=["cell_0", "cell_1", "cell_2"])
    var = pd.DataFrame(index=["MT-ND1", "ACTB", "RPS3", "MALAT1"])

    return ad.AnnData(X=matrix, obs=obs, var=var)


def make_context(
    tmp_path: Path,
    *,
    adata: ad.AnnData | None = None,
    config: object | None = None,
) -> PipelineContext:
    """
    Build a PipelineContext for preprocessing stage tests.

    Args:
        tmp_path: pytest temporary path.
        adata: Optional AnnData object.
        config: Optional runtime config.

    Returns:
        PipelineContext with initialized output paths.
    """

    # Build standardized pipeline paths.
    paths = PipelinePaths.from_output_dir(tmp_path / "run")

    # Create the path directories.
    paths.ensure_directories()

    # Return a PipelineContext.
    return PipelineContext(
        config=_cp10k_config() if config is None else config,
        paths=paths,
        adata=make_test_adata() if adata is None else adata,
        run_id="preprocessing-test-run",
        random_seed=123,
    )


def test_preprocessing_returns_counts_normalized_expression_and_artifacts(tmp_path):
    adata = make_test_adata()
    original_counts = adata.X.copy()
    result = PreprocessingStage().run(make_context(tmp_path, adata=adata))

    assert isinstance(result, StageResult)
    assert isinstance(result.adata, ad.AnnData)
    np.testing.assert_array_equal(result.adata.layers["counts"], original_counts)
    assert "cellquorum_normalized" in result.adata.layers
    assert result.metrics["stage_name"] == "preprocessing"
    assert result.metrics["enabled"] is True
    assert "recipe" in result.metrics
    assert result.metrics["output_layer"] == "cellquorum_normalized"
    assert (result.metrics["n_cells"], result.metrics["n_genes"]) == adata.shape
    assert any("Preprocessing completed" in note for note in result.notes)
    summary = next(a for a in result.artifacts if "summary" in a.name)
    assert summary.path.is_file()
    figures = [a for a in result.artifacts if a.kind == "figure"]
    assert len(figures) == 4
    assert all(a.path.is_file() for a in figures)


def test_preprocessing_stage_disabled_by_config(tmp_path):
    """Test that preprocessing is disabled when config.enabled is False."""
    # Build config with preprocessing disabled.
    config = CellQuorumConfig(preprocessing=PreprocessingConfig(enabled=False))
    context = make_context(tmp_path, config=config)

    # Run preprocessing stage.
    stage = PreprocessingStage()
    result = stage.run(context)

    # Should return disabled result.
    assert result.metrics.get("enabled") is False
    assert len(result.artifacts) == 0


def test_preprocessing_stage_disabled_by_stages_flag(tmp_path):
    """Test that preprocessing is disabled when stages.preprocessing is False."""
    # Build config with stages.preprocessing disabled.
    from cellquorum.config.models import StageSelectionConfig

    config = CellQuorumConfig(stages=StageSelectionConfig(preprocessing=False))
    context = make_context(tmp_path, config=config)

    # Run preprocessing stage.
    stage = PreprocessingStage()
    result = stage.run(context)

    # Should return disabled result.
    assert result.metrics.get("enabled") is False
    assert len(result.artifacts) == 0


def test_preprocessing_stage_disabled_normalization_includes_shape_metrics(tmp_path):
    """Test that disabled normalization includes n_cells and n_genes in metrics."""
    # Build config with normalization disabled but preprocessing enabled.
    from cellquorum.stages.preprocessing.config import NormalizationConfig

    config = CellQuorumConfig(
        preprocessing=PreprocessingConfig(
            enabled=True,
            normalization=NormalizationConfig(enabled=False),
        )
    )
    input_adata = make_test_adata()
    context = make_context(tmp_path, config=config, adata=input_adata)

    # Run preprocessing stage.
    stage = PreprocessingStage()
    result = stage.run(context)

    # Should return disabled normalization result with shape metrics.
    assert result.metrics.get("enabled") is True
    assert result.metrics.get("normalization_enabled") is False
    assert result.metrics.get("n_cells") == input_adata.n_obs
    assert result.metrics.get("n_genes") == input_adata.n_vars
    assert len(result.artifacts) == 0


def test_preprocessing_stage_skips_figures_when_disabled(tmp_path):
    config = _cp10k_config()
    config = config.model_copy(
        update={"preprocessing": config.preprocessing.model_copy(update={"write_figures": False})}
    )

    result = PreprocessingStage().run(make_context(tmp_path, config=config))

    assert not any(a.kind == "figure" for a in result.artifacts)


def test_cpu_normalization_does_not_require_rapids_in_gpu_workflow(tmp_path, monkeypatch):
    config = _cp10k_config()
    config.compute.backend = "gpu"
    config.compute.prefer_gpu = True
    config.compute.fallback_to_cpu = False

    def unexpected_probe():
        raise AssertionError("CPU normalization must not probe RAPIDS")

    monkeypatch.setattr("cellquorum.backends.compute.gpu_compute_available", unexpected_probe)
    result = PreprocessingStage().run(make_context(tmp_path, config=config))
    assert result.metrics["compute"] == "cpu"
    assert "cellquorum_normalized" in result.adata.layers
