"""Tests for configured AnnData input loading in pipeline context construction."""

from __future__ import annotations

# Import Path for temporary file annotations.
from pathlib import Path

# Import AnnData for test object construction and type assertions.
import anndata as ad

# Import NumPy for deterministic matrix construction.
import numpy as np

# Import pandas for AnnData obs/var metadata.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import backend primitives for deterministic test registries.
from cellquorum.backends.base import BaseBackend

# Import backend registry for test context construction.
from cellquorum.backends.registry import BackendRegistry

# Import the top-level config model.
from cellquorum.config.models import CellQuorumConfig

# Import pipeline input-loading helpers under test.
from cellquorum.core.pipeline import build_pipeline_context, load_input_adata_from_config

# Import AnnData loading exception type.
from cellquorum.io import AnnDataLoadError


def build_test_backend_registry() -> BackendRegistry:
    """
    Build a deterministic backend registry for pipeline input-loading tests.

    Returns:
        BackendRegistry containing one available Python backend.
    """

    # Create an empty registry.
    registry = BackendRegistry()

    # Register an available Python backend.
    registry.register(BaseBackend(name="python", kind="python"))

    # Return the deterministic registry.
    return registry


def make_test_adata() -> ad.AnnData:
    """
    Build a small deterministic AnnData object.

    Returns:
        AnnData object suitable for h5ad round-trip tests.
    """

    # Build a deterministic count matrix.
    matrix = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ]
    )

    # Build observation metadata.
    obs = pd.DataFrame(
        {
            "sample": ["sample_a", "sample_b"],
        },
        index=["cell_1", "cell_2"],
    )

    # Build variable metadata.
    var = pd.DataFrame(index=["gene_1", "gene_2", "gene_3"])

    # Build the AnnData object.
    adata = ad.AnnData(X=matrix, obs=obs, var=var)

    # Store a raw-counts layer so counts_layer config can be represented.
    adata.layers["counts"] = matrix.copy()

    # Return the AnnData object.
    return adata


def write_test_h5ad(tmp_path: Path, *, filename: str = "input.h5ad") -> Path:
    """
    Write a small AnnData object to h5ad.

    Args:
        tmp_path: Temporary directory.
        filename: h5ad filename.

    Returns:
        Path to the written h5ad file.
    """

    # Build the output path.
    path = tmp_path / filename

    # Write a deterministic AnnData file.
    make_test_adata().write_h5ad(path)

    # Return the h5ad path.
    return path


def build_test_config(
    *,
    h5ad_path: Path | None = None,
    counts_layer: str | None = None,
) -> CellQuorumConfig:
    """
    Build a deterministic CellQuorum config for input-loading tests.

    Args:
        h5ad_path: Optional AnnData h5ad path.
        counts_layer: Optional raw-counts layer name.

    Returns:
        Validated CellQuorumConfig.
    """

    # Build the input block only when needed.
    input_block: dict[str, object] = {}

    # Store the h5ad path when provided.
    if h5ad_path is not None:
        input_block["h5ad"] = str(h5ad_path)

    # Store the counts layer when provided.
    if counts_layer is not None:
        input_block["counts_layer"] = counts_layer

    # Return a deterministic config.
    return CellQuorumConfig(
        project={
            "name": "pipeline_input_project",
        },
        compute={
            "backend": "cpu",
            "prefer_gpu": False,
            "fallback_to_cpu": True,
        },
        r={
            "enabled": False,
        },
        input=input_block,
    )


def test_load_input_adata_from_config_returns_none_without_h5ad() -> None:
    """
    Verify omitted input.h5ad returns None.

    Programmatic workflows may inject AnnData directly, so file input must remain
    optional.
    """

    # Build a config with no input file.
    config = build_test_config()

    # Load configured AnnData.
    loaded = load_input_adata_from_config(config)

    # Confirm no AnnData was loaded.
    assert loaded is None


def test_load_input_adata_from_config_loads_h5ad(tmp_path: Path) -> None:
    """
    Verify configured input.h5ad is loaded into AnnData.

    This connects InputConfig to the AnnData I/O layer.
    """

    # Write an h5ad file.
    h5ad_path = write_test_h5ad(tmp_path)

    # Build a config pointing to the h5ad file.
    config = build_test_config(h5ad_path=h5ad_path, counts_layer="counts")

    # Load configured AnnData.
    loaded = load_input_adata_from_config(config)

    # Confirm an AnnData object was loaded.
    assert isinstance(loaded, ad.AnnData)

    # Confirm the shape round-tripped.
    assert loaded.shape == (2, 3)

    # Confirm metadata round-tripped.
    assert loaded.obs["sample"].tolist() == ["sample_a", "sample_b"]

    # Confirm the counts layer round-tripped.
    assert "counts" in loaded.layers


def test_load_input_adata_from_config_rejects_wrong_config_type() -> None:
    """
    Verify load_input_adata_from_config rejects invalid config objects.

    Helper-level type checking keeps errors clear for external callers.
    """

    # Confirm non-config inputs fail clearly.
    with pytest.raises(TypeError, match="expected a CellQuorumConfig"):
        load_input_adata_from_config(object())  # type: ignore[arg-type]


def test_load_input_adata_from_config_raises_for_missing_h5ad(tmp_path: Path) -> None:
    """
    Verify missing configured h5ad files raise AnnDataLoadError.

    Config validation checks suffix only; runtime input loading checks existence.
    """

    # Build a missing h5ad path.
    missing_path = tmp_path / "missing.h5ad"

    # Build a config pointing to the missing file.
    config = build_test_config(h5ad_path=missing_path)

    # Confirm missing files fail through the AnnData loader.
    with pytest.raises(AnnDataLoadError, match="does not exist"):
        load_input_adata_from_config(config)


def test_build_pipeline_context_does_not_load_input_by_default(tmp_path: Path) -> None:
    """
    Verify build_pipeline_context preserves old behavior by default.

    Adding input loading should not surprise callers that only wanted bootstrap
    context construction.
    """

    # Write an h5ad file.
    h5ad_path = write_test_h5ad(tmp_path)

    # Build a config pointing to the h5ad file.
    config = build_test_config(h5ad_path=h5ad_path, counts_layer="counts")

    # Build the context without load_input=True.
    context = build_pipeline_context(
        config,
        output_dir=tmp_path / "run_without_loading",
        backend_registry=build_test_backend_registry(),
    )

    # Confirm AnnData was not loaded.
    assert context.adata is None

    # Confirm metadata records that input loading did not happen.
    assert context.metadata["input_h5ad"] == str(h5ad_path)
    assert context.metadata["input_counts_layer"] == "counts"
    assert context.metadata["input_loaded"] is False


def test_build_pipeline_context_loads_input_when_requested(tmp_path: Path) -> None:
    """
    Verify build_pipeline_context can load config.input.h5ad into context.adata.

    This is the bridge needed before the executor can run QCStage from config.
    """

    # Write an h5ad file.
    h5ad_path = write_test_h5ad(tmp_path)

    # Build a config pointing to the h5ad file.
    config = build_test_config(h5ad_path=h5ad_path, counts_layer="counts")

    # Build the context with input loading enabled.
    context = build_pipeline_context(
        config,
        output_dir=tmp_path / "run_with_loading",
        backend_registry=build_test_backend_registry(),
        load_input=True,
    )

    # Confirm AnnData was loaded into context.
    assert isinstance(context.adata, ad.AnnData)

    # Confirm the loaded AnnData has the expected shape.
    assert context.adata.shape == (2, 3)

    # Confirm metadata records the loaded input.
    assert context.metadata["input_h5ad"] == str(h5ad_path)
    assert context.metadata["input_counts_layer"] == "counts"
    assert context.metadata["input_loaded"] is True


def test_build_pipeline_context_load_input_true_without_h5ad_keeps_adata_none(
    tmp_path: Path,
) -> None:
    """
    Verify load_input=True is safe when no input.h5ad is configured.

    This keeps programmatic no-file workflows valid.
    """

    # Build a config without an h5ad path.
    config = build_test_config()

    # Build context with load_input enabled.
    context = build_pipeline_context(
        config,
        output_dir=tmp_path / "run_without_h5ad",
        backend_registry=build_test_backend_registry(),
        load_input=True,
    )

    # Confirm no AnnData was loaded.
    assert context.adata is None

    # Confirm metadata records no configured file.
    assert context.metadata["input_h5ad"] is None
    assert context.metadata["input_counts_layer"] is None
    assert context.metadata["input_loaded"] is False


def _config_with_cohort(h5ad_path: Path, cohort: dict[str, object]) -> CellQuorumConfig:
    """Build an input-loading config that also declares a cohort block."""
    return CellQuorumConfig(
        project={"name": "cohort_project"},
        compute={"backend": "cpu", "prefer_gpu": False, "fallback_to_cpu": True},
        r={"enabled": False},
        input={"h5ad": str(h5ad_path), "counts_layer": "counts"},
        cohort=cohort,
    )


def test_build_pipeline_context_warns_on_cohort_key_absent_from_obs(tmp_path: Path, caplog) -> None:
    """A cohort key declared in config but absent from obs must warn (wired guard).

    The test AnnData carries only a ``sample`` obs column; a cohort declaring
    ``donor_key='patient_id'`` therefore references a missing column. The startup
    guard must log the mismatch and record it in run metadata rather than letting
    it surface only as an obscure per-stage fallback later.
    """
    h5ad_path = write_test_h5ad(tmp_path)
    config = _config_with_cohort(h5ad_path, {"sample_key": "sample", "donor_key": "patient_id"})

    with caplog.at_level("WARNING"):
        context = build_pipeline_context(
            config,
            output_dir=tmp_path / "cohort_mismatch_run",
            backend_registry=build_test_backend_registry(),
            load_input=True,
        )

    warnings = context.metadata["cohort_warnings"]
    assert any("donor_key" in warning and "patient_id" in warning for warning in warnings)
    # sample_key IS present, so it must not be flagged.
    assert not any("sample_key" in warning for warning in warnings)
    assert any("patient_id" in record.message for record in caplog.records)


def test_build_pipeline_context_clean_cohort_records_no_warnings(tmp_path: Path) -> None:
    """A cohort whose declared keys all exist in obs records no warnings."""
    h5ad_path = write_test_h5ad(tmp_path)
    config = _config_with_cohort(h5ad_path, {"sample_key": "sample"})

    context = build_pipeline_context(
        config,
        output_dir=tmp_path / "cohort_clean_run",
        backend_registry=build_test_backend_registry(),
        load_input=True,
    )

    assert context.metadata["cohort_warnings"] == []


def test_build_pipeline_context_load_input_true_raises_for_missing_h5ad(
    tmp_path: Path,
) -> None:
    """
    Verify context construction raises when requested input loading fails.

    This prevents downstream stages from running with an absent AnnData object
    when the user explicitly configured an input file.
    """

    # Build a config pointing to a missing h5ad file.
    config = build_test_config(h5ad_path=tmp_path / "missing.h5ad")

    # Confirm input loading failure propagates clearly.
    with pytest.raises(AnnDataLoadError, match="does not exist"):
        build_pipeline_context(
            config,
            output_dir=tmp_path / "bad_input_run",
            backend_registry=build_test_backend_registry(),
            load_input=True,
        )
