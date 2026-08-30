"""Tests for the public CellQuorum QC API."""

from __future__ import annotations

# Import AnnData for public workflow smoke tests.
import anndata as ad

# Import NumPy for deterministic test matrices.
import numpy as np

# Import pandas for AnnData metadata.
import pandas as pd

# Import the public QC package API.
import cellquorum.stages.qc as qc

# Import artifact objects for identity checks.
from cellquorum.stages.qc.artifacts import QCArtifactManifest, write_qc_artifacts

# Import config objects for identity checks.
from cellquorum.stages.qc.config import QCConfig, validate_qc_config_dict

# Import decision objects for identity checks.
from cellquorum.stages.qc.decisions import QCDecisionResult, build_qc_decisions

# Import feature objects for identity checks.
from cellquorum.stages.qc.features import MITO_COLUMN, build_feature_masks

# Import metric objects for identity checks.
from cellquorum.stages.qc.metrics import QCMetricsResult, calculate_qc_metrics

# Import stage objects for identity checks.
from cellquorum.stages.qc.stage import QCStage

# Import threshold objects for identity checks.
from cellquorum.stages.qc.thresholds import QCThresholdResult, build_qc_thresholds

# Import validation objects for identity checks.
from cellquorum.stages.qc.validation import QCInputValidationSummary, validate_qc_input_adata


def make_public_api_adata() -> ad.AnnData:
    """
    Build a small AnnData object for public API smoke tests.

    Returns:
        Small AnnData object with mitochondrial and non-mitochondrial genes.
    """

    # Build a deterministic count matrix.
    matrix = np.array(
        [
            [5.0, 5.0, 0.0],
            [9.0, 0.0, 0.0],
            [0.0, 2.0, 1.0],
        ]
    )

    # Build observation metadata.
    obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])

    # Build variable metadata.
    var = pd.DataFrame(index=["MT-ND1", "ACTB", "MALAT1"])

    # Return AnnData.
    return ad.AnnData(X=matrix, obs=obs, var=var)


def make_public_api_qc_config() -> qc.QCConfig:
    """
    Build a deterministic QCConfig through the public API.

    Returns:
        QCConfig configured with fixed thresholds only.
    """

    # Return fixed-threshold QC configuration.
    return qc.QCConfig(
        mode="flag_no_drop",
        threshold_strategy="fixed",
        metrics={"percent_top": [2]},
        basic={
            "min_genes_per_cell": 2,
            "min_cells_per_gene": 2,
            "max_mito_percent": 60.0,
        },
        mad={"enabled": False},
        outputs={
            "write_h5ad": False,
            "write_figures": False,
        },
    )


def test_qc_public_api_exports_expected_core_objects() -> None:
    """
    Verify the QC package exports stable user-facing objects.

    The public API should let callers use the QC module without importing from
    internal submodules directly.
    """

    # Confirm config exports.
    assert qc.QCConfig is QCConfig
    assert qc.validate_qc_config_dict is validate_qc_config_dict

    # Confirm validation exports.
    assert qc.QCInputValidationSummary is QCInputValidationSummary
    assert qc.validate_qc_input_adata is validate_qc_input_adata

    # Confirm feature exports.
    assert qc.MITO_COLUMN is MITO_COLUMN
    assert qc.build_feature_masks is build_feature_masks

    # Confirm metric exports.
    assert qc.QCMetricsResult is QCMetricsResult
    assert qc.calculate_qc_metrics is calculate_qc_metrics

    # Confirm threshold exports.
    assert qc.QCThresholdResult is QCThresholdResult
    assert qc.build_qc_thresholds is build_qc_thresholds

    # Confirm decision exports.
    assert qc.QCDecisionResult is QCDecisionResult
    assert qc.build_qc_decisions is build_qc_decisions

    # Confirm artifact exports.
    assert qc.QCArtifactManifest is QCArtifactManifest
    assert qc.write_qc_artifacts is write_qc_artifacts

    # Confirm stage exports.
    assert qc.QCStage is QCStage


def test_qc_public_api_all_is_unique_and_contains_no_private_names() -> None:
    """
    Verify __all__ is explicit, unique, and public.

    This protects wildcard imports and docs generation from accidental private
    symbol exposure.
    """

    # Convert __all__ to a list.
    exported_names = list(qc.__all__)

    # Confirm exported names are unique.
    assert len(exported_names) == len(set(exported_names))

    # Confirm no private names are exported.
    assert all(not name.startswith("_") for name in exported_names)

    # Confirm every exported name exists on the package.
    assert all(hasattr(qc, name) for name in exported_names)


def test_qc_public_api_can_validate_config_mapping() -> None:
    """
    Verify QC config mappings can be validated through the package API.

    This is the path future YAML integration will use.
    """

    # Validate a simple QC config mapping.
    config = qc.validate_qc_config_dict(
        {
            "enabled": True,
            "mode": "flag_no_drop",
            "threshold_strategy": "fixed",
            "mad": {"enabled": False},
        }
    )

    # Confirm a QCConfig was returned.
    assert isinstance(config, qc.QCConfig)

    # Confirm configured fields were preserved.
    assert config.enabled is True
    assert config.mode == "flag_no_drop"
    assert config.threshold_strategy == "fixed"


def test_qc_public_api_can_run_metrics_thresholds_and_decisions() -> None:
    """
    Verify the public API supports the core QC workflow.

    This smoke test exercises the intended user-facing sequence:
    validate input, calculate metrics, build thresholds, and build decisions.
    """

    # Build test data and config.
    adata = make_public_api_adata()
    config = make_public_api_qc_config()

    # Validate QC input through the public API.
    validation_summary = qc.validate_qc_input_adata(adata, config)

    # Calculate QC metrics through the public API.
    metrics_result = qc.calculate_qc_metrics(adata, config)

    # Build QC thresholds through the public API.
    threshold_result = qc.build_qc_thresholds(
        cell_metrics=metrics_result.cell_metrics,
        gene_metrics=metrics_result.gene_metrics,
        config=config,
    )

    # Build QC decisions through the public API.
    decision_result = qc.build_qc_decisions(
        cell_metrics=metrics_result.cell_metrics,
        gene_metrics=metrics_result.gene_metrics,
        thresholds=threshold_result,
        config=config,
    )

    # Confirm validation returned a structured summary.
    assert isinstance(validation_summary, qc.QCInputValidationSummary)

    # Confirm metrics returned a structured result.
    assert isinstance(metrics_result, qc.QCMetricsResult)

    # Confirm thresholds returned a structured result.
    assert isinstance(threshold_result, qc.QCThresholdResult)

    # Confirm decisions returned a structured result.
    assert isinstance(decision_result, qc.QCDecisionResult)

    # Confirm the expected fixed-threshold decision behavior.
    assert decision_result.cell_decisions["keep"].tolist() == [True, False, True]
    assert decision_result.gene_decisions["keep"].tolist() == [True, True, False]


def test_qc_public_api_stage_can_be_instantiated() -> None:
    """
    Verify QCStage can be instantiated from the public API.

    This confirms downstream code can access the pipeline stage as
    cellquorum.stages.qc.QCStage.
    """

    # Build a QC stage through the public API.
    stage = qc.QCStage(config=make_public_api_qc_config())

    # Confirm the stable stage name.
    assert stage.name == "qc"

    # Confirm the config was retained.
    assert isinstance(stage.config, qc.QCConfig)
