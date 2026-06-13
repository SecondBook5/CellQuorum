"""Quality-control public API for CellQuorum."""

from __future__ import annotations

# Import QC artifact writing public objects.
from cellquorum.qc.artifacts import (
    QCArtifactError,
    QCArtifactManifest,
    write_qc_artifacts,
)

# Import QC configuration public objects.
from cellquorum.qc.config import (
    QCAmbientRNAConfig,
    QCBasicThresholdConfig,
    QCConfig,
    QCDoubletConfig,
    QCDuplicateNameConfig,
    QCFeaturePatternConfig,
    QCMadThresholdConfig,
    QCMetricCalculationConfig,
    QCOutputConfig,
    validate_qc_config_dict,
)

# Import QC decision public objects.
from cellquorum.qc.decisions import (
    QCDecisionError,
    QCDecisionResult,
    build_qc_decisions,
)

# Import QC feature annotation public objects.
from cellquorum.qc.features import (
    CUSTOM_EXCLUDE_COLUMN,
    HEMOGLOBIN_COLUMN,
    MITO_COLUMN,
    RIBO_COLUMN,
    QCFeatureAnnotationError,
    QCFeatureMaskSummary,
    annotate_qc_feature_masks,
    build_feature_masks,
    summarize_feature_masks,
)

# Import QC metric calculation public objects.
from cellquorum.qc.metrics import (
    QCMetricsError,
    QCMetricsResult,
    calculate_qc_metrics,
)

# Import QC stage public objects.
from cellquorum.qc.stage import (
    QCStage,
    QCStageError,
)

# Import QC threshold public objects.
from cellquorum.qc.thresholds import (
    QCThreshold,
    QCThresholdError,
    QCThresholdResult,
    build_qc_thresholds,
)

# Import QC input validation public objects.
from cellquorum.qc.validation import (
    QCInputValidationError,
    QCInputValidationSummary,
    get_qc_matrix,
    require_obs_columns,
    summarize_adata_shape,
    validate_duplicate_name_policy,
    validate_mad_groupby_columns,
    validate_qc_input_adata,
    validate_qc_matrix,
)

__all__ = [
    "CUSTOM_EXCLUDE_COLUMN",
    "HEMOGLOBIN_COLUMN",
    "MITO_COLUMN",
    "QCAmbientRNAConfig",
    "QCArtifactError",
    "QCArtifactManifest",
    "QCBasicThresholdConfig",
    "QCConfig",
    "QCDecisionError",
    "QCDecisionResult",
    "QCDoubletConfig",
    "QCDuplicateNameConfig",
    "QCFeatureAnnotationError",
    "QCFeatureMaskSummary",
    "QCFeaturePatternConfig",
    "QCInputValidationError",
    "QCInputValidationSummary",
    "QCMadThresholdConfig",
    "QCMetricCalculationConfig",
    "QCMetricsError",
    "QCMetricsResult",
    "QCOutputConfig",
    "QCStage",
    "QCStageError",
    "QCThreshold",
    "QCThresholdError",
    "QCThresholdResult",
    "RIBO_COLUMN",
    "annotate_qc_feature_masks",
    "build_feature_masks",
    "build_qc_decisions",
    "build_qc_thresholds",
    "calculate_qc_metrics",
    "get_qc_matrix",
    "require_obs_columns",
    "summarize_adata_shape",
    "summarize_feature_masks",
    "validate_duplicate_name_policy",
    "validate_mad_groupby_columns",
    "validate_qc_config_dict",
    "validate_qc_input_adata",
    "validate_qc_matrix",
    "write_qc_artifacts",
]
