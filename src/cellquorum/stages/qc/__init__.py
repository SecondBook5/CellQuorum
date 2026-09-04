"""Quality-control public API for CellQuorum."""

from __future__ import annotations

# Import QC artifact writing public objects.
from cellquorum.stages.qc.artifacts import (
    QCArtifactError,
    QCArtifactManifest,
    write_qc_artifacts,
)

# Import QC configuration public objects.
from cellquorum.stages.qc.config import (
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
# Import QC feature annotation public objects.
from cellquorum.stages.qc.features import (
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

# Import the floor path that replaced fixed-and-MAD thresholds.
from cellquorum.stages.qc.floors import (
    FloorResult,
    QCFloorError,
    apply_floors,
    build_qc_report_table,
)

# Import QC metric calculation public objects.
from cellquorum.stages.qc.metrics import (
    QCMetricsError,
    QCMetricsResult,
    calculate_qc_metrics,
)

# Import QC stage public objects.
from cellquorum.stages.qc.stage import (
    QCStage,
    QCStageError,
)

# Import QC threshold public objects.
# Import QC input validation public objects.
from cellquorum.stages.qc.validation import (
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
    "FloorResult",
    "QCFloorError",
    "apply_floors",
    "build_qc_report_table",
    "CUSTOM_EXCLUDE_COLUMN",
    "HEMOGLOBIN_COLUMN",
    "MITO_COLUMN",
    "QCAmbientRNAConfig",
    "QCArtifactError",
    "QCArtifactManifest",
    "QCBasicThresholdConfig",
    "QCConfig",
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
    "RIBO_COLUMN",
    "annotate_qc_feature_masks",
    "build_feature_masks",
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
