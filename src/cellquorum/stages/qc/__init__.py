"""Quality-control public API for CellQuorum.

Two design documents are normative for this package, and both are tracked in the repository
rather than kept as working notes:

``docs/design/qc-graded-adjudication.md``
    The architecture: evidence families, concordance, ``core``/``borderline``/``quarantine``,
    per-analysis eligibility, and the stage order (``qc_evidence`` at 20, ``query_projection``
    at 105, ``qc_finalization`` at 135). **Frozen** — do not change the architecture or invent
    numeric thresholds against it.

``docs/design/qc-reporting-figures.md``
    The figure and table contract: which figures exist, paired control-first ordering, the rule
    that a plot may never substitute ``0`` for evidence that was not measured, and the
    requirement that the visualization layer *consume* QC state rather than recompute it.

They are cited here because they were previously gitignored, which is how a session rebuilt the
figure set twice against a spec that had been on disk the whole time. A normative document the
code depends on belongs where the code is.
"""

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
    QCConfig,
    QCDoubletConfig,
    QCDuplicateNameConfig,
    QCFeaturePatternConfig,
    QCFloorConfig,
    QCMetricCalculationConfig,
    QCOutputConfig,
    validate_qc_config_dict,
)

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

# Import QC input validation public objects.
from cellquorum.stages.qc.validation import (
    QCInputValidationError,
    QCInputValidationSummary,
    get_qc_matrix,
    require_obs_columns,
    summarize_adata_shape,
    validate_duplicate_name_policy,
    validate_mixture_groupby_columns,
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
    "QCFloorConfig",
    "QCConfig",
    "QCDoubletConfig",
    "QCDuplicateNameConfig",
    "QCFeatureAnnotationError",
    "QCFeatureMaskSummary",
    "QCFeaturePatternConfig",
    "QCInputValidationError",
    "QCInputValidationSummary",
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
    "validate_mixture_groupby_columns",
    "validate_qc_config_dict",
    "validate_qc_input_adata",
    "validate_qc_matrix",
    "write_qc_artifacts",
]
