"""QC configuration models for CellQuorum."""

from __future__ import annotations

# Import Mapping for dictionary-based QC config validation.
from collections.abc import Mapping

# Import Literal for constrained QC configuration values.
from typing import Literal

# Import Pydantic validation utilities.
from pydantic import Field, ValidationError, field_validator, model_validator

# Import the strict base model used by CellQuorum configuration models.
from cellquorum.config.base import StrictBaseModel

# Import reusable configuration validation helpers.
from cellquorum.config.validation import (
    ConfigValidationError,
    reject_unknown_keys,
    require_mapping,
)

# Define supported QC execution modes.
type QCMode = Literal["report_only", "filter", "both"]

# Define supported QC threshold strategies.
type ThresholdStrategy = Literal["fixed", "mad", "fixed_and_mad"]

# Define supported doublet detection methods.
type DoubletMethod = Literal["none", "scrublet", "scdblfinder"]

# Define supported ambient RNA methods.
type AmbientMethod = Literal["none", "audit", "soupx", "decontx"]

# Define supported duplicate-name handling policies.
type DuplicateNamePolicy = Literal["warn", "make_unique", "error", "ignore"]

# Define supported QC figure output formats.
type QCFigureFormat = Literal["png", "pdf", "svg"]


class QCMetricCalculationConfig(StrictBaseModel):
    """
    Store settings for QC metric calculation.

    Single-cell QC depends on transparent metric calculation before any filtering
    happens. This config controls Scanpy-compatible metrics such as total counts,
    detected genes, log1p-transformed count metrics, and the percentage of counts
    contained in the most highly expressed genes. The default `percent_top=[20]`
    matches the Single-cell Best Practices workflow.

    Args:
        percent_top: Top-n gene ranks used to calculate cumulative count fractions.
        log1p: Whether log1p QC metrics should be calculated.
        layer: Optional AnnData layer to use for QC metric calculation.
        use_raw: Whether AnnData.raw should be used when available.
    """

    # Store top-n gene ranks for percent-top QC metrics.
    percent_top: list[int] = Field(default_factory=lambda: [20])

    # Store whether log1p QC metrics should be calculated.
    log1p: bool = True

    # Store an optional AnnData layer used for QC.
    layer: str | None = None

    # Store whether AnnData.raw should be used when available.
    use_raw: bool = False

    @field_validator("percent_top", mode="before")
    @classmethod
    def validate_percent_top(cls, value: object) -> list[int]:
        """
        Validate percent-top settings.

        Args:
            value: Candidate percent_top value.

        Returns:
            Cleaned list of positive integer ranks.

        Raises:
            ValueError: If the value is not a non-empty list of positive integers.
        """

        # Reject a missing percent_top list.
        if value is None:
            raise ValueError("percent_top cannot be None.")

        # Reject strings because they are sequences but not valid integer lists.
        if isinstance(value, str):
            raise ValueError("percent_top must be a list of positive integers, not a string.")

        # Reject non-list and non-tuple values.
        if not isinstance(value, list | tuple):
            raise ValueError(
                "percent_top must be a list of positive integers. "
                f"Received: {type(value).__name__}."
            )

        # Reject empty percent_top lists.
        if not value:
            raise ValueError("percent_top must contain at least one positive integer.")

        # Initialize the cleaned percent_top list.
        cleaned_values: list[int] = []

        # Iterate over each candidate value.
        for item in value:
            # Reject booleans because bool is a subclass of int.
            if isinstance(item, bool):
                raise ValueError("percent_top values must be integers, not booleans.")

            # Reject non-integer values.
            if not isinstance(item, int):
                raise ValueError(
                    "percent_top values must be integers. " f"Received: {type(item).__name__}."
                )

            # Reject non-positive values.
            if item <= 0:
                raise ValueError("percent_top values must be > 0.")

            # Store the cleaned integer.
            cleaned_values.append(item)

        # Return the sorted unique values for deterministic metric names.
        return sorted(set(cleaned_values))

    @field_validator("layer", mode="before")
    @classmethod
    def validate_optional_layer(cls, value: object) -> str | None:
        """
        Validate the optional AnnData layer name.

        Args:
            value: Candidate layer name.

        Returns:
            Cleaned layer name or None.

        Raises:
            ValueError: If the layer name is empty or non-string.
        """

        # Preserve absent layer names.
        if value is None:
            return None

        # Reject non-string layer names.
        if not isinstance(value, str):
            raise ValueError(
                "QC metric layer must be a string or None. " f"Received: {type(value).__name__}."
            )

        # Strip harmless whitespace.
        cleaned_value = value.strip()

        # Reject empty layer names.
        if not cleaned_value:
            raise ValueError("QC metric layer cannot be empty.")

        # Return the cleaned layer name.
        return cleaned_value


class QCFeaturePatternConfig(StrictBaseModel):
    """
    Store gene-feature patterns used by QC metrics.

    Feature-family definitions are configurable because gene naming conventions
    differ across species and references. Human mitochondrial genes often use
    `MT-`, while mouse mitochondrial genes often use `mt-`. Hemoglobin is stored
    as a regex pattern by default because naive `HB` prefix matching is too broad.

    Args:
        mitochondrial_prefixes: Prefixes treated as mitochondrial genes.
        ribosomal_prefixes: Prefixes treated as ribosomal genes.
        hemoglobin_regexes: Regex patterns treated as hemoglobin genes.
        custom_exclude_prefixes: Optional project-specific prefixes flagged for QC.
    """

    # Store mitochondrial gene prefixes.
    mitochondrial_prefixes: list[str] = Field(default_factory=lambda: ["MT-"])

    # Store ribosomal gene prefixes.
    ribosomal_prefixes: list[str] = Field(default_factory=lambda: ["RPS", "RPL"])

    # Store hemoglobin regex patterns.
    hemoglobin_regexes: list[str] = Field(default_factory=lambda: [r"^HB[ABDEGMQZ]\d*(?!\w)"])

    # Store optional custom gene prefixes flagged for QC review.
    custom_exclude_prefixes: list[str] = Field(default_factory=list)

    @field_validator(
        "mitochondrial_prefixes",
        "ribosomal_prefixes",
        "hemoglobin_regexes",
        "custom_exclude_prefixes",
        mode="before",
    )
    @classmethod
    def validate_string_patterns(cls, value: object) -> list[str]:
        """
        Validate a string-pattern list.

        Args:
            value: Candidate pattern list.

        Returns:
            Cleaned pattern list.

        Raises:
            ValueError: If the value is not a list of non-empty strings.
        """

        # Return an empty list when an optional list is omitted.
        if value is None:
            return []

        # Reject a single string because users should provide a list explicitly.
        if isinstance(value, str):
            raise ValueError("Feature patterns must be provided as a list, not a string.")

        # Reject non-list and non-tuple values.
        if not isinstance(value, list | tuple):
            raise ValueError(
                "Feature patterns must be provided as a list of strings. "
                f"Received: {type(value).__name__}."
            )

        # Initialize the cleaned pattern list.
        cleaned_patterns: list[str] = []

        # Iterate over candidate patterns.
        for item in value:
            # Reject non-string patterns.
            if not isinstance(item, str):
                raise ValueError(
                    "Feature patterns must be strings. " f"Received: {type(item).__name__}."
                )

            # Strip harmless whitespace.
            cleaned_item = item.strip()

            # Reject empty patterns.
            if not cleaned_item:
                raise ValueError("Feature patterns cannot be empty.")

            # Store the cleaned pattern.
            cleaned_patterns.append(cleaned_item)

        # Return the cleaned patterns.
        return cleaned_patterns


class QCBasicThresholdConfig(StrictBaseModel):
    """
    Store fixed QC thresholds.

    Fixed thresholds are transparent and useful for reproducibility, but they can
    remove real biology if applied too aggressively. CellQuorum therefore keeps
    QC in report-only mode by default and makes all fixed thresholds auditable.

    Args:
        min_genes_per_cell: Optional minimum detected genes per barcode.
        max_genes_per_cell: Optional maximum detected genes per barcode.
        min_counts_per_cell: Optional minimum total counts per barcode.
        max_counts_per_cell: Optional maximum total counts per barcode.
        min_cells_per_gene: Optional minimum cells in which a gene must be detected.
        max_mito_percent: Optional hard mitochondrial percentage cutoff.
        max_ribo_percent: Optional hard ribosomal percentage cutoff.
        max_hemoglobin_percent: Optional hard hemoglobin percentage cutoff.
    """

    # Store the optional minimum detected genes per barcode.
    min_genes_per_cell: int | None = 200

    # Store the optional maximum detected genes per barcode.
    max_genes_per_cell: int | None = None

    # Store the optional minimum total counts per barcode.
    min_counts_per_cell: int | None = None

    # Store the optional maximum total counts per barcode.
    max_counts_per_cell: int | None = None

    # Store the optional minimum cells per detected gene.
    min_cells_per_gene: int | None = 3

    # Store the optional hard mitochondrial percentage cutoff.
    max_mito_percent: float | None = 8.0

    # Store the optional hard ribosomal percentage cutoff.
    max_ribo_percent: float | None = None

    # Store the optional hard hemoglobin percentage cutoff.
    max_hemoglobin_percent: float | None = None

    @field_validator(
        "min_genes_per_cell",
        "max_genes_per_cell",
        "min_counts_per_cell",
        "max_counts_per_cell",
        "min_cells_per_gene",
        mode="before",
    )
    @classmethod
    def validate_optional_non_negative_int(cls, value: object) -> int | None:
        """
        Validate optional non-negative integer thresholds.

        Args:
            value: Candidate threshold.

        Returns:
            Validated integer threshold or None.

        Raises:
            ValueError: If the value is negative, boolean, or non-integer.
        """

        # Preserve absent thresholds.
        if value is None:
            return None

        # Reject booleans because bool is a subclass of int.
        if isinstance(value, bool):
            raise ValueError("Integer QC thresholds cannot be boolean values.")

        # Reject non-integer values.
        if not isinstance(value, int):
            raise ValueError(
                "Integer QC thresholds must be integers. " f"Received: {type(value).__name__}."
            )

        # Reject negative values.
        if value < 0:
            raise ValueError("Integer QC thresholds must be >= 0.")

        # Return the validated integer.
        return value

    @field_validator(
        "max_mito_percent",
        "max_ribo_percent",
        "max_hemoglobin_percent",
        mode="before",
    )
    @classmethod
    def validate_optional_percent(cls, value: object) -> float | None:
        """
        Validate optional percentage thresholds.

        Args:
            value: Candidate percentage threshold.

        Returns:
            Validated percentage threshold or None.

        Raises:
            ValueError: If the value is outside [0, 100], boolean, or non-numeric.
        """

        # Preserve absent thresholds.
        if value is None:
            return None

        # Reject booleans because bool values are not valid percentages.
        if isinstance(value, bool):
            raise ValueError("Percentage QC thresholds cannot be boolean values.")

        # Reject non-numeric values.
        if not isinstance(value, int | float):
            raise ValueError(
                "Percentage QC thresholds must be numeric. " f"Received: {type(value).__name__}."
            )

        # Convert the threshold to float.
        float_value = float(value)

        # Reject values outside the valid percentage range.
        if float_value < 0.0 or float_value > 100.0:
            raise ValueError("Percentage QC thresholds must be between 0 and 100.")

        # Return the validated percentage.
        return float_value

    @model_validator(mode="after")
    def validate_threshold_pairs(self) -> QCBasicThresholdConfig:
        """
        Validate fixed min/max threshold pairs.

        Returns:
            Validated fixed-threshold configuration.

        Raises:
            ValueError: If a minimum threshold exceeds its paired maximum.
        """

        # Validate detected gene threshold ordering when both values are present.
        if (
            self.min_genes_per_cell is not None
            and self.max_genes_per_cell is not None
            and self.min_genes_per_cell > self.max_genes_per_cell
        ):
            raise ValueError("min_genes_per_cell cannot exceed max_genes_per_cell.")

        # Validate total count threshold ordering when both values are present.
        if (
            self.min_counts_per_cell is not None
            and self.max_counts_per_cell is not None
            and self.min_counts_per_cell > self.max_counts_per_cell
        ):
            raise ValueError("min_counts_per_cell cannot exceed max_counts_per_cell.")

        # Return the validated model.
        return self


class QCMadThresholdConfig(StrictBaseModel):
    """
    Store adaptive MAD-based QC settings.

    MAD-based filtering supports permissive, dataset-aware outlier detection.
    The default general threshold is 5 MADs, matching the best-practices example.
    Mitochondrial filtering is configured separately because it is commonly
    handled more strictly.

    Args:
        enabled: Whether MAD-based QC thresholds are enabled.
        n_mads: Number of MADs for general outlier metrics.
        metrics: QC metrics evaluated with the general MAD threshold.
        mito_metric: Mitochondrial percentage metric.
        mito_n_mads: Number of MADs for mitochondrial outlier detection.
        groupby: Optional metadata columns used for group-wise MAD thresholds.
        log1p_metrics: Whether count-like metrics should be log1p-transformed upstream.
        skip_zero_mad: Whether zero-MAD metrics should be skipped with a warning.
    """

    # Store whether MAD thresholding is enabled.
    enabled: bool = True

    # Store the general MAD multiplier.
    n_mads: float = 5.0

    # Store the default general outlier metrics.
    metrics: list[str] = Field(
        default_factory=lambda: [
            "log1p_total_counts",
            "log1p_n_genes_by_counts",
            "pct_counts_in_top_20_genes",
        ]
    )

    # Store the mitochondrial metric evaluated separately.
    mito_metric: str = "pct_counts_mito"

    # Store the mitochondrial MAD multiplier.
    mito_n_mads: float = 3.0

    # Store optional grouping columns for group-wise MAD thresholding.
    groupby: list[str] = Field(default_factory=list)

    # Store whether count-like metrics should be log1p transformed upstream.
    log1p_metrics: bool = True

    # Store whether zero-MAD metrics should be skipped instead of forced.
    skip_zero_mad: bool = True

    @field_validator("n_mads", "mito_n_mads", mode="before")
    @classmethod
    def validate_positive_float(cls, value: object) -> float:
        """
        Validate positive floating-point threshold settings.

        Args:
            value: Candidate numeric value.

        Returns:
            Validated positive float.

        Raises:
            ValueError: If the value is boolean, non-numeric, or non-positive.
        """

        # Reject booleans because they behave numerically but are invalid here.
        if isinstance(value, bool):
            raise ValueError("MAD multipliers cannot be boolean values.")

        # Reject non-numeric values.
        if not isinstance(value, int | float):
            raise ValueError(
                "MAD multipliers must be numeric. " f"Received: {type(value).__name__}."
            )

        # Convert the value to float.
        float_value = float(value)

        # Reject non-positive multipliers.
        if float_value <= 0.0:
            raise ValueError("MAD multipliers must be > 0.")

        # Return the validated multiplier.
        return float_value

    @field_validator("metrics", "groupby", mode="before")
    @classmethod
    def validate_string_list(cls, value: object) -> list[str]:
        """
        Validate a list of non-empty strings.

        Args:
            value: Candidate string list.

        Returns:
            Cleaned list of strings.

        Raises:
            ValueError: If the value is not a list of non-empty strings.
        """

        # Return an empty list when an optional list is omitted.
        if value is None:
            return []

        # Reject a single string because it is ambiguous.
        if isinstance(value, str):
            raise ValueError("MAD fields must be provided as lists, not strings.")

        # Reject non-list and non-tuple values.
        if not isinstance(value, list | tuple):
            raise ValueError(
                "MAD fields must be lists of strings. " f"Received: {type(value).__name__}."
            )

        # Initialize the cleaned list.
        cleaned_values: list[str] = []

        # Iterate over candidate entries.
        for item in value:
            # Reject non-string entries.
            if not isinstance(item, str):
                raise ValueError(
                    "MAD list entries must be strings. " f"Received: {type(item).__name__}."
                )

            # Strip harmless whitespace.
            cleaned_item = item.strip()

            # Reject empty entries.
            if not cleaned_item:
                raise ValueError("MAD list entries cannot be empty.")

            # Store the cleaned entry.
            cleaned_values.append(cleaned_item)

        # Return the cleaned values.
        return cleaned_values

    @field_validator("mito_metric", mode="before")
    @classmethod
    def validate_mito_metric(cls, value: object) -> str:
        """
        Validate the mitochondrial metric name.

        Args:
            value: Candidate mitochondrial metric name.

        Returns:
            Cleaned mitochondrial metric name.

        Raises:
            ValueError: If the metric name is not a non-empty string.
        """

        # Reject non-string metric names.
        if not isinstance(value, str):
            raise ValueError("mito_metric must be a string. " f"Received: {type(value).__name__}.")

        # Strip harmless whitespace.
        cleaned_value = value.strip()

        # Reject empty metric names.
        if not cleaned_value:
            raise ValueError("mito_metric cannot be empty.")

        # Return the cleaned metric name.
        return cleaned_value


class QCDoubletConfig(StrictBaseModel):
    """
    Store doublet-detection settings.

    Doublet detection is enabled as an audit by default. The default method is
    scDblFinder because it is highlighted in the best-practices workflow, but
    automatic removal is disabled so users can inspect doublets before filtering.

    Args:
        enabled: Whether doublet detection or auditing is enabled.
        method: Doublet detection method.
        methods: Detectors to run (consensus over these); overrides single method when set.
        consensus: How to combine per-method calls: any | all | majority.
        remove: Whether predicted doublets should be removed automatically.
        expected_doublet_rate: Expected doublet rate as a probability.
        score_threshold: Optional manual doublet score threshold.
        per_sample: Whether doublets should be evaluated per sample.
    """

    # Store whether doublet auditing is enabled.
    enabled: bool = True

    # Store the selected doublet detection method.
    method: DoubletMethod = "scdblfinder"

    # Store detectors to run (consensus over these); overrides single method when
    # set. Default matches the single-method default (`method`) so an unconfigured
    # run uses the same detector regardless of which field the caller reads.
    methods: list[str] = Field(default_factory=lambda: ["scdblfinder"])

    # Store how to combine per-method calls: any | all | majority.
    consensus: str = "any"

    # Store whether predicted doublets should be removed automatically.
    remove: bool = False

    # Store the expected doublet rate.
    expected_doublet_rate: float = 0.06

    # Store an optional manual doublet score threshold.
    score_threshold: float | None = None

    # Store whether doublets should be evaluated per sample.
    per_sample: bool = True

    @field_validator("expected_doublet_rate", "score_threshold", mode="before")
    @classmethod
    def validate_optional_probability(cls, value: object) -> float | None:
        """
        Validate optional probability fields.

        Args:
            value: Candidate probability value.

        Returns:
            Validated probability or None.

        Raises:
            ValueError: If the value is outside [0, 1], boolean, or non-numeric.
        """

        # Preserve absent optional probabilities.
        if value is None:
            return None

        # Reject booleans because they are not valid probabilities.
        if isinstance(value, bool):
            raise ValueError("Doublet probability fields cannot be boolean values.")

        # Reject non-numeric values.
        if not isinstance(value, int | float):
            raise ValueError(
                "Doublet probability fields must be numeric. " f"Received: {type(value).__name__}."
            )

        # Convert the value to float.
        float_value = float(value)

        # Reject values outside the probability range.
        if float_value < 0.0 or float_value > 1.0:
            raise ValueError("Doublet probability fields must be between 0 and 1.")

        # Return the validated probability.
        return float_value

    @model_validator(mode="after")
    def validate_doublet_consistency(self) -> QCDoubletConfig:
        """
        Validate consistency between doublet flags and method.

        Returns:
            Validated doublet configuration.

        Raises:
            ValueError: If doublets are enabled or removed with method none.
        """

        # Reject enabled doublet auditing without a real method.
        if self.enabled and self.method == "none":
            raise ValueError("Doublet detection cannot be enabled when method is 'none'.")

        # Reject doublet removal without a real method.
        if self.remove and self.method == "none":
            raise ValueError("Doublet removal cannot be enabled when method is 'none'.")

        # Return the validated model.
        return self


class QCCellCycleConfig(StrictBaseModel):
    """
    Store cell-cycle scoring settings (opt-in).

    Cell-cycle scoring is disabled by default and is intended to run on the
    log-normalized layer. Gene lists default to empty; callers fill them from
    the Tirosh constants when empty to avoid import cycles.

    Args:
        enabled: Whether cell-cycle scoring is enabled.
        score_layer: Layer to score on (must be log-normalized).
        s_genes: S-phase gene list (default empty; filled by caller).
        g2m_genes: G2M-phase gene list (default empty; filled by caller).
        random_state: Random seed for reproducibility.
    """

    # Store whether cell-cycle scoring is enabled.
    enabled: bool = False

    # Store the layer to score on (must be log-normalized).
    score_layer: str = "cellquorum_normalized"

    # Store S-phase genes (empty by default; caller fills from Tirosh constants).
    s_genes: list[str] = Field(default_factory=list)

    # Store G2M-phase genes (empty by default; caller fills from Tirosh constants).
    g2m_genes: list[str] = Field(default_factory=list)

    # Store the random seed for deterministic scoring.
    random_state: int = 0

    @field_validator("s_genes", "g2m_genes", mode="before")
    @classmethod
    def validate_gene_lists(cls, value: object) -> list[str]:
        """
        Validate gene lists.

        Args:
            value: Candidate gene list.

        Returns:
            Cleaned gene list.

        Raises:
            ValueError: If the value is not a list of non-empty strings.
        """

        # Return an empty list when genes are omitted.
        if value is None:
            return []

        # Reject a single string because it is ambiguous.
        if isinstance(value, str):
            raise ValueError("Gene lists must be provided as lists, not strings.")

        # Reject non-list and non-tuple values.
        if not isinstance(value, list | tuple):
            raise ValueError(
                "Gene lists must be lists of strings. " f"Received: {type(value).__name__}."
            )

        # Initialize the cleaned gene list.
        cleaned_genes: list[str] = []

        # Iterate over gene values.
        for gene in value:
            # Reject non-string genes.
            if not isinstance(gene, str):
                raise ValueError("Gene names must be strings. " f"Received: {type(gene).__name__}.")

            # Strip harmless whitespace.
            cleaned_gene = gene.strip()

            # Reject empty genes.
            if not cleaned_gene:
                raise ValueError("Gene names cannot be empty.")

            # Store the cleaned gene.
            cleaned_genes.append(cleaned_gene)

        # Return the cleaned genes.
        return cleaned_genes

    @field_validator("random_state", mode="before")
    @classmethod
    def validate_random_state(cls, value: object) -> int:
        """
        Validate the random state seed.

        Args:
            value: Candidate random state.

        Returns:
            Validated random state.

        Raises:
            ValueError: If the value is not a non-negative integer.
        """

        # Reject booleans because bool is a subclass of int.
        if isinstance(value, bool):
            raise ValueError("random_state cannot be a boolean value.")

        # Reject non-integer values.
        if not isinstance(value, int):
            raise ValueError(
                "random_state must be an integer. " f"Received: {type(value).__name__}."
            )

        # Reject negative values.
        if value < 0:
            raise ValueError("random_state must be >= 0.")

        # Return the validated random state.
        return value


class QCAmbientRNAConfig(StrictBaseModel):
    """
    Store ambient RNA assessment settings.

    Ambient RNA correction requires extra inputs such as raw droplet matrices and
    often clustering. CellQuorum therefore defaults to audit mode and keeps actual
    correction disabled until those inputs exist.

    Args:
        enabled: Whether ambient RNA assessment is enabled.
        method: Ambient RNA method.
        correction_enabled: Whether ambient RNA correction should modify counts.
        contamination_fraction: Optional assumed contamination fraction.
        marker_genes: Optional marker genes used for contamination audits.
        require_raw_droplets_for_correction: Whether correction requires raw droplets.
    """

    # Store whether ambient RNA assessment is enabled.
    enabled: bool = True

    # Store the selected ambient RNA method.
    method: AmbientMethod = "audit"

    # Store whether ambient RNA correction should modify counts.
    correction_enabled: bool = False

    # Store an optional assumed contamination fraction.
    contamination_fraction: float | None = None

    # Store optional marker genes used for ambient RNA audits.
    marker_genes: list[str] = Field(default_factory=list)

    # Store whether correction requires raw droplet data.
    require_raw_droplets_for_correction: bool = True

    @field_validator("contamination_fraction", mode="before")
    @classmethod
    def validate_contamination_fraction(cls, value: object) -> float | None:
        """
        Validate optional contamination fraction.

        Args:
            value: Candidate contamination fraction.

        Returns:
            Validated contamination fraction or None.

        Raises:
            ValueError: If the value is outside [0, 1], boolean, or non-numeric.
        """

        # Preserve absent contamination fractions.
        if value is None:
            return None

        # Reject booleans because they are not valid fractions.
        if isinstance(value, bool):
            raise ValueError("contamination_fraction cannot be boolean.")

        # Reject non-numeric values.
        if not isinstance(value, int | float):
            raise ValueError(
                "contamination_fraction must be numeric. " f"Received: {type(value).__name__}."
            )

        # Convert the value to float.
        float_value = float(value)

        # Reject invalid fractions.
        if float_value < 0.0 or float_value > 1.0:
            raise ValueError("contamination_fraction must be between 0 and 1.")

        # Return the validated fraction.
        return float_value

    @field_validator("marker_genes", mode="before")
    @classmethod
    def validate_marker_genes(cls, value: object) -> list[str]:
        """
        Validate ambient RNA marker genes.

        Args:
            value: Candidate marker gene list.

        Returns:
            Cleaned marker gene list.

        Raises:
            ValueError: If the value is not a list of non-empty strings.
        """

        # Return an empty list when marker genes are omitted.
        if value is None:
            return []

        # Reject a single string because it is ambiguous.
        if isinstance(value, str):
            raise ValueError("marker_genes must be provided as a list, not a string.")

        # Reject non-list and non-tuple values.
        if not isinstance(value, list | tuple):
            raise ValueError(
                "marker_genes must be a list of strings. " f"Received: {type(value).__name__}."
            )

        # Initialize the cleaned marker gene list.
        cleaned_genes: list[str] = []

        # Iterate over marker gene values.
        for gene in value:
            # Reject non-string marker genes.
            if not isinstance(gene, str):
                raise ValueError(
                    "Ambient RNA marker genes must be strings. " f"Received: {type(gene).__name__}."
                )

            # Strip harmless whitespace.
            cleaned_gene = gene.strip()

            # Reject empty marker genes.
            if not cleaned_gene:
                raise ValueError("Ambient RNA marker genes cannot be empty.")

            # Store the cleaned marker gene.
            cleaned_genes.append(cleaned_gene)

        # Return the cleaned marker genes.
        return cleaned_genes

    @model_validator(mode="after")
    def validate_ambient_consistency(self) -> QCAmbientRNAConfig:
        """
        Validate consistency between ambient RNA flags and method.

        Returns:
            Validated ambient RNA configuration.

        Raises:
            ValueError: If enabled or correction flags conflict with method none.
        """

        # Reject enabled ambient RNA assessment without a real method.
        if self.enabled and self.method == "none":
            raise ValueError("Ambient RNA assessment cannot be enabled when method is 'none'.")

        # Reject correction without a correction-capable method.
        if self.correction_enabled and self.method not in {"soupx", "decontx"}:
            raise ValueError(
                "Ambient RNA correction requires method 'soupx' or 'decontx'. "
                "Use method='audit' only for non-mutating assessment."
            )

        # Return the validated model.
        return self


class QCDuplicateNameConfig(StrictBaseModel):
    """
    Store duplicate observation and variable name policies.

    Duplicate feature names are common in raw inputs and can cause downstream
    ambiguity. The best-practices workflow calls `var_names_make_unique()` after
    loading data. CellQuorum makes this policy explicit.

    Args:
        var_names: Policy for duplicate AnnData variable names.
        obs_names: Policy for duplicate AnnData observation names.
    """

    # Store duplicate variable-name handling policy.
    var_names: DuplicateNamePolicy = "make_unique"

    # Store duplicate observation-name handling policy.
    obs_names: DuplicateNamePolicy = "warn"


class QCOutputConfig(StrictBaseModel):
    """
    Store QC output settings.

    QC should emit machine-readable tables and audit-friendly summaries. Figures
    are enabled by default because QC requires visual inspection, but tables and
    JSON summaries remain the primary contract.

    Args:
        write_metrics_table: Whether to write cell and gene QC metric tables.
        write_filter_table: Whether to write filtering decision tables.
        write_threshold_table: Whether to write threshold tables.
        write_summary_json: Whether to write a JSON QC summary.
        write_h5ad: Whether to write a QC AnnData object.
        write_figures: Whether to write QC figures.
        figure_format: File format used for QC figures.
    """

    # Store whether QC metric tables should be written.
    write_metrics_table: bool = True

    # Store whether filtering decision tables should be written.
    write_filter_table: bool = True

    # Store whether threshold tables should be written.
    write_threshold_table: bool = True

    # Store whether QC summary JSON should be written.
    write_summary_json: bool = True

    # Store whether a QC AnnData object should be written.
    write_h5ad: bool = True

    # Store whether QC figures should be written.
    write_figures: bool = True

    # Store the QC figure format.
    figure_format: QCFigureFormat = "png"

    # Store the QC figure DPI resolution.
    figure_dpi: int = 300


class QCConfig(StrictBaseModel):
    """
    Store full QC module configuration.

    This model controls the first real CellQuorum analysis module. It separates
    reporting from filtering, keeps default filtering permissive, and makes
    doublet and ambient RNA behavior explicit rather than hidden.

    Args:
        enabled: Whether the QC module is enabled.
        mode: QC behavior, one of report_only, filter, or both.
        threshold_strategy: Which threshold family should be used.
        metrics: QC metric calculation settings.
        basic: Fixed QC threshold settings.
        mad: Adaptive MAD QC settings.
        features: Feature family pattern settings.
        doublets: Doublet detection settings.
        cell_cycle: Cell-cycle scoring settings.
        ambient: Ambient RNA assessment settings.
        duplicate_names: Duplicate name handling settings.
        outputs: QC output settings.
        fail_on_empty_result: Whether filtering to zero cells or genes is fatal.
    """

    # Store whether QC is enabled.
    enabled: bool = True

    # Store whether QC should report only, filter only, or both.
    mode: QCMode = "report_only"

    # Store which threshold strategy should be applied.
    threshold_strategy: ThresholdStrategy = "fixed_and_mad"

    # Store metric calculation settings.
    metrics: QCMetricCalculationConfig = Field(default_factory=QCMetricCalculationConfig)

    # Store fixed QC threshold settings.
    basic: QCBasicThresholdConfig = Field(default_factory=QCBasicThresholdConfig)

    # Store adaptive MAD threshold settings.
    mad: QCMadThresholdConfig = Field(default_factory=QCMadThresholdConfig)

    # Store feature pattern settings.
    features: QCFeaturePatternConfig = Field(default_factory=QCFeaturePatternConfig)

    # Store doublet detection settings.
    doublets: QCDoubletConfig = Field(default_factory=QCDoubletConfig)

    # Store cell-cycle scoring settings.
    cell_cycle: QCCellCycleConfig = Field(default_factory=QCCellCycleConfig)

    # Store ambient RNA assessment settings.
    ambient: QCAmbientRNAConfig = Field(default_factory=QCAmbientRNAConfig)

    # Store duplicate name policy settings.
    duplicate_names: QCDuplicateNameConfig = Field(default_factory=QCDuplicateNameConfig)

    # Store output settings.
    outputs: QCOutputConfig = Field(default_factory=QCOutputConfig)

    # Store whether empty filtered results should fail.
    fail_on_empty_result: bool = True

    @model_validator(mode="after")
    def validate_strategy_consistency(self) -> QCConfig:
        """
        Validate consistency among QC strategy settings.

        Returns:
            Validated QC configuration.

        Raises:
            ValueError: If threshold strategy settings are inconsistent.
        """

        # Reject MAD strategies when MAD thresholding is disabled.
        if self.threshold_strategy in {"mad", "fixed_and_mad"} and not self.mad.enabled:
            raise ValueError(
                "MAD threshold strategy requires mad.enabled=true. "
                "Use threshold_strategy='fixed' or enable MAD thresholds."
            )

        # Return the validated model.
        return self

    def should_filter(self) -> bool:
        """
        Return whether QC should apply filtering.

        Returns:
            True when mode is filter or both.
        """

        # Return whether the QC mode includes filtering.
        return self.mode in {"filter", "both"}

    def should_report(self) -> bool:
        """
        Return whether QC should produce reports and metrics.

        Returns:
            True when mode is report_only or both.
        """

        # Return whether the QC mode includes reporting.
        return self.mode in {"report_only", "both"}

    def enabled_metric_families(self) -> list[str]:
        """
        Return enabled QC metric families.

        Returns:
            Ordered list of enabled QC metric family labels.
        """

        # Initialize the enabled metric family list.
        families = ["basic"]

        # Add MAD metrics when MAD thresholding is enabled.
        if self.mad.enabled:
            families.append("mad")

        # Add doublet metrics when doublet auditing is enabled.
        if self.doublets.enabled:
            families.append("doublets")

        # Add ambient RNA metrics when ambient RNA assessment is enabled.
        if self.ambient.enabled:
            families.append("ambient_rna")

        # Return the enabled metric families.
        return families


def validate_qc_config_dict(config: Mapping[str, object]) -> QCConfig:
    """
    Validate a plain mapping into a QCConfig object.

    Args:
        config: Mapping containing QC configuration values.

    Returns:
        Validated QCConfig object.

    Raises:
        ConfigValidationError: If the input is not a mapping, contains unknown
            top-level keys, or fails Pydantic validation.
    """

    # Validate and copy the input mapping.
    config_dict = require_mapping(config, field_path="qc")

    # Reject unknown top-level QC keys before Pydantic validation.
    reject_unknown_keys(
        config_dict,
        allowed_keys=[
            "enabled",
            "mode",
            "threshold_strategy",
            "metrics",
            "basic",
            "mad",
            "features",
            "doublets",
            "cell_cycle",
            "ambient",
            "duplicate_names",
            "outputs",
            "fail_on_empty_result",
        ],
        field_path="qc",
    )

    # Try validating the dictionary through Pydantic.
    try:
        # Return the validated QC configuration.
        return QCConfig.model_validate(config_dict)

    # Convert Pydantic errors into CellQuorum config validation errors.
    except ValidationError as error:
        raise ConfigValidationError(f"Invalid QC configuration:\n{error}") from error


__all__ = [
    "AmbientMethod",
    "DoubletMethod",
    "DuplicateNamePolicy",
    "QCAmbientRNAConfig",
    "QCBasicThresholdConfig",
    "QCCellCycleConfig",
    "QCConfig",
    "QCDoubletConfig",
    "QCDuplicateNameConfig",
    "QCFigureFormat",
    "QCFeaturePatternConfig",
    "QCMadThresholdConfig",
    "QCMetricCalculationConfig",
    "QCMode",
    "QCOutputConfig",
    "ThresholdStrategy",
    "validate_qc_config_dict",
]
