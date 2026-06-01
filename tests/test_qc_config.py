"""Tests for CellQuorum QC configuration models."""

from __future__ import annotations

# Import pytest for exception assertions.
import pytest

# Import Pydantic validation errors for direct model validation checks.
from pydantic import ValidationError

# Import reusable CellQuorum config validation error.
from cellquorum.config.validation import ConfigValidationError

# Import QC configuration models under test.
from cellquorum.qc.config import (
    QCAmbientRNAConfig,
    QCBasicThresholdConfig,
    QCConfig,
    QCDoubletConfig,
    QCFeaturePatternConfig,
    QCMadThresholdConfig,
    QCMetricCalculationConfig,
    QCOutputConfig,
    validate_qc_config_dict,
)


def test_qc_config_defaults_follow_single_cell_best_practices() -> None:
    """
    Verify that default QC settings reflect the intended conservative workflow.

    The default QC configuration should calculate standard Scanpy-compatible
    metrics, use percent_top=[20], use permissive 5-MAD filtering for general
    outliers, handle mitochondrial filtering separately with 3 MADs plus a hard
    cutoff, and keep doublet/ambient behavior in audit mode rather than removing
    or correcting data automatically.
    """

    # Build the default QC configuration.
    config = QCConfig()

    # Confirm QC is enabled by default.
    assert config.enabled is True

    # Confirm default QC mode is report-only.
    assert config.mode == "report_only"

    # Confirm default thresholding uses fixed and MAD logic.
    assert config.threshold_strategy == "fixed_and_mad"

    # Confirm percent-top metric calculation follows the best-practices example.
    assert config.metrics.percent_top == [20]

    # Confirm log1p QC metrics are enabled by default.
    assert config.metrics.log1p is True

    # Confirm no AnnData layer is forced by default.
    assert config.metrics.layer is None

    # Confirm AnnData.raw is not used by default.
    assert config.metrics.use_raw is False

    # Confirm default general MAD threshold is permissive.
    assert config.mad.n_mads == 5.0

    # Confirm default MAD metrics match the intended QC covariates.
    assert config.mad.metrics == [
        "log1p_total_counts",
        "log1p_n_genes_by_counts",
        "pct_counts_in_top_20_genes",
    ]

    # Confirm mitochondrial MAD filtering is configured separately.
    assert config.mad.mito_metric == "pct_counts_mito"
    assert config.mad.mito_n_mads == 3.0

    # Confirm the hard mitochondrial cutoff is conservative and configurable.
    assert config.basic.max_mito_percent == 8.0

    # Confirm human mitochondrial prefixes are used by default.
    assert config.features.mitochondrial_prefixes == ["MT-"]

    # Confirm ribosomal prefixes are available by default.
    assert config.features.ribosomal_prefixes == ["RPS", "RPL"]

    # Confirm hemoglobin matching uses a regex-style pattern.
    assert config.features.hemoglobin_regexes == [r"^HB[ABDEGMQZ]\d*(?!\w)"]

    # Confirm scDblFinder is the default doublet method.
    assert config.doublets.method == "scdblfinder"

    # Confirm doublet removal is not automatic.
    assert config.doublets.remove is False

    # Confirm ambient RNA starts in audit mode.
    assert config.ambient.method == "audit"

    # Confirm ambient RNA correction is not automatic.
    assert config.ambient.correction_enabled is False

    # Confirm duplicate variable names are made unique by default.
    assert config.duplicate_names.var_names == "make_unique"

    # Confirm duplicate observation names warn by default.
    assert config.duplicate_names.obs_names == "warn"


def test_qc_config_mode_helpers_report_filter_and_both() -> None:
    """
    Verify that QC mode helper methods expose filtering and reporting behavior.

    Later QC stage code should not reimplement mode checks. These helpers define
    whether a config should write metrics, apply filters, or do both.
    """

    # Build a report-only configuration.
    report_only = QCConfig(mode="report_only")

    # Confirm report-only mode reports but does not filter.
    assert report_only.should_report() is True
    assert report_only.should_filter() is False

    # Build a filter-only configuration.
    filter_only = QCConfig(mode="filter")

    # Confirm filter-only mode filters but does not report.
    assert filter_only.should_report() is False
    assert filter_only.should_filter() is True

    # Build a configuration that reports and filters.
    both = QCConfig(mode="both")

    # Confirm both mode reports and filters.
    assert both.should_report() is True
    assert both.should_filter() is True


def test_qc_config_enabled_metric_families_reflect_enabled_submodules() -> None:
    """
    Verify that enabled metric-family reporting follows submodule flags.

    The enabled metric-family helper will later support reports, provenance, and
    stage summaries. It should reflect MAD, doublet, and ambient RNA settings.
    """

    # Build the default QC configuration.
    default_config = QCConfig()

    # Confirm all default metric families are enabled.
    assert default_config.enabled_metric_families() == [
        "basic",
        "mad",
        "doublets",
        "ambient_rna",
    ]

    # Build a configuration with optional audit families disabled.
    reduced_config = QCConfig(
        threshold_strategy="fixed",
        mad={"enabled": False},
        doublets={"enabled": False, "method": "none"},
        ambient={"enabled": False, "method": "none"},
    )

    # Confirm only basic metrics remain enabled.
    assert reduced_config.enabled_metric_families() == ["basic"]


def test_metric_calculation_config_cleans_percent_top_values() -> None:
    """
    Verify that percent_top values are sorted and deduplicated.

    Deterministic percent_top values are important because downstream metric names
    such as pct_counts_in_top_20_genes are derived from these settings.
    """

    # Build a metric config with duplicate and unsorted percent_top values.
    config = QCMetricCalculationConfig(percent_top=[50, 20, 20, 100])

    # Confirm percent_top values are sorted and deduplicated.
    assert config.percent_top == [20, 50, 100]


@pytest.mark.parametrize(
    "bad_percent_top",
    [
        None,
        [],
        "20",
        [0],
        [-1],
        [True],
        [20.5],
    ],
)
def test_metric_calculation_config_rejects_invalid_percent_top(
    bad_percent_top: object,
) -> None:
    """
    Verify that invalid percent_top settings fail validation.

    Percent-top metrics require positive integer cutoffs. Invalid values should
    fail at configuration time rather than producing malformed metric names later.
    """

    # Confirm invalid percent_top values fail Pydantic validation.
    with pytest.raises(ValidationError, match="percent_top"):
        QCMetricCalculationConfig(percent_top=bad_percent_top)


def test_metric_calculation_config_cleans_layer_name() -> None:
    """
    Verify that optional AnnData layer names are stripped.

    Layer names may come from YAML or user input with harmless whitespace. The
    config should normalize those names before QC metric calculation uses them.
    """

    # Build a metric configuration with a padded layer name.
    config = QCMetricCalculationConfig(layer=" counts ")

    # Confirm the layer name was stripped.
    assert config.layer == "counts"


def test_metric_calculation_config_rejects_empty_layer_name() -> None:
    """
    Verify that empty AnnData layer names fail validation.

    An empty layer name would later cause confusing AnnData lookup errors, so it
    should be rejected during configuration validation.
    """

    # Confirm an empty layer name fails validation.
    with pytest.raises(ValidationError, match="layer"):
        QCMetricCalculationConfig(layer="   ")


def test_feature_pattern_config_cleans_lists_and_supports_mouse_mito_prefix() -> None:
    """
    Verify that feature-pattern settings are configurable and cleaned.

    Human datasets commonly use MT- mitochondrial prefixes, while mouse datasets
    often use mt-. The config must support both without hard-coding the species.
    """

    # Build a feature pattern config with padded custom values.
    config = QCFeaturePatternConfig(
        mitochondrial_prefixes=[" mt- "],
        ribosomal_prefixes=[" Rps ", " Rpl "],
        hemoglobin_regexes=[r" ^Hb[ab] "],
        custom_exclude_prefixes=[" MALAT1 "],
    )

    # Confirm mitochondrial prefixes were cleaned.
    assert config.mitochondrial_prefixes == ["mt-"]

    # Confirm ribosomal prefixes were cleaned.
    assert config.ribosomal_prefixes == ["Rps", "Rpl"]

    # Confirm hemoglobin regexes were cleaned.
    assert config.hemoglobin_regexes == [r"^Hb[ab]"]

    # Confirm custom exclude prefixes were cleaned.
    assert config.custom_exclude_prefixes == ["MALAT1"]


@pytest.mark.parametrize(
    "field_name",
    [
        "mitochondrial_prefixes",
        "ribosomal_prefixes",
        "hemoglobin_regexes",
        "custom_exclude_prefixes",
    ],
)
def test_feature_pattern_config_rejects_single_string_patterns(field_name: str) -> None:
    """
    Verify that feature-pattern fields reject single strings.

    A single string is ambiguous because strings are iterable. Requiring explicit
    lists prevents accidental character-wise interpretation later.
    """

    # Confirm a single string is rejected for each pattern-list field.
    with pytest.raises(ValidationError, match=field_name):
        QCFeaturePatternConfig.model_validate({field_name: "MT-"})


def test_basic_threshold_config_accepts_valid_fixed_thresholds() -> None:
    """
    Verify that valid fixed QC thresholds can be configured.

    Fixed thresholds remain useful for transparent, reproducible QC even though
    CellQuorum defaults to conservative report-only behavior.
    """

    # Build a fixed-threshold config with valid min/max settings.
    config = QCBasicThresholdConfig(
        min_genes_per_cell=100,
        max_genes_per_cell=6000,
        min_counts_per_cell=500,
        max_counts_per_cell=50000,
        min_cells_per_gene=20,
        max_mito_percent=12.5,
        max_ribo_percent=80.0,
        max_hemoglobin_percent=10.0,
    )

    # Confirm integer thresholds were retained.
    assert config.min_genes_per_cell == 100
    assert config.max_genes_per_cell == 6000
    assert config.min_counts_per_cell == 500
    assert config.max_counts_per_cell == 50000
    assert config.min_cells_per_gene == 20

    # Confirm percentage thresholds were converted to floats.
    assert config.max_mito_percent == 12.5
    assert config.max_ribo_percent == 80.0
    assert config.max_hemoglobin_percent == 10.0


@pytest.mark.parametrize(
    "field_name",
    [
        "min_genes_per_cell",
        "max_genes_per_cell",
        "min_counts_per_cell",
        "max_counts_per_cell",
        "min_cells_per_gene",
    ],
)
def test_basic_threshold_config_rejects_invalid_integer_thresholds(field_name: str) -> None:
    """
    Verify that integer QC thresholds reject invalid values.

    Integer thresholds should not accept booleans, negative values, or fractional
    values because those would produce confusing filtering rules.
    """

    # Confirm boolean values are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCBasicThresholdConfig.model_validate({field_name: True})

    # Confirm negative values are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCBasicThresholdConfig.model_validate({field_name: -1})

    # Confirm floating-point values are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCBasicThresholdConfig.model_validate({field_name: 1.5})


@pytest.mark.parametrize(
    "field_name",
    [
        "max_mito_percent",
        "max_ribo_percent",
        "max_hemoglobin_percent",
    ],
)
def test_basic_threshold_config_rejects_invalid_percent_thresholds(field_name: str) -> None:
    """
    Verify that percentage QC thresholds reject invalid values.

    Percent thresholds must stay within [0, 100] and should not accept booleans
    or string-like values.
    """

    # Confirm boolean values are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCBasicThresholdConfig.model_validate({field_name: True})

    # Confirm values below zero are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCBasicThresholdConfig.model_validate({field_name: -0.1})

    # Confirm values above one hundred are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCBasicThresholdConfig.model_validate({field_name: 100.1})

    # Confirm strings are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCBasicThresholdConfig.model_validate({field_name: "8"})


def test_basic_threshold_config_rejects_inconsistent_gene_threshold_pair() -> None:
    """
    Verify that minimum gene thresholds cannot exceed maximum gene thresholds.

    This prevents impossible fixed filtering rules from reaching the QC decision
    engine.
    """

    # Confirm inconsistent gene thresholds fail validation.
    with pytest.raises(ValidationError, match="min_genes_per_cell"):
        QCBasicThresholdConfig(min_genes_per_cell=5000, max_genes_per_cell=1000)


def test_basic_threshold_config_rejects_inconsistent_count_threshold_pair() -> None:
    """
    Verify that minimum count thresholds cannot exceed maximum count thresholds.

    This prevents impossible total-count filtering rules from reaching the QC
    decision engine.
    """

    # Confirm inconsistent count thresholds fail validation.
    with pytest.raises(ValidationError, match="min_counts_per_cell"):
        QCBasicThresholdConfig(min_counts_per_cell=50000, max_counts_per_cell=1000)


def test_mad_threshold_config_defaults_match_best_practices() -> None:
    """
    Verify that MAD threshold defaults match the intended QC strategy.

    General outlier filtering should use 5 MADs on log-count, log-gene, and
    percent-top metrics, while mitochondrial filtering has a separate 3-MAD rule.
    """

    # Build the default MAD threshold configuration.
    config = QCMadThresholdConfig()

    # Confirm MAD thresholding is enabled.
    assert config.enabled is True

    # Confirm the general MAD multiplier is permissive.
    assert config.n_mads == 5.0

    # Confirm default general MAD metrics.
    assert config.metrics == [
        "log1p_total_counts",
        "log1p_n_genes_by_counts",
        "pct_counts_in_top_20_genes",
    ]

    # Confirm mitochondrial MAD settings are separate.
    assert config.mito_metric == "pct_counts_mito"
    assert config.mito_n_mads == 3.0

    # Confirm zero-MAD behavior is explicit.
    assert config.skip_zero_mad is True


@pytest.mark.parametrize("field_name", ["n_mads", "mito_n_mads"])
def test_mad_threshold_config_rejects_invalid_mad_multipliers(field_name: str) -> None:
    """
    Verify that MAD multipliers must be positive numeric values.

    MAD multipliers of zero, negative values, booleans, or strings would make
    adaptive QC thresholding invalid or ambiguous.
    """

    # Confirm boolean values are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCMadThresholdConfig.model_validate({field_name: True})

    # Confirm zero is rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCMadThresholdConfig.model_validate({field_name: 0})

    # Confirm negative values are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCMadThresholdConfig.model_validate({field_name: -1})

    # Confirm strings are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCMadThresholdConfig.model_validate({field_name: "5"})


def test_mad_threshold_config_cleans_groupby_and_metric_lists() -> None:
    """
    Verify that MAD metric and groupby lists are cleaned.

    These fields will later become DataFrame column lookups, so harmless
    whitespace should be stripped before threshold construction.
    """

    # Build a MAD config with padded metric and groupby values.
    config = QCMadThresholdConfig(
        metrics=[" total_counts ", " n_genes_by_counts "],
        groupby=[" sample_id ", " batch "],
    )

    # Confirm metrics were cleaned.
    assert config.metrics == ["total_counts", "n_genes_by_counts"]

    # Confirm groupby fields were cleaned.
    assert config.groupby == ["sample_id", "batch"]


def test_mad_threshold_config_rejects_empty_mito_metric() -> None:
    """
    Verify that the mitochondrial metric name cannot be empty.

    Threshold construction needs a real metric column name for mitochondrial
    outlier detection.
    """

    # Confirm an empty mitochondrial metric name fails validation.
    with pytest.raises(ValidationError, match="mito_metric"):
        QCMadThresholdConfig(mito_metric="   ")


def test_qc_config_rejects_mad_strategy_when_mad_disabled() -> None:
    """
    Verify that MAD threshold strategies require MAD settings to be enabled.

    This prevents a configuration from requesting MAD-based filtering while
    simultaneously disabling the MAD threshold module.
    """

    # Confirm mad-only strategy fails when MAD is disabled.
    with pytest.raises(ValidationError, match="MAD threshold strategy"):
        QCConfig(threshold_strategy="mad", mad={"enabled": False})

    # Confirm fixed-and-MAD strategy fails when MAD is disabled.
    with pytest.raises(ValidationError, match="MAD threshold strategy"):
        QCConfig(threshold_strategy="fixed_and_mad", mad={"enabled": False})

    # Confirm fixed-only strategy is valid when MAD is disabled.
    config = QCConfig(threshold_strategy="fixed", mad={"enabled": False})

    # Confirm the fixed-only configuration was accepted.
    assert config.threshold_strategy == "fixed"
    assert config.mad.enabled is False


def test_doublet_config_defaults_to_audit_without_removal() -> None:
    """
    Verify that doublet settings default to inspect-first behavior.

    The QC module should support doublet detection, but automatic doublet removal
    should be explicitly requested rather than silently applied.
    """

    # Build the default doublet configuration.
    config = QCDoubletConfig()

    # Confirm doublet auditing is enabled.
    assert config.enabled is True

    # Confirm the default method is scDblFinder.
    assert config.method == "scdblfinder"

    # Confirm doublet removal is disabled by default.
    assert config.remove is False

    # Confirm the expected doublet rate is a probability.
    assert config.expected_doublet_rate == 0.06

    # Confirm per-sample detection is enabled.
    assert config.per_sample is True


@pytest.mark.parametrize("field_name", ["expected_doublet_rate", "score_threshold"])
def test_doublet_config_rejects_invalid_probability_fields(field_name: str) -> None:
    """
    Verify that doublet probability settings reject invalid values.

    Doublet rates and score thresholds must be probabilities between 0 and 1.
    """

    # Confirm booleans are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCDoubletConfig.model_validate({field_name: True})

    # Confirm values below zero are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCDoubletConfig.model_validate({field_name: -0.1})

    # Confirm values above one are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCDoubletConfig.model_validate({field_name: 1.1})

    # Confirm strings are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCDoubletConfig.model_validate({field_name: "0.1"})


def test_doublet_config_rejects_enabled_none_method() -> None:
    """
    Verify that enabled doublet detection requires a real method.

    A configuration should not claim doublet detection is enabled while selecting
    method='none'.
    """

    # Confirm enabled doublet detection with method none fails.
    with pytest.raises(ValidationError, match="method is 'none'"):
        QCDoubletConfig(enabled=True, method="none")


def test_doublet_config_rejects_removal_with_none_method() -> None:
    """
    Verify that doublet removal requires a real doublet method.

    Removing doublets without a detection method would make the filtering rule
    undefined.
    """

    # Confirm doublet removal with method none fails.
    with pytest.raises(ValidationError, match="method is 'none'"):
        QCDoubletConfig(enabled=False, remove=True, method="none")


def test_ambient_config_defaults_to_audit_without_correction() -> None:
    """
    Verify that ambient RNA defaults to non-mutating audit behavior.

    Actual ambient RNA correction requires extra data such as raw droplet
    matrices, so the first QC module should not mutate counts by default.
    """

    # Build the default ambient RNA configuration.
    config = QCAmbientRNAConfig()

    # Confirm ambient RNA assessment is enabled.
    assert config.enabled is True

    # Confirm the default method is audit.
    assert config.method == "audit"

    # Confirm correction is disabled by default.
    assert config.correction_enabled is False

    # Confirm raw droplet data is required for correction.
    assert config.require_raw_droplets_for_correction is True


def test_ambient_config_allows_correction_capable_methods() -> None:
    """
    Verify that correction-capable ambient RNA methods can enable correction.

    SoupX and DecontX are correction-capable methods. The config should allow
    them while still making correction explicit.
    """

    # Build a SoupX correction configuration.
    soupx_config = QCAmbientRNAConfig(method="soupx", correction_enabled=True)

    # Confirm SoupX correction is accepted.
    assert soupx_config.correction_enabled is True

    # Build a DecontX correction configuration.
    decontx_config = QCAmbientRNAConfig(method="decontx", correction_enabled=True)

    # Confirm DecontX correction is accepted.
    assert decontx_config.correction_enabled is True


def test_ambient_config_rejects_correction_with_audit_method() -> None:
    """
    Verify that audit mode cannot modify counts.

    Audit mode should inspect possible ambient RNA contamination without changing
    the count matrix.
    """

    # Confirm audit mode with correction enabled fails validation.
    with pytest.raises(ValidationError, match="Ambient RNA correction requires"):
        QCAmbientRNAConfig(method="audit", correction_enabled=True)


def test_ambient_config_rejects_enabled_none_method() -> None:
    """
    Verify that enabled ambient RNA assessment requires a real method.

    A configuration should not claim ambient RNA assessment is enabled while
    selecting method='none'.
    """

    # Confirm enabled ambient assessment with method none fails.
    with pytest.raises(ValidationError, match="method is 'none'"):
        QCAmbientRNAConfig(enabled=True, method="none")


def test_ambient_config_rejects_invalid_contamination_fraction() -> None:
    """
    Verify that ambient contamination fractions must be valid probabilities.

    Assumed contamination fractions outside [0, 1] are invalid and should fail
    during configuration validation.
    """

    # Confirm booleans are rejected.
    with pytest.raises(ValidationError, match="contamination_fraction"):
        QCAmbientRNAConfig(contamination_fraction=True)

    # Confirm values below zero are rejected.
    with pytest.raises(ValidationError, match="contamination_fraction"):
        QCAmbientRNAConfig(contamination_fraction=-0.1)

    # Confirm values above one are rejected.
    with pytest.raises(ValidationError, match="contamination_fraction"):
        QCAmbientRNAConfig(contamination_fraction=1.1)

    # Confirm strings are rejected.
    with pytest.raises(ValidationError, match="contamination_fraction"):
        QCAmbientRNAConfig(contamination_fraction="0.1")


def test_ambient_config_cleans_marker_genes() -> None:
    """
    Verify that ambient RNA marker genes are cleaned.

    Marker genes may be supplied from YAML or notebooks with harmless whitespace.
    The config should normalize those names before downstream audit logic uses
    them.
    """

    # Build an ambient RNA config with padded marker genes.
    config = QCAmbientRNAConfig(marker_genes=[" MALAT1 ", " HBB "])

    # Confirm marker genes were stripped.
    assert config.marker_genes == ["MALAT1", "HBB"]


def test_output_config_rejects_invalid_figure_format() -> None:
    """
    Verify that QC output figure formats are constrained.

    Figure output should be limited to known formats so artifact paths and report
    rendering stay predictable.
    """

    # Confirm unsupported figure formats fail validation.
    with pytest.raises(ValidationError, match="figure_format"):
        QCOutputConfig(figure_format="jpg")


def test_validate_qc_config_dict_accepts_valid_mapping() -> None:
    """
    Verify that dictionary-based QC config validation works.

    This helper supports future YAML sections, Hydra groups, notebooks, and
    plugin code without requiring callers to instantiate Pydantic models directly.
    """

    # Validate a representative QC configuration dictionary.
    config = validate_qc_config_dict(
        {
            "mode": "both",
            "threshold_strategy": "fixed",
            "mad": {
                "enabled": False,
            },
            "metrics": {
                "percent_top": [50, 20, 20],
            },
            "doublets": {
                "enabled": False,
                "method": "none",
            },
            "ambient": {
                "enabled": False,
                "method": "none",
            },
        }
    )

    # Confirm the helper returned a QCConfig object.
    assert isinstance(config, QCConfig)

    # Confirm mode was parsed.
    assert config.mode == "both"

    # Confirm percent_top values were cleaned.
    assert config.metrics.percent_top == [20, 50]

    # Confirm MAD thresholding was disabled.
    assert config.mad.enabled is False


def test_validate_qc_config_dict_rejects_non_mapping_input() -> None:
    """
    Verify that dictionary validation rejects non-mapping inputs.

    Config validation helpers should fail clearly when callers pass the wrong
    input shape.
    """

    # Confirm non-mapping inputs fail validation.
    with pytest.raises(ConfigValidationError, match="must be a mapping"):
        validate_qc_config_dict(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_validate_qc_config_dict_rejects_unknown_top_level_keys() -> None:
    """
    Verify that dictionary validation rejects unknown top-level QC keys.

    Unknown keys often indicate YAML typos. They should fail early instead of
    silently producing ignored configuration.
    """

    # Confirm unknown top-level QC keys fail validation.
    with pytest.raises(ConfigValidationError, match="unsupported key"):
        validate_qc_config_dict({"not_a_real_qc_key": True})


def test_validate_qc_config_dict_wraps_pydantic_errors() -> None:
    """
    Verify that Pydantic validation errors are wrapped as ConfigValidationError.

    Callers using the dictionary helper should only need to catch the shared
    CellQuorum config validation error type.
    """

    # Confirm invalid nested settings are wrapped in ConfigValidationError.
    with pytest.raises(ConfigValidationError, match="Invalid QC configuration"):
        validate_qc_config_dict({"mode": "invalid_mode"})
