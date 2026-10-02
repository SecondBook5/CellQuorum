"""Tests for CellQuorum QC configuration models."""

from __future__ import annotations

# Import pytest for exception assertions.
import pytest

# Import Pydantic validation errors for direct model validation checks.
from pydantic import ValidationError

# Import reusable CellQuorum config validation error.
from cellquorum.config.validation import ConfigValidationError

# Import QC configuration models under test.
from cellquorum.stages.qc.config import (
    QCAmbientRNAConfig,
    QCConfig,
    QCDoubletConfig,
    QCFeaturePatternConfig,
    QCFloorConfig,
    QCMetricCalculationConfig,
    QCOutputConfig,
    validate_qc_config_dict,
)


def test_qc_config_defaults_follow_single_cell_best_practices() -> None:
    """
    Verify that default QC settings reflect the intended conservative workflow.

    The default QC configuration should calculate standard Scanpy-compatible metrics, use
    percent_top=[20], apply only absolute detection floors, and keep doublet/ambient behavior
    in audit mode rather than removing or correcting data automatically.
    """

    # Build the default QC configuration.
    config = QCConfig()

    # Confirm QC is enabled by default.
    assert config.enabled is True

    # Confirm percent-top metric calculation follows the best-practices example.
    assert config.metrics.percent_top == [20]

    # Confirm log1p QC metrics are enabled by default.
    assert config.metrics.log1p is True

    # Confirm no AnnData layer is forced by default.
    assert config.metrics.layer is None

    # Confirm AnnData.raw is not used by default.
    assert config.metrics.use_raw is False

    # Confirm the default floors are detection limits, and that only three exist. The gene
    # floor is on because a barcode below it has no population to be unusual against; the count
    # floor is off because total counts vary by an order of magnitude across real cell types.
    assert config.floors.min_genes_per_cell == 200
    assert config.floors.min_counts_per_cell is None
    assert config.floors.min_cells_per_gene == 3
    assert set(type(config.floors).model_fields) == {
        "min_genes_per_cell",
        "min_counts_per_cell",
        "min_cells_per_gene",
    }

    # Confirm no cohort-relative bound is configured anywhere on the model. Graded severity
    # owns every such judgement now, so a fixed ceiling reappearing here is a regression.
    assert not hasattr(config, "mode")
    assert not hasattr(config, "threshold_strategy")
    assert not hasattr(config, "mad")
    assert not hasattr(config, "basic")

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


@pytest.mark.parametrize(
    ("removed_key", "value", "expected_guidance"),
    [
        ("mode", "flag_no_drop", "floors"),
        ("mode", "filter", "floors"),
        ("mode", "report_only", "floors"),
        ("threshold_strategy", "fixed", "graded"),
        ("mad", {"enabled": True}, "lineage-conditional"),
        ("basic", {"max_mito_percent": 8.0}, "keratinocytes"),
    ],
)
def test_every_removed_v1_key_fails_with_migration_guidance(
    removed_key: str, value: object, expected_guidance: str
) -> None:
    """A config still setting a v1 threshold key must fail, and say what replaces it.

    Silently ignoring these would be the worse failure. A config that says
    ``max_mito_percent: 8.0`` was written expecting a hard ceiling to be applied; accepting it
    and doing nothing would change that run's results without telling anyone. So each removed
    key raises and names its replacement.
    """

    from cellquorum.core.exceptions import CellQuorumConfigError

    with pytest.raises(CellQuorumConfigError, match=expected_guidance):
        QCConfig(**{removed_key: value})


def test_running_without_removing_anything_is_expressed_as_null_floors() -> None:
    """The capability the deleted ``mode: flag_no_drop`` provided still exists.

    It is now stated per floor rather than as a global switch, which says *which* limit was
    lifted. Graded adjudication never deletes a cell in any configuration, so with every floor
    off nothing can leave the object.
    """

    config = QCConfig(
        floors={
            "min_genes_per_cell": None,
            "min_counts_per_cell": None,
            "min_cells_per_gene": None,
        }
    )

    assert config.floors.min_genes_per_cell is None
    assert config.floors.min_counts_per_cell is None
    assert config.floors.min_cells_per_gene is None


def test_qc_config_enabled_metric_families_reflect_enabled_submodules() -> None:
    """
    Verify that enabled metric-family reporting follows submodule flags.

    The enabled metric-family helper will later support reports, provenance, and
    stage summaries. It should reflect doublet and ambient RNA settings.
    """

    # Build the default QC configuration.
    default_config = QCConfig()

    # Confirm all default metric families are enabled.
    assert default_config.enabled_metric_families() == [
        "floors",
        "doublets",
        "ambient_rna",
    ]

    # Build a configuration with optional audit families disabled.
    reduced_config = QCConfig(
        doublets={"enabled": False, "method": "none"},
        ambient={"enabled": False, "method": "none"},
    )

    # Confirm only basic metrics remain enabled.
    assert reduced_config.enabled_metric_families() == ["floors"]


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


def test_floor_config_accepts_the_three_surviving_floors() -> None:
    """The floors are absolute detection limits, not cohort-relative bounds.

    Only three survive the threshold path. Each says where the assay stops resolving anything,
    which is why none of them is a quantile: "fewer than 100 genes detected" is a statement
    about the instrument, not about how this cohort happens to be distributed.
    """

    config = QCFloorConfig(
        min_genes_per_cell=100,
        min_counts_per_cell=500,
        min_cells_per_gene=20,
    )

    assert config.min_genes_per_cell == 100
    assert config.min_counts_per_cell == 500
    assert config.min_cells_per_gene == 20


@pytest.mark.parametrize(
    "field_name",
    ["min_genes_per_cell", "min_counts_per_cell", "min_cells_per_gene"],
)
def test_floor_config_rejects_invalid_integer_floors(field_name: str) -> None:
    """
    Verify that integer QC floors reject invalid values.

    Floors should not accept booleans, negative values, or fractional values because those
    would produce confusing filtering rules.
    """

    # Confirm boolean values are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCFloorConfig.model_validate({field_name: True})

    # Confirm negative values are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCFloorConfig.model_validate({field_name: -1})

    # Confirm floating-point values are rejected.
    with pytest.raises(ValidationError, match=field_name):
        QCFloorConfig.model_validate({field_name: 1.5})


@pytest.mark.parametrize(
    "removed_ceiling",
    [
        "max_genes_per_cell",
        "max_counts_per_cell",
        "max_mito_percent",
        "max_ribo_percent",
        "max_hemoglobin_percent",
    ],
)
def test_the_five_removed_ceilings_are_not_silently_accepted(removed_ceiling: str) -> None:
    """A ceiling reaching the floor config must fail rather than be ignored.

    ``QCFloorConfig`` is strict, so an unknown field raises. That is the point: these five
    ceilings judged a cell against a cohort-wide number, which is exactly the rule that
    removed 20% of keratinocytes and 47% of smooth muscle for being legitimately
    mitochondrion-rich. Accepting the key and ignoring it would leave a config that reads as
    though a limit is enforced when it is not.
    """

    with pytest.raises(ValidationError):
        QCFloorConfig.model_validate({removed_ceiling: 10.0})


def test_a_floor_of_zero_is_distinct_from_no_floor() -> None:
    """``0`` and ``None`` must not collapse into each other.

    ``min_cells_per_gene: 0`` keeps every gene while still recording that a floor was applied;
    ``None`` disables the rule. Both retain everything, so a bug conflating them would be
    invisible in the output and visible only in the provenance a reader trusts.
    """

    assert QCFloorConfig(min_cells_per_gene=0).min_cells_per_gene == 0
    assert QCFloorConfig(min_cells_per_gene=None).min_cells_per_gene is None


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

    # Confirm percent_top values were cleaned.
    assert config.metrics.percent_top == [20, 50]

    # Confirm the optional audit families were disabled.
    assert config.doublets.enabled is False
    assert config.ambient.enabled is False

    # Confirm the live sub-configs are reachable through this helper. The hand-maintained
    # allow-list this replaced had drifted and rejected `graded`, `mito_mixture` and
    # `attrition_audit` — three sections that do real work — so a config using them failed
    # validation with an "unknown key" error naming a key that was perfectly valid.
    for section in ("graded", "mito_mixture", "attrition_audit", "floors"):
        assert isinstance(validate_qc_config_dict({section: {}}), QCConfig)


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
        validate_qc_config_dict({"floors": {"min_genes_per_cell": -5}})
