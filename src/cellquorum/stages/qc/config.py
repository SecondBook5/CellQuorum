"""Configuration and validation for QC, query projection, and finalization."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import ClassVar, Literal, overload

from pydantic import Field, ValidationError, field_validator, model_validator

from cellquorum.config.base import StrictBaseModel
from cellquorum.config.validation import (
    ConfigValidationError,
    reject_unknown_keys,
    require_mapping,
)
from cellquorum.core.exceptions import CellQuorumConfigError


def _type_name(value: object) -> str:
    """Return the runtime type name used in "Received: ..." error suffixes."""

    return type(value).__name__


def coerce_percent_top(value: object) -> list[int]:
    """Coerce and validate a ``percent_top`` setting."""

    if value is None:
        raise ValueError("percent_top cannot be None.")

    if isinstance(value, str):
        raise ValueError("percent_top must be a list of positive integers, not a string.")

    if not isinstance(value, list | tuple):
        raise ValueError(
            f"percent_top must be a list of positive integers. Received: {_type_name(value)}."
        )

    if not value:
        raise ValueError("percent_top must contain at least one positive integer.")

    cleaned_values: list[int] = []

    for item in value:
        if isinstance(item, bool):
            raise ValueError("percent_top values must be integers, not booleans.")

        if not isinstance(item, int):
            raise ValueError(f"percent_top values must be integers. Received: {_type_name(item)}.")

        if item <= 0:
            raise ValueError("percent_top values must be > 0.")

        cleaned_values.append(item)

    return sorted(set(cleaned_values))


@overload
def coerce_stripped_string(
    value: object,
    *,
    optional: Literal[False],
    type_message: str,
    empty_message: str,
) -> str: ...


@overload
def coerce_stripped_string(
    value: object,
    *,
    optional: bool,
    type_message: str,
    empty_message: str,
) -> str | None: ...


def coerce_stripped_string(
    value: object,
    *,
    optional: bool,
    type_message: str,
    empty_message: str,
) -> str | None:
    """Coerce a candidate into a stripped, non-empty string."""

    if optional and value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(f"{type_message} Received: {_type_name(value)}.")

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(empty_message)

    return cleaned_value


def coerce_string_list(
    value: object,
    *,
    not_a_list_message: str,
    wrong_container_message: str,
    item_type_message: str,
    empty_item_message: str,
) -> list[str]:
    """Coerce a candidate into a list of stripped, non-empty strings."""

    if value is None:
        return []

    if isinstance(value, str):
        raise ValueError(not_a_list_message)

    if not isinstance(value, list | tuple):
        raise ValueError(f"{wrong_container_message} Received: {_type_name(value)}.")
    cleaned_values: list[str] = []

    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{item_type_message} Received: {_type_name(item)}.")

        cleaned_item = item.strip()

        if not cleaned_item:
            raise ValueError(empty_item_message)

        cleaned_values.append(cleaned_item)

    return cleaned_values


@overload
def coerce_non_negative_int(
    value: object,
    *,
    optional: Literal[False],
    bool_message: str,
    type_message: str,
    negative_message: str,
) -> int: ...


@overload
def coerce_non_negative_int(
    value: object,
    *,
    optional: bool,
    bool_message: str,
    type_message: str,
    negative_message: str,
) -> int | None: ...


def coerce_non_negative_int(
    value: object,
    *,
    optional: bool,
    bool_message: str,
    type_message: str,
    negative_message: str,
) -> int | None:
    """Coerce a candidate into a non-negative integer."""

    if optional and value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(bool_message)

    if not isinstance(value, int):
        raise ValueError(f"{type_message} Received: {_type_name(value)}.")

    if value < 0:
        raise ValueError(negative_message)

    return value


def coerce_float_in_range(
    value: object,
    *,
    optional: bool,
    low: float,
    high: float,
    bool_message: str,
    type_message: str,
    range_message: str,
) -> float | None:
    """Coerce a candidate into a float within the inclusive ``[low, high]`` range."""

    if optional and value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(bool_message)

    if not isinstance(value, int | float):
        raise ValueError(f"{type_message} Received: {_type_name(value)}.")

    float_value = float(value)

    if not isfinite(float_value) or float_value < low or float_value > high:
        raise ValueError(range_message)

    return float_value


def coerce_positive_float(
    value: object,
    *,
    bool_message: str,
    type_message: str,
    nonpositive_message: str,
) -> float:
    """Coerce a candidate into a strictly positive float."""

    if isinstance(value, bool):
        raise ValueError(bool_message)

    if not isinstance(value, int | float):
        raise ValueError(f"{type_message} Received: {_type_name(value)}.")

    float_value = float(value)

    if not isfinite(float_value) or float_value <= 0.0:
        raise ValueError(nonpositive_message)

    return float_value


type DoubletMethod = Literal["none", "scrublet", "scdblfinder"]

type AmbientMethod = Literal["none", "audit", "soupx", "decontx"]

type DuplicateNamePolicy = Literal["warn", "make_unique", "error", "ignore"]

type QCFigureFormat = Literal["png", "pdf", "svg"]


class QCMetricCalculationConfig(StrictBaseModel):
    """Store settings for QC metric calculation.

    Args:
        percent_top: Top-n gene ranks used to calculate cumulative count fractions.
        log1p: Whether log1p QC metrics should be calculated.
        layer: Optional AnnData layer to use for QC metric calculation.
        use_raw: Whether AnnData.raw should be used when available.
    """

    percent_top: list[int] = Field(default_factory=lambda: [20])
    log1p: bool = True
    layer: str | None = None
    use_raw: bool = False

    @field_validator("percent_top", mode="before")
    @classmethod
    def validate_percent_top(cls, value: object) -> list[int]:
        """Validate percent-top settings."""

        return coerce_percent_top(value)

    @field_validator("layer", mode="before")
    @classmethod
    def validate_optional_layer(cls, value: object) -> str | None:
        """Validate the optional AnnData layer name."""

        return coerce_stripped_string(
            value,
            optional=True,
            type_message="QC metric layer must be a string or None.",
            empty_message="QC metric layer cannot be empty.",
        )


class QCFeaturePatternConfig(StrictBaseModel):
    """Store gene-feature patterns used by QC metrics.

    Args:
        mitochondrial_prefixes: Prefixes treated as mitochondrial genes.
        ribosomal_prefixes: Prefixes treated as ribosomal genes.
        hemoglobin_regexes: Regex patterns treated as hemoglobin genes.
        custom_exclude_prefixes: Optional project-specific prefixes flagged for QC.
    """

    mitochondrial_prefixes: list[str] = Field(default_factory=lambda: ["MT-"])
    ribosomal_prefixes: list[str] = Field(default_factory=lambda: ["RPS", "RPL"])
    hemoglobin_regexes: list[str] = Field(default_factory=lambda: [r"^HB[ABDEGMQZ]\d*(?!\w)"])
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
        """Validate a string-pattern list."""

        return coerce_string_list(
            value,
            not_a_list_message="Feature patterns must be provided as a list, not a string.",
            wrong_container_message="Feature patterns must be provided as a list of strings.",
            item_type_message="Feature patterns must be strings.",
            empty_item_message="Feature patterns cannot be empty.",
        )


class QCFloorConfig(StrictBaseModel):
    """Absolute floors, below which a barcode is not a cell and a gene is not measurable.

    Args:
        min_genes_per_cell: Genes a barcode must detect to be a cell. None disables the floor.
        min_counts_per_cell: Counts a barcode must carry. None disables the floor.
        min_cells_per_gene: Cells a gene must be detected in to be measurable. None disables it.
    """

    min_genes_per_cell: int | None = 200

    min_counts_per_cell: int | None = None

    min_cells_per_gene: int | None = 3

    @field_validator(
        "min_genes_per_cell",
        "min_counts_per_cell",
        "min_cells_per_gene",
        mode="before",
    )
    @classmethod
    def validate_optional_non_negative_int(cls, value: object) -> int | None:
        """Validate optional non-negative integer floors."""

        return coerce_non_negative_int(
            value,
            optional=True,
            bool_message="QC floors cannot be boolean values.",
            type_message="QC floors must be integers.",
            negative_message="QC floors must be >= 0.",
        )


class QCMitoMixtureConfig(StrictBaseModel):
    """Store mixture-model (miQC) mitochondrial QC settings.

    Args:
        enabled: Whether mixture-model mitochondrial filtering is enabled.
        mito_metric: Mitochondrial percentage metric, the regression response.
        complexity_metric: Library complexity metric, the regression predictor.
        posterior_cutoff: Compromised probability above which a cell is removed.
        monotone_mito_projection: Whether to reduce the fitted model to one
            mitochondrial ceiling per group and filter on that instead of on the
            posterior directly. Strongly recommended. The posterior depends on
            both mitochondrial fraction and complexity, so on a lineage with
            little mitochondrial spread the mixture separates on COMPLEXITY and
            the rule stops being a mitochondrial rule at all: on the skin atlas
            it removed plasma cells from 1.71% mitochondrial content upward while
            keeping others at 2.49%, and removed the deepest neutrophils (2,088
            genes) while keeping the shallowest (509). Projecting onto the
            mitochondrial axis makes "higher mitochondrial fraction is worse" true
            by construction, and turns the model into a per-lineage ceiling that
            can be stated in a methods section.
        keep_all_below_boundary: Whether to keep every cell below the intact
            component's own fitted line regardless of its posterior.
        enforce_left_cutoff: Whether to also remove cells that are both no more
            complex and no less mitochondrial than the least-mitochondrial cell
            already being removed.
        groupby: Metadata columns defining the fitting groups. Grouping is the
            entire point, and it should carry cell IDENTITY and NOTHING ELSE --
            in particular not sample. Two findings on the skin atlas fix this:

            Identity must be in the grouping, because mitochondrial baseline is
            lineage-specific and a fit spanning lineages splits on identity
            rather than viability. A per-sample-only fit removed 63% of one
            sample's keratinocytes at a median of 2,506 detected genes, because
            keratinocyte mitochondrial content is 4.6x the fibroblast median in
            the same sample.

            Sample must NOT be in the grouping, because a two-component mixture
            splits whatever it is given, including a group with no damaged cells
            in it. Adding ``sample_id`` made the cleanest sample's fibroblasts
            lose 21.1% of cells at a median of 0.67% mitochondrial content, and
            reproduced the pathology of per-sample MAD: the cleanest sample gets
            the harshest boundary. Damage is an absolute biophysical state, so
            what varies between samples is the PROPORTION of damaged cells, not
            the mitochondrial fraction at which damage begins. Pooling samples
            and grouping on identity lets the proportion vary and holds the
            boundary fixed, and per-sample attrition then tracks sample quality
            monotonically (1.7% on the cleanest sample, 23.5% on the dirtiest).

            So: ``[cell_type]`` on mixed populations, and an empty list on a
            single-lineage subset, which is already identity-grouped. Note that
            annotation must therefore precede this rule; on an unannotated object
            an empty list pools everything into one model, which is a learned
            global ceiling rather than a per-lineage one.
        fallback_groupby: Progressively coarser groupings tried when a group is
            too small to fit. Each fallback model is estimated on all of that
            coarser group's cells but applied only to the cells still awaiting a
            model, so a rare cell type borrows strength across samples instead of
            going unfiltered. ``[[]]`` -- one pooled model -- is the fallback that
            always resolves, because a pooled level has exactly one group and so
            either works for every cell or for none.
        level_policy: How the grouping hierarchy is resolved.

            ``uniform`` (default) resolves ONE level for the whole dataset: the
            finest level at which every group can be fit, or the next one down.
            ``per_group`` resolves it per group, so a group that cannot be fit
            borrows a coarser model while its fittable neighbours keep their own.

            ``per_group`` filters more cells and models each lineage more
            faithfully, and it is the right choice when the groups are a large
            atlas's cell types. It also has a failure mode that no amount of care
            in the config prevents: group SIZE correlates with study arm in most
            real cohorts (rarer condition, fewer donors, fewer cells), so WHICH
            cells got a fine model correlates with the factor under test, and a
            threshold that varies with the design factor is a covariate rather
            than a filter. ``uniform`` removes that by construction. Either way
            the stage audits the attrition it produced -- see
            ``cellquorum.stages.qc.attrition``.
        min_cells: Smallest group that will be fit rather than deferred.
        max_iterations: Expectation-maximisation iteration cap.
        tolerance: Relative log-likelihood improvement treated as converged.
        n_restarts: Restarts used to escape local optima. Restart 0 is
            deterministic, so the usual case does not depend on the seed.
        random_state: Seed for the randomized restarts.
        min_component_weight: Smallest share of cells a component may hold before
            the fit is treated as collapsed to one component.
    """

    enabled: bool = False
    mito_metric: str = "pct_counts_mito"

    complexity_metric: str = "n_genes_by_counts"
    posterior_cutoff: float = 0.75

    monotone_mito_projection: bool = True
    keep_all_below_boundary: bool = True
    enforce_left_cutoff: bool = True

    groupby: list[str] = Field(default_factory=list)
    fallback_groupby: list[list[str]] = Field(default_factory=list)

    level_policy: Literal["uniform", "per_group"] = "uniform"
    min_cells: int = 100
    max_iterations: int = 500
    tolerance: float = 1e-6
    n_restarts: int = 5
    random_state: int = 0
    min_component_weight: float = 0.01

    @field_validator("mito_metric", "complexity_metric", mode="before")
    @classmethod
    def validate_metric_name(cls, value: object) -> str:
        """Validate a modelled metric column name."""

        return coerce_stripped_string(
            value,
            optional=False,
            type_message="Mixture metric names must be strings.",
            empty_message="Mixture metric names cannot be empty.",
        )

    @field_validator("groupby", mode="before")
    @classmethod
    def validate_groupby(cls, value: object) -> list[str]:
        """Validate the fitting group columns."""

        return coerce_string_list(
            value,
            not_a_list_message="mito_mixture.groupby must be a list, not a string.",
            wrong_container_message="mito_mixture.groupby must be a list of strings.",
            item_type_message="mito_mixture.groupby entries must be strings.",
            empty_item_message="mito_mixture.groupby entries cannot be empty.",
        )

    @field_validator("fallback_groupby", mode="before")
    @classmethod
    def validate_fallback_groupby(cls, value: object) -> list[list[str]]:
        """Validate the coarser fallback groupings."""

        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError(
                "mito_mixture.fallback_groupby must be a list of groupings, each "
                "itself a list of column names."
            )

        return [
            coerce_string_list(
                grouping,
                not_a_list_message=(
                    "Each mito_mixture.fallback_groupby entry must be a list of "
                    "column names, not a string."
                ),
                wrong_container_message=(
                    "Each mito_mixture.fallback_groupby entry must be a list of strings."
                ),
                item_type_message="mito_mixture.fallback_groupby names must be strings.",
                empty_item_message="mito_mixture.fallback_groupby names cannot be empty.",
            )
            for grouping in value
        ]

    @field_validator("posterior_cutoff", "min_component_weight", mode="before")
    @classmethod
    def validate_probability(cls, value: object) -> float:
        """Validate a probability strictly inside the open unit interval."""

        probability = coerce_float_in_range(
            value,
            optional=False,
            low=0.0,
            high=1.0,
            bool_message="Mixture probabilities cannot be boolean values.",
            type_message="Mixture probabilities must be numeric.",
            range_message="Mixture probabilities must lie between 0 and 1.",
        )

        if probability is None or probability in {0.0, 1.0}:
            raise ValueError(
                "Mixture probabilities must be strictly between 0 and 1, " f"not {probability}."
            )

        return float(probability)

    @field_validator("tolerance", mode="before")
    @classmethod
    def validate_tolerance(cls, value: object) -> float:
        """Validate the convergence tolerance."""

        return coerce_positive_float(
            value,
            bool_message="mito_mixture.tolerance cannot be a boolean value.",
            type_message="mito_mixture.tolerance must be numeric.",
            nonpositive_message="mito_mixture.tolerance must be > 0.",
        )

    @field_validator("min_cells", "max_iterations", "n_restarts", mode="before")
    @classmethod
    def validate_positive_int(cls, value: object) -> int:
        """Validate a strictly positive integer setting."""

        count = coerce_non_negative_int(
            value,
            optional=False,
            bool_message="Mixture counts cannot be boolean values.",
            type_message="Mixture counts must be integers.",
            negative_message="Mixture counts cannot be negative.",
        )

        if count < 1:
            raise ValueError("Mixture counts must be >= 1.")

        return count


class QCDoubletConfig(StrictBaseModel):
    """Store doublet-detection settings.

    Args:
        enabled: Whether doublet detection or auditing is enabled.
        method: Doublet detection method.
        methods: Detectors to run (consensus over these); overrides single method when set.
        consensus: How to combine per-method calls: any | all | majority.
        remove: Whether predicted doublets should be removed automatically.
        expected_doublet_rate: Expected doublet fraction for both detectors (scDblFinder dbr).
        score_threshold: Optional manual doublet score threshold.
        per_sample: Whether doublets should be evaluated per sample.
    """

    enabled: bool = True
    method: DoubletMethod = "scdblfinder"

    methods: list[str] = Field(default_factory=lambda: ["scdblfinder"])
    consensus: str = "any"
    remove: bool = False
    expected_doublet_rate: float = 0.06
    score_threshold: float | None = None
    per_sample: bool = True

    @field_validator("expected_doublet_rate", "score_threshold", mode="before")
    @classmethod
    def validate_optional_probability(cls, value: object) -> float | None:
        """Validate optional probability fields."""

        return coerce_float_in_range(
            value,
            optional=True,
            low=0.0,
            high=1.0,
            bool_message="Doublet probability fields cannot be boolean values.",
            type_message="Doublet probability fields must be numeric.",
            range_message="Doublet probability fields must be between 0 and 1.",
        )

    @model_validator(mode="after")
    def validate_doublet_consistency(self) -> QCDoubletConfig:
        """Validate consistency between doublet flags and method."""

        if self.enabled and self.method == "none":
            raise ValueError("Doublet detection cannot be enabled when method is 'none'.")

        if self.remove and self.method == "none":
            raise ValueError("Doublet removal cannot be enabled when method is 'none'.")

        return self


class QCAmbientRNAConfig(StrictBaseModel):
    """Legacy ambient settings checked against upstream correction provenance.

    Args:
        enabled: Whether ambient RNA assessment is enabled.
        method: Ambient RNA method.
        correction_enabled: Require matching upstream correction provenance. QC itself
            does not perform ambient correction.
        contamination_fraction: Optional assumed contamination fraction.
        marker_genes: Optional marker genes used for contamination audits.
        require_raw_droplets_for_correction: Whether correction requires raw droplets.
    """

    enabled: bool = True
    method: AmbientMethod = "audit"
    correction_enabled: bool = False
    contamination_fraction: float | None = None
    marker_genes: list[str] = Field(default_factory=list)
    require_raw_droplets_for_correction: bool = True

    @field_validator("contamination_fraction", mode="before")
    @classmethod
    def validate_contamination_fraction(cls, value: object) -> float | None:
        """Validate optional contamination fraction."""

        return coerce_float_in_range(
            value,
            optional=True,
            low=0.0,
            high=1.0,
            bool_message="contamination_fraction cannot be boolean.",
            type_message="contamination_fraction must be numeric.",
            range_message="contamination_fraction must be between 0 and 1.",
        )

    @field_validator("marker_genes", mode="before")
    @classmethod
    def validate_marker_genes(cls, value: object) -> list[str]:
        """Validate ambient RNA marker genes."""

        return coerce_string_list(
            value,
            not_a_list_message="marker_genes must be provided as a list, not a string.",
            wrong_container_message="marker_genes must be a list of strings.",
            item_type_message="Ambient RNA marker genes must be strings.",
            empty_item_message="Ambient RNA marker genes cannot be empty.",
        )

    @model_validator(mode="after")
    def validate_ambient_consistency(self) -> QCAmbientRNAConfig:
        """Validate consistency between ambient RNA flags and method."""

        if self.enabled and self.method == "none":
            raise ValueError("Ambient RNA assessment cannot be enabled when method is 'none'.")

        if self.correction_enabled and self.method not in {"soupx", "decontx"}:
            raise ValueError(
                "Ambient RNA correction requires method 'soupx' or 'decontx'. "
                "Use method='audit' only for non-mutating assessment."
            )

        return self


class QCDuplicateNameConfig(StrictBaseModel):
    """Store duplicate observation and variable name policies.

    Args:
        var_names: Policy for duplicate AnnData variable names.
        obs_names: Policy for duplicate AnnData observation names.
    """

    var_names: DuplicateNamePolicy = "make_unique"
    obs_names: DuplicateNamePolicy = "warn"


class QCOutputConfig(StrictBaseModel):
    """Store QC output settings.

    Args:
        write_metrics_table: Whether to write cell and gene QC metric tables.
        write_filter_table: Whether to write filtering decision tables.
        write_mixture_table: Whether to write threshold tables.
        write_report_table: Whether to write the per-group QC report table
            (cells before/removed/%/after per cell type + a TOTAL row). Enabled
            by default; the grouping falls back to a single TOTAL row when no
            cell-type labels are present on the input object.
        cell_labels: Whether to write ``cell_labels.csv`` — the sample, donor,
            condition and cell-type labels of every cell that ENTERED QC. Under
            ``mode="filter"`` the written h5ad has lost the removed cells, so
            without this table a later re-render can only guess at their labels,
            and a by-cell-type attrition figure built on the guess reports every
            cell type as losing nothing. With it, the run directory can re-render
            every QC figure and table exactly, off the tables alone.
        attrition_audit: Whether to write ``qc_attrition.csv`` -- the per-factor
            differential-attrition tests. One row per (factor, unit of analysis),
            including the tests that were skipped and why, so the table answers
            "was this checked" and not only "was anything found".
        write_summary_json: Whether to write a JSON QC summary.
        write_h5ad: Whether to write a QC AnnData object.
        write_figures: Master switch for every QC figure. False writes no
            figures at all, whatever the per-writer flags below say.
        figure_format: File format used for QC figures.
        html_report: Whether to write the single-file HTML QC report
            (``qc_report.html``): cohort funnel, per-sample attrition, rule
            attribution, applied thresholds. The CSVs stay canonical; this is the
            human-readable view of them, and it is what makes a large per-sample
            drop legible without joining four tables by hand.
        overview_figures: Whether to write the figure-ready QC panel set
            (``qc_overview`` plus its standalone panels). These answer "what did
            QC do to this cohort" — funnel, rule attribution, donor-paired
            contrast, joint scatter with the exclusion regions drawn, per-sample
            attrition and a per-sample metric matrix — as opposed to the
            per-metric audit distributions the other two writers produce.
        publication_tables: Whether to write the typeset QC tables — the Table 1
            a manuscript needs — as one HTML page plus a ``booktabs`` ``.tex``
            and a raster of each. Same numbers as the CSVs, set rather than
            dumped, so the QC paragraph of a paper can be written from them.
    """

    write_metrics_table: bool = True
    write_filter_table: bool = True
    write_mixture_table: bool = True
    write_report_table: bool = True
    cell_labels: bool = True
    attrition_audit: bool = True
    write_summary_json: bool = True
    write_h5ad: bool = True
    write_figures: bool = True
    figure_format: QCFigureFormat = "png"
    html_report: bool = True
    overview_figures: bool = True
    publication_tables: bool = True

    figure_dpi: int = 300


class QCAttritionAuditConfig(StrictBaseModel):
    """Store settings for the differential-attrition audit.

    Args:
        enabled: Whether the audit runs. On by default -- it is a handful of
            contingency tests on a table the stage already built, and the failure
            it detects is invisible in every downstream result.
        factors: Extra ``obs`` columns to audit, beyond the condition and batch
            keys resolved from the cohort and design blocks. Name whatever enters
            a downstream model: treatment, timepoint, site.
        block: ``obs`` column to stratify and pair on, or None to resolve the
            cohort/design donor key. Donor quality varies far more than QC
            thresholds do and donors are rarely balanced across arms, so the
            pooled test can report an association that no donor exhibits.
        audit_batch: Whether the batch key is audited alongside condition.
            Attrition tracking batch is the same defect as attrition tracking
            condition, and integration will not repair it.
        audit_subsets: Whether every test is repeated within each subset of the
            object. A cohort removal rate is an average and the analyses that
            follow QC are not: a half-point cohort gap can be four points inside
            one lineage and zero everywhere else, and it is the per-lineage
            contrast that reaches a figure. Subset p-values are
            Benjamini-Hochberg adjusted, so switching this on does not cost the
            cohort test any power.
        subset: ``obs`` column to stratify the audit by, or None to resolve the
            engine's cell-type annotation convention. Named explicitly, a column
            the object does not carry simply produces no subset pass.
        alpha: Significance level for the warning.
        min_rate_difference: Smallest removal-rate gap, as a fraction, that may
            raise a warning. Significance alone is worthless here: above a few
            tens of thousands of cells a half-point gap is significant at any
            alpha, and an engine that warns about it trains its users to ignore
            the warning. Measured gaps are always recorded whatever this is set
            to; the flag only controls what gets shouted about.
    """

    enabled: bool = True
    factors: list[str] = Field(default_factory=list)
    block: str | None = None
    audit_batch: bool = True
    audit_subsets: bool = True
    subset: str | None = None
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_rate_difference: float = Field(default=0.02, ge=0.0, le=1.0)


class QCGradedConfig(StrictBaseModel):
    """Graded adjudication: technical evidence -> core / borderline / quarantine.

    Args:
        enabled: Whether graded adjudication runs. On by default — it is the QC system.
        concern_severity: Family severity at or above which a family is concerning, making
            the cell at least borderline.
        severe_severity: Family severity at or above which a family is severe. Only severe
            families feed the concordance route to quarantine.
        min_concordant_families: Independent damage families that must be severe before
            quarantine is justified. Must be at least 2 — one would let a single model
            condemn a cell, which is the failure this design prevents.
        uninformative_capture_severity: Capture severity at or above which the barcode
            carries no usable information, justifying quarantine on its own.
        min_coverage_for_quarantine: Evidence coverage below which quarantine is withheld
            in favour of borderline. Less evidence must make the system more conservative.
        multiplet_severity: Multiplet severity at or above which a cell is flagged a
            probable multiplet. Recorded separately from damage; never quarantines.
        nuclear_axis_applicable: False for single-nucleus assays, where high
            nuclear-retained signal is expected rather than evidence of leakage.
    """

    enabled: bool = True

    concern_severity: float = 0.50

    severe_severity: float = 0.667

    min_concordant_families: int = 2

    uninformative_capture_severity: float = 0.90

    min_coverage_for_quarantine: float = 0.50
    multiplet_severity: float = 0.60
    nuclear_axis_applicable: bool = True

    lineage_conditional: bool = True

    lineage_resolution: float = 0.5

    lineage_min_cells: int = 25

    lineage_min_genes: int = 50

    lineage_suspect_severity: float = 0.667
    lineage_vulnerable_fraction: float = 0.50

    archetype_audit: bool = True
    archetype_max: int = 10
    archetype_bootstrap: int = 0

    archetype_max_cells: int = 10_000

    archetype_restarts: int = 1

    archetype_timeout_seconds: int = 900

    self_check: bool = True

    self_check_fails_run: bool = True

    self_check_minimum_core: float = 0.50


class QCSampleQCConfig(StrictBaseModel):
    """Configure upstream SampleQC as an assessment, without automatic exclusion."""

    enabled: bool = False
    sample_key: str = Field(default="sample_id", min_length=1)
    n_components: int | None = Field(default=None, strict=True, ge=1)
    alpha: float = Field(default=0.01, gt=0, lt=1, allow_inf_nan=False)
    min_cells_per_sample: int = Field(default=25, strict=True, ge=4)
    random_state: int = Field(default=22, strict=True, ge=0, le=2147483647)
    timeout_seconds: int = Field(default=600, strict=True, ge=1)

    @model_validator(mode="after")
    def require_components(self) -> QCSampleQCConfig:
        if self.enabled and self.n_components is None:
            raise ValueError("sampleqc.n_components is required when SampleQC is enabled.")
        if not self.sample_key.strip():
            raise ValueError("sampleqc.sample_key cannot be blank.")
        return self


class QCConfig(StrictBaseModel):
    """Store full QC module configuration.

    Args:
        enabled: Whether the QC module is enabled.
        metrics: QC metric calculation settings.
        floors: Absolute floors, the only place a barcode leaves the object.
        graded: Graded-adjudication settings (evidence -> core/borderline/quarantine).
        mito_mixture: Mixture-model (miQC) mitochondrial QC settings.
        sampleqc: Optional upstream multivariate assessment settings.
        features: Feature family pattern settings.
        doublets: Doublet detection settings.
        ambient: Ambient RNA assessment settings.
        duplicate_names: Duplicate name handling settings.
        attrition_audit: Differential-attrition audit settings -- whether QC
            removed cells at the same rate in every arm of the design.
        outputs: QC output settings.
        fail_on_empty_result: Whether filtering to zero cells or genes is fatal.
    """

    enabled: bool = True
    metrics: QCMetricCalculationConfig = Field(default_factory=QCMetricCalculationConfig)

    floors: QCFloorConfig = Field(default_factory=QCFloorConfig)
    graded: QCGradedConfig = Field(default_factory=QCGradedConfig)
    mito_mixture: QCMitoMixtureConfig = Field(default_factory=QCMitoMixtureConfig)
    sampleqc: QCSampleQCConfig = Field(default_factory=QCSampleQCConfig)
    features: QCFeaturePatternConfig = Field(default_factory=QCFeaturePatternConfig)
    doublets: QCDoubletConfig = Field(default_factory=QCDoubletConfig)
    ambient: QCAmbientRNAConfig = Field(default_factory=QCAmbientRNAConfig)
    duplicate_names: QCDuplicateNameConfig = Field(default_factory=QCDuplicateNameConfig)
    attrition_audit: QCAttritionAuditConfig = Field(default_factory=QCAttritionAuditConfig)
    outputs: QCOutputConfig = Field(default_factory=QCOutputConfig)
    fail_on_empty_result: bool = True

    _REMOVED_KEYS: ClassVar[dict[str, str]] = {
        "mode": (
            "QC no longer has a mode. The floors always remove what they match and graded "
            "adjudication never removes anything, so there is nothing left for a mode to "
            "select. For the old `flag_no_drop` behaviour set every floor to null:\n"
            "  qc:\n    floors:\n      min_genes_per_cell: null\n"
            "      min_counts_per_cell: null\n      min_cells_per_gene: null"
        ),
        "threshold_strategy": (
            "Threshold strategies ('fixed', 'mad', 'fixed_and_mad') are gone with the "
            "threshold path. Severity is graded per lineage instead; tune `qc.graded`."
        ),
        "mad": (
            "MAD thresholding is replaced by graded severity, which is a robust z against a "
            "lineage-conditional null rather than a cohort-wide MAD bound — the difference "
            "that stops rare populations being removed for being rare. Tune `qc.graded`."
        ),
        "cell_cycle": (
            "Cell-cycle scoring never worked here and has been removed. It needs a "
            "log-normalized layer, which preprocessing creates at order 30 — QC runs at 20, so "
            "`qc.cell_cycle.enabled: true` raised KeyError('cellquorum_normalized') on every "
            "real input. Score it where the normalized layer exists: set "
            "`embeddings.overlay.cell_cycle: true`, which now defaults to the same Tirosh gene "
            "sets this block used."
        ),
        "basic": (
            "`basic` is now `floors`, and keeps only min_genes_per_cell, min_counts_per_cell "
            "and min_cells_per_gene. The five `max_*` ceilings are gone: a cohort-wide "
            "`max_mito_percent` cannot distinguish a mitochondrion-rich cell type from a "
            "damaged cell, which is why it removed 20% of keratinocytes. Graded metabolic "
            "evidence answers that per lineage and cannot condemn a cell on its own."
        ),
    }

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_threshold_keys(cls, data: object) -> object:
        """Fail with a migration message when a config still sets a v1 threshold key."""
        if not isinstance(data, Mapping):
            return data
        for key, guidance in cls._REMOVED_KEYS.items():
            if key in data:
                raise CellQuorumConfigError(f"`qc.{key}` was removed. {guidance}")
        return data

    def enabled_metric_families(self) -> list[str]:
        """Return enabled QC metric families."""

        families = ["floors"]

        if self.doublets.enabled:
            families.append("doublets")

        if self.ambient.enabled:
            families.append("ambient_rna")

        return families


def validate_qc_config_dict(config: Mapping[str, object]) -> QCConfig:
    """Validate a plain mapping into a QCConfig object."""

    config_dict = require_mapping(config, field_path="qc")

    reject_unknown_keys(
        config_dict,
        allowed_keys=[*QCConfig.model_fields, *QCConfig._REMOVED_KEYS],
        field_path="qc",
    )

    try:
        return QCConfig.model_validate(config_dict)

    except ValidationError as error:
        raise ConfigValidationError(f"Invalid QC configuration:\n{error}") from error


class QCFinalizationConfig(StrictBaseModel):
    """Thresholds for per-cell rescue into qc_state_final."""

    min_neighborhood_support: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)

    max_ood_score: float = Field(default=0.95, ge=0, le=1, allow_inf_nan=False)

    severe_severity: float = Field(default=0.9, ge=0, le=1, allow_inf_nan=False)


class QueryProjectionConfig(StrictBaseModel):
    """Settings for projecting borderline cells onto the frozen core manifold."""

    use_rep: str | None = None

    label_column: str | None = None

    k: int = Field(default=15, ge=1, strict=True)


class QCSpliceMetricsConfig(StrictBaseModel):
    """Settings for the optional lightweight splice-QC stage (order=15).

    Reuses the trajectory stage's own manifest column names rather than inventing a
    second way to say the same thing: one manifest, one ``sample_col``/``loom_path_col``
    pair, whether the reader is this stage or the velocity method.
    """

    sample_col: str = "sample_id"

    loom_path_col: str = "loom_path"


__all__ = [
    "coerce_float_in_range",
    "coerce_non_negative_int",
    "coerce_percent_top",
    "coerce_positive_float",
    "coerce_string_list",
    "coerce_stripped_string",
    "QCFloorConfig",
    "AmbientMethod",
    "DoubletMethod",
    "DuplicateNamePolicy",
    "QCAmbientRNAConfig",
    "QCAttritionAuditConfig",
    "QCConfig",
    "QCDoubletConfig",
    "QCSpliceMetricsConfig",
    "QCDuplicateNameConfig",
    "QCFigureFormat",
    "QCFeaturePatternConfig",
    "QCMetricCalculationConfig",
    "QCOutputConfig",
    "validate_qc_config_dict",
    "QCFinalizationConfig",
    "QueryProjectionConfig",
]
