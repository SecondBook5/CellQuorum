"""QC configuration models for CellQuorum."""

from __future__ import annotations

# Import Mapping for dictionary-based QC config validation.
from collections.abc import Mapping

# Import Literal for constrained QC configuration values.
from typing import ClassVar, Literal

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

# Import CellQuorum exceptions for mode-rejection validation.
from cellquorum.core.exceptions import CellQuorumConfigError

# Import the reusable field-coercion helpers backing the field validators below.
from cellquorum.stages.qc.config_validators import (
    coerce_float_in_range,
    coerce_non_negative_int,
    coerce_percent_top,
    coerce_positive_float,
    coerce_string_list,
    coerce_stripped_string,
)

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

    percent_top: list[int] = Field(default_factory=lambda: [20])
    log1p: bool = True
    layer: str | None = None
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

        # Delegate to the shared positive-integer-rank coercion helper.
        return coerce_percent_top(value)

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

        # Delegate to the shared optional stripped-string coercion helper.
        return coerce_stripped_string(
            value,
            optional=True,
            type_message="QC metric layer must be a string or None.",
            empty_message="QC metric layer cannot be empty.",
        )


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
        """
        Validate a string-pattern list.

        Args:
            value: Candidate pattern list.

        Returns:
            Cleaned pattern list.

        Raises:
            ValueError: If the value is not a list of non-empty strings.
        """

        # Delegate to the shared string-list coercion helper.
        return coerce_string_list(
            value,
            not_a_list_message="Feature patterns must be provided as a list, not a string.",
            wrong_container_message="Feature patterns must be provided as a list of strings.",
            item_type_message="Feature patterns must be strings.",
            empty_item_message="Feature patterns cannot be empty.",
        )


class QCFloorConfig(StrictBaseModel):
    """Absolute floors, below which a barcode is not a cell and a gene is not measurable.

    These are the only fixed numbers left in QC, and none of them is a threshold in the v1
    sense. A floor states where the *assay's detection limit* lies; it does not judge a cell
    against its cohort. Everything that used to be judged by a fixed ceiling — mitochondrial
    percentage, gene and count maxima — is now graded severity, which asks whether a cell is
    unusual *for cells like it* and requires concordance across independent evidence families
    before condemning anything. See :mod:`cellquorum.stages.qc.floors`.

    The five ceilings that used to live here (``max_genes_per_cell``, ``max_counts_per_cell``,
    ``max_mito_percent``, ``max_ribo_percent``, ``max_hemoglobin_percent``) are gone. A hard
    ``max_mito_percent: 8.0`` is exactly the rule that removed 20% of keratinocytes and 47% of
    smooth muscle: those populations are legitimately mitochondrion-rich, and a cohort-wide
    ceiling cannot tell that from damage.

    **To run QC without removing anything**, set all three floors to ``null``. That replaces the
    old ``mode: flag_no_drop`` and is strictly more informative, because it says which floor was
    lifted rather than switching the whole stage off. Note that graded adjudication never deletes
    a cell in any configuration — it assigns per-analysis permissions — so the floors are the only
    place a barcode can leave the object.

    Args:
        min_genes_per_cell: Genes a barcode must detect to be a cell. None disables the floor.
        min_counts_per_cell: Counts a barcode must carry. None disables the floor.
        min_cells_per_gene: Cells a gene must be detected in to be measurable. None disables it.
    """

    #: Genes a barcode must detect. Below this there is no population to be unusual against,
    #: and an empty droplet admitted here can anchor a provisional lineage and corrupt its null.
    min_genes_per_cell: int | None = 200

    #: Counts a barcode must carry. Off by default — the gene floor is the better-behaved of
    #: the two, since total counts vary by an order of magnitude across real cell types.
    min_counts_per_cell: int | None = None

    #: Cells a gene must be detected in. Gene filtering has no home in graded QC, which scores
    #: cells and never genes, so it lives here on its own axis.
    min_cells_per_gene: int | None = 3

    @field_validator(
        "min_genes_per_cell",
        "min_counts_per_cell",
        "min_cells_per_gene",
        mode="before",
    )
    @classmethod
    def validate_optional_non_negative_int(cls, value: object) -> int | None:
        """
        Validate optional non-negative integer floors.

        Args:
            value: Candidate floor.

        Returns:
            Validated integer floor or None.

        Raises:
            ValueError: If the value is negative, boolean, or non-integer.
        """

        # Delegate to the shared optional non-negative-integer coercion helper.
        return coerce_non_negative_int(
            value,
            optional=True,
            bool_message="QC floors cannot be boolean values.",
            type_message="QC floors must be integers.",
            negative_message="QC floors must be >= 0.",
        )


class QCMitoMixtureConfig(StrictBaseModel):
    """
    Store mixture-model (miQC) mitochondrial QC settings.

    This is the principled alternative to the other two mitochondrial policies in
    this config, both of which answer the wrong question. `basic.max_mito_percent`
    asks "what percentage is too high?", which has no data-driven answer.
    `mad.mito_metric` asks "how far from the median is unusual?", which assumes one
    healthy mode and therefore tightens as a sample gets cleaner. This model asks
    "is this cell better explained by the intact population or the damaged one?"
    and derives the boundary per sample from the joint structure of mitochondrial
    fraction and library complexity. See `cellquorum.stages.qc.mixture` for the
    model and Hippen et al., PLoS Comput Biol 2021 for the method.

    Enabling this requires setting `mad.mito_metric: null`, because two adaptive
    mitochondrial rules at once is not a policy -- whichever is stricter silently
    wins. Keep `basic.max_mito_percent` as a loose hard backstop rather than a
    filter; if it is set tight it will override the model on exactly the cells the
    model was brought in to judge, and the stage warns when that happens.

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

    # Store the library complexity metric used as the regression predictor.
    #
    # Raw detected genes rather than a log transform, matching miQC: the model
    # wants the linear relationship between complexity and mitochondrial fraction
    # that RNA leakage produces.
    complexity_metric: str = "n_genes_by_counts"
    posterior_cutoff: float = 0.75

    # Store whether the fitted model is reduced to one mitochondrial ceiling per
    # group before filtering.
    #
    # On by default because the unprojected posterior is a function of two
    # variables and can therefore discard a cleaner cell than one it keeps, which
    # is indefensible in a mitochondrial QC rule. Projection also produces the
    # number a methods section actually needs: a per-group ceiling in percent.
    monotone_mito_projection: bool = True
    keep_all_below_boundary: bool = True
    enforce_left_cutoff: bool = True

    # Store the fitting groups. Empty by default because the safe default is a
    # single-lineage object, where identity grouping is already implicit.
    groupby: list[str] = Field(default_factory=list)
    fallback_groupby: list[list[str]] = Field(default_factory=list)

    # Store how the grouping hierarchy is resolved.
    #
    # Uniform by default: one level for the whole dataset. The alternative lets
    # the level vary between groups, which is a better model of each lineage and a
    # worse guarantee about the design, because group size and study arm are
    # correlated in most cohorts.
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
        """
        Validate a modelled metric column name.

        Args:
            value: Candidate metric name.

        Returns:
            Cleaned metric name.

        Raises:
            ValueError: If the name is not a non-empty string.
        """

        # Delegate to the shared stripped-string coercion helper.
        return coerce_stripped_string(
            value,
            optional=False,
            type_message="Mixture metric names must be strings.",
            empty_message="Mixture metric names cannot be empty.",
        )

    @field_validator("groupby", mode="before")
    @classmethod
    def validate_groupby(cls, value: object) -> list[str]:
        """
        Validate the fitting group columns.

        Args:
            value: Candidate groupby list.

        Returns:
            Cleaned list of column names.

        Raises:
            ValueError: If the value is not a list of non-empty strings.
        """

        # Delegate to the shared string-list coercion helper.
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
        """
        Validate the coarser fallback groupings.

        Args:
            value: Candidate list of groupings.

        Returns:
            Cleaned list of column-name lists.

        Raises:
            ValueError: If the value is not a list of lists of non-empty strings.
        """

        # Treat an absent value as no fallback.
        if value is None:
            return []

        # Reject anything that is not a list of groupings.
        if not isinstance(value, list):
            raise ValueError(
                "mito_mixture.fallback_groupby must be a list of groupings, each "
                "itself a list of column names."
            )

        # Validate each grouping with the shared string-list helper, allowing the
        # empty grouping as an explicit request for a final pooled fit.
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
        """
        Validate a probability strictly inside the open unit interval.

        Args:
            value: Candidate probability.

        Returns:
            Validated probability.

        Raises:
            ValueError: If the value is boolean, non-numeric, or not in (0, 1).
        """

        # Reject values outside the unit interval first, for a clear message.
        probability = coerce_float_in_range(
            value,
            optional=False,
            low=0.0,
            high=1.0,
            bool_message="Mixture probabilities cannot be boolean values.",
            type_message="Mixture probabilities must be numeric.",
            range_message="Mixture probabilities must lie between 0 and 1.",
        )

        # Reject the endpoints, which would make the rule fire on every cell or
        # on none of them regardless of the model.
        if probability is None or probability in {0.0, 1.0}:
            raise ValueError(
                "Mixture probabilities must be strictly between 0 and 1, " f"not {probability}."
            )

        # Return the validated probability.
        return float(probability)

    @field_validator("tolerance", mode="before")
    @classmethod
    def validate_tolerance(cls, value: object) -> float:
        """
        Validate the convergence tolerance.

        Args:
            value: Candidate tolerance.

        Returns:
            Validated positive tolerance.

        Raises:
            ValueError: If the value is boolean, non-numeric, or non-positive.
        """

        # Delegate to the shared strictly-positive-float coercion helper.
        return coerce_positive_float(
            value,
            bool_message="mito_mixture.tolerance cannot be a boolean value.",
            type_message="mito_mixture.tolerance must be numeric.",
            nonpositive_message="mito_mixture.tolerance must be > 0.",
        )

    @field_validator("min_cells", "max_iterations", "n_restarts", mode="before")
    @classmethod
    def validate_positive_int(cls, value: object) -> int:
        """
        Validate a strictly positive integer setting.

        Args:
            value: Candidate integer value.

        Returns:
            Validated positive integer.

        Raises:
            ValueError: If the value is boolean, non-integer, or below one.
        """

        # Reject negatives and non-integers with the shared helper.
        count = coerce_non_negative_int(
            value,
            optional=False,
            bool_message="Mixture counts cannot be boolean values.",
            type_message="Mixture counts must be integers.",
            negative_message="Mixture counts cannot be negative.",
        )

        # Reject zero, which would disable fitting rather than configure it.
        if count < 1:
            raise ValueError("Mixture counts must be >= 1.")

        # Return the validated count.
        return count


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

    enabled: bool = True
    method: DoubletMethod = "scdblfinder"

    # Store detectors to run (consensus over these); overrides single method when
    # set. Default matches the single-method default (`method`) so an unconfigured
    # run uses the same detector regardless of which field the caller reads.
    methods: list[str] = Field(default_factory=lambda: ["scdblfinder"])
    consensus: str = "any"
    remove: bool = False
    expected_doublet_rate: float = 0.06
    score_threshold: float | None = None
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

        # Delegate to the shared bounded-float coercion helper (0-1 probability).
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

    enabled: bool = False
    score_layer: str = "cellquorum_normalized"
    s_genes: list[str] = Field(default_factory=list)
    g2m_genes: list[str] = Field(default_factory=list)
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

        # Delegate to the shared string-list coercion helper.
        return coerce_string_list(
            value,
            not_a_list_message="Gene lists must be provided as lists, not strings.",
            wrong_container_message="Gene lists must be lists of strings.",
            item_type_message="Gene names must be strings.",
            empty_item_message="Gene names cannot be empty.",
        )

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

        # Delegate to the shared required non-negative-integer coercion helper.
        return coerce_non_negative_int(
            value,
            optional=False,
            bool_message="random_state cannot be a boolean value.",
            type_message="random_state must be an integer.",
            negative_message="random_state must be >= 0.",
        )


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

    enabled: bool = True
    method: AmbientMethod = "audit"
    correction_enabled: bool = False
    contamination_fraction: float | None = None
    marker_genes: list[str] = Field(default_factory=list)
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

        # Delegate to the shared bounded-float coercion helper (0-1 fraction).
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
        """
        Validate ambient RNA marker genes.

        Args:
            value: Candidate marker gene list.

        Returns:
            Cleaned marker gene list.

        Raises:
            ValueError: If the value is not a list of non-empty strings.
        """

        # Delegate to the shared string-list coercion helper.
        return coerce_string_list(
            value,
            not_a_list_message="marker_genes must be provided as a list, not a string.",
            wrong_container_message="marker_genes must be a list of strings.",
            item_type_message="Ambient RNA marker genes must be strings.",
            empty_item_message="Ambient RNA marker genes cannot be empty.",
        )

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

    var_names: DuplicateNamePolicy = "make_unique"
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

    # Store the QC figure DPI resolution.
    figure_dpi: int = 300


class QCAttritionAuditConfig(StrictBaseModel):
    """
    Store settings for the differential-attrition audit.

    Every threshold in this module is chosen to be defensible on its own terms.
    None of that guarantees the property downstream statistics depend on, which
    is that QC removed cells at the same RATE in every arm of the design. When it
    did not, QC has stopped being a filter and become a covariate: whatever the
    diseased arm looks like afterwards is partly a statement about which of its
    cells survived. Adaptive thresholds make this MORE likely, not less, because
    they are estimated from data that differ between arms.

    So the engine tests for it on every run rather than leaving it to whoever
    happens to look at the attrition figures. See
    :mod:`cellquorum.stages.qc.attrition` for the three tests and why the unit of
    analysis matters.

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

    Replaces "fail any threshold -> removed" with a verdict that no single statistical
    model can force. There are exactly two routes to quarantine: an uninformative barcode
    (capture so poor there is nothing to adjudicate), or concordant severe evidence across
    independent evidence *families*. Correlated metrics inside one family count once.

    Every bar is a property of an assay and a tissue, to be read off calibration figures.
    The defaults below are the permissive end on purpose: they exist so the stage runs, and
    they are deliberately not tuned for any dataset.

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

    # Whether graded adjudication runs. Defaulted False while it ran *alongside* the threshold
    # rules and was the experimental path; with those rules deleted, False now means "compute the
    # evidence and then judge nothing", so a default run would apply detection floors and hand
    # every real question downstream unanswered. That is not a conservative default, it is an
    # inert one, so this is on.
    #
    # The bars below are uncalibrated on purpose (see the frozen architecture spec: calibration
    # is read off the figures, never guessed). Turning the switch on does not make them more or
    # less calibrated than a config that sets `enabled: true` by hand — which every real config
    # already did — it only stops the default from being a QC stage that decides nothing.
    enabled: bool = True

    # Bars on family severity. Severity is `z / (z + half_severity_z)` where z is a robust
    # deviation from the healthy mode, so every bar converts to a number of deviations:
    #
    #     bar    0.50   0.60   0.667   0.75   0.80   0.90
    #     z       3.0    4.5    6.0     9.0   12.0   27.0
    #
    # Getting that wrong is not a small error. These bars were first carried over from a
    # relative severity scale, where 0.80 sat near the top of the observed range; on the
    # absolute scale it means 12 deviations, which capture, nuclear and multiplet severity
    # never reach. Quarantine then required two severe families when only one could ever
    # qualify, and a 201,923-cell cohort produced 12 quarantined cells.
    #
    # Calibrated against that cohort, the share of real cells each bar flags per family:
    #
    #     bar      capture   metabolic   multiplet   nuclear
    #     0.50       1.93%     10.77%       4.24%     4.71%
    #     0.667      0.03%      9.69%       1.54%     0.56%
    #     0.80       0.00%      8.90%       0.00%     0.02%
    concern_severity: float = 0.50

    # 6 deviations — genuinely severe, and low enough that capture, nuclear and multiplet
    # can still participate in concordance rather than metabolic deciding alone.
    severe_severity: float = 0.667

    min_concordant_families: int = 2

    # ~27 deviations. Correctly extreme: this route claims the barcode carries no usable
    # information at all, not that the cell is damaged.
    uninformative_capture_severity: float = 0.90

    min_coverage_for_quarantine: float = 0.50
    multiplet_severity: float = 0.60
    nuclear_axis_applicable: bool = True

    # ── Lineage-conditional severity ──────────────────────────────────────────────── #
    #
    # Judge a cell against cells of its own kind, not against the whole library. This is not
    # a refinement; without it the system deletes rare cell types outright.
    #
    # Measured: a synthetic cohort of 950 ordinary cells plus 50 perfectly healthy cells whose
    # constitutive biology is low-complexity and high-mitochondrial (the neutrophil /
    # erythrocyte / plasma-cell profile) quarantined 50 of 50 rare cells and 0 of 950 ordinary
    # ones. Two independent routes fired: concordance (capture 0.946 + metabolic 0.974 = two
    # severe families) and the uninformative-barcode route (capture >= 0.90 alone). Against a
    # sample-wide null a rare cell type and a dying cell are geometrically identical, so no
    # threshold separates them. On the 201,923-cell validation cohort 2,091 cells carry that
    # signature and every one is barred from fitting.
    #
    # On by default because the failure it prevents is silent and destroys the discovery.
    lineage_conditional: bool = True

    # Coarse on purpose. Splitting one lineage in two costs almost nothing; merging two
    # weakens their nulls slightly. Both are far better than one null per library.
    lineage_resolution: float = 0.5

    # Smallest group that can support its own null. Below this a cell borrows a coarser
    # level — lineage across libraries, then library, then pooled — which widens the null and
    # so lowers severity. Falling back is the conservative direction.
    lineage_min_cells: int = 25

    # Absolute gene floor for taking part in the provisional grouping. Not a cohort
    # statistic: a barcode with almost no genes cannot be a rare cell type, and letting it
    # in would allow true empties to anchor a group.
    lineage_min_genes: int = 50

    # Bars for the per-lineage audit, which catches what per-cell conditioning cannot: a
    # lineage that is uniformly damaged is exonerated by its own uniformity, because every
    # cell looks ordinary next to neighbours that are also debris.
    lineage_suspect_severity: float = 0.667
    lineage_vulnerable_fraction: float = 0.50

    # ── Archetype audit (optional) ─────────────────────────────────────────────────── #
    #
    # Asks the one question the automated verdict cannot ask about itself: is there a
    # coherent population being removed? Archetypes are polytope vertices rather than dense
    # blobs, so unlike Leiden they do not need a population to be numerous to find it.
    #
    # Runs through an isolated environment because partipy is GPL-3 and CellQuorum is
    # BSD-3. Without that environment the audit reports itself unavailable and the run is
    # unaffected, so leaving this on by default costs nothing.
    archetype_audit: bool = True
    archetype_max: int = 10
    archetype_bootstrap: int = 0

    # Cells entering the archetype fit. Capped because archetypal analysis solves a
    # nonnegative least-squares problem per cell per iteration: uncapped on the
    # 201,923-cell cohort it sat at 0% CPU for 27 minutes and blocked the run. A uniform
    # subsample keeps exclusion rates unbiased and still holds ~2,000 cells of a population
    # at 10% frequency.
    archetype_max_cells: int = 10_000

    # Restarts per candidate count. The selection sweep is one fit per candidate, so this
    # multiplies the entire sweep.
    archetype_restarts: int = 1

    # Hard subprocess cap. An audit must never be able to hang a run; exceeding this
    # degrades to "audit unavailable" and the run continues.
    archetype_timeout_seconds: int = 900

    # ── Self-check ────────────────────────────────────────────────────────────────── #
    #
    # Compare the verdict against the evidence it rests on, and stop rather than emit a
    # plausible wrong answer. On by default because every defect in this area was found by a
    # human asking a question rather than by a test — a rescaled posterior that moved 22,541
    # cells, a fallback null that dropped damage detection from 100% to 10%, an audit that
    # called a doublet cluster a lost population. All shipped with a green suite.
    self_check: bool = True

    # Whether a failed check stops the run. False downgrades to warnings, which is the right
    # setting for deliberately degraded input and the wrong one for anything else.
    self_check_fails_run: bool = True

    # Core fraction below which the run is questioned. Graded QC assigns permissions rather
    # than deleting, so a cohort whose manifold is defined by a minority may be correct — but
    # never silently.
    self_check_minimum_core: float = 0.50


class QCConfig(StrictBaseModel):
    """
    Store full QC module configuration.

    This model controls the first real CellQuorum analysis module. It separates
    reporting from filtering, keeps default filtering permissive, and makes
    doublet and ambient RNA behavior explicit rather than hidden.

    Args:
        enabled: Whether the QC module is enabled.
        metrics: QC metric calculation settings.
        floors: Absolute floors, the only place a barcode leaves the object.
        graded: Graded-adjudication settings (evidence -> core/borderline/quarantine).
        mito_mixture: Mixture-model (miQC) mitochondrial QC settings.
        features: Feature family pattern settings.
        doublets: Doublet detection settings.
        cell_cycle: Cell-cycle scoring settings.
        ambient: Ambient RNA assessment settings.
        duplicate_names: Duplicate name handling settings.
        attrition_audit: Differential-attrition audit settings -- whether QC
            removed cells at the same rate in every arm of the design.
        outputs: QC output settings.
        fail_on_empty_result: Whether filtering to zero cells or genes is fatal.
    """

    enabled: bool = True
    metrics: QCMetricCalculationConfig = Field(default_factory=QCMetricCalculationConfig)

    # Store the absolute floors. There is no `mode` beside this: the floors always remove what
    # they match, graded adjudication never removes anything, and a run that should drop nothing
    # sets all three floors to null. A `mode` switch on top of that could only contradict one of
    # the two, which is what the old `flag_no_drop` default did — it made QC report without acting.
    floors: QCFloorConfig = Field(default_factory=QCFloorConfig)
    graded: QCGradedConfig = Field(default_factory=QCGradedConfig)
    mito_mixture: QCMitoMixtureConfig = Field(default_factory=QCMitoMixtureConfig)
    features: QCFeaturePatternConfig = Field(default_factory=QCFeaturePatternConfig)
    doublets: QCDoubletConfig = Field(default_factory=QCDoubletConfig)
    cell_cycle: QCCellCycleConfig = Field(default_factory=QCCellCycleConfig)
    ambient: QCAmbientRNAConfig = Field(default_factory=QCAmbientRNAConfig)
    duplicate_names: QCDuplicateNameConfig = Field(default_factory=QCDuplicateNameConfig)
    attrition_audit: QCAttritionAuditConfig = Field(default_factory=QCAttritionAuditConfig)
    outputs: QCOutputConfig = Field(default_factory=QCOutputConfig)
    fail_on_empty_result: bool = True

    #: Keys the v1 threshold path owned, mapped to what replaces them. Kept as a real error
    #: rather than silent acceptance: a config that still says `max_mito_percent: 8.0` was
    #: written expecting a hard ceiling, and quietly ignoring it would change that run's
    #: results without telling anyone.
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
        """Fail with a migration message when a config still sets a v1 threshold key.

        Args:
            data: Raw config mapping, before field validation.

        Returns:
            The mapping unchanged when it uses no removed key.

        Raises:
            CellQuorumConfigError: If a removed v1 key is present.
        """
        if not isinstance(data, Mapping):
            return data
        for key, guidance in cls._REMOVED_KEYS.items():
            if key in data:
                raise CellQuorumConfigError(f"`qc.{key}` was removed. {guidance}")
        return data

    def enabled_metric_families(self) -> list[str]:
        """
        Return enabled QC metric families.

        Returns:
            Ordered list of enabled QC metric family labels.
        """

        # Initialize the enabled metric family list.
        families = ["floors"]

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
        # Derived from the model rather than restated: the hand-maintained list this replaces
        # had drifted and was silently rejecting `graded`, `mito_mixture` and `attrition_audit`
        # — three live sub-configs — because nobody updated it when they landed.
        #
        # The removed v1 keys are allowed *through* this gate on purpose, so that
        # `_reject_removed_threshold_keys` can answer with a migration message instead of this
        # function reporting them as an unrecognised name.
        allowed_keys=[*QCConfig.model_fields, *QCConfig._REMOVED_KEYS],
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
    "QCFloorConfig",
    "AmbientMethod",
    "DoubletMethod",
    "DuplicateNamePolicy",
    "QCAmbientRNAConfig",
    "QCAttritionAuditConfig",
    "QCCellCycleConfig",
    "QCConfig",
    "QCDoubletConfig",
    "QCDuplicateNameConfig",
    "QCFigureFormat",
    "QCFeaturePatternConfig",
    "QCMetricCalculationConfig",
    "QCOutputConfig",
    "validate_qc_config_dict",
]
