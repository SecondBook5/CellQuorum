"""QC evidence measurement, family aggregation, and graded adjudication.

Read top to bottom; the file is ordered the way the logic flows.

    1. Vocabulary      what the words mean: families, availability, direction
    2. One axis        a single measurement's severity, qualified by availability
    3. All axes        the table, rolled up from axes to families
    4. Adjudication    families -> core | borderline | quarantine, with reasons

Three invariants shape everything, each from a real failure.

**Absent evidence is not evidence of health.** If "unmeasured" reads as "normal", the
system gets more permissive exactly where it knows least. Availability is therefore a
first-class per-axis value with five states, and severity is blanked to NaN wherever it
is not usable.

**Concordance is across families, not metrics.** Total UMI and detected genes are nearly
the same measurement; counting them as two hits manufactures corroboration, and is how a
small quiescent cell gets condemned twice for being small once. Axes roll up to their
family before anything counts hits.

**No single statistical model may condemn a cell.** A mitochondrial mixture posterior of
0.96 describes a fitted distribution, not a membrane. The system this replaces removed
cells on exactly that basis: one fixed mitochondrial ceiling accounted for essentially
all of its removals, and the cells it removed had normal complexity on every other axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc.lineage import NullGrouping
from cellquorum.stages.qc.validation import resolve_qc_matrix


class QCEvidenceError(CellQuorumDataError):
    """Malformed evidence construction — a producer bug, not a data condition."""


class QCAdjudicationError(CellQuorumDataError):
    """An invalid adjudication policy."""


class EvidenceFamily(StrEnum):
    """Independent axes of technical evidence. Membership defines what corroborates."""

    CAPTURE_COMPLEXITY = "capture_complexity"
    NUCLEAR_INTEGRITY = "nuclear_integrity"
    METABOLIC_STRESS = "metabolic_stress"
    AMBIENT_BACKGROUND = "ambient_background"
    CELL_CALLING = "cell_calling"
    MULTIPLET = "multiplet"


class EvidenceAvailability(StrEnum):
    """Why an axis does or does not inform a given cell."""

    AVAILABLE_VALID = "available_valid"
    UNAVAILABLE_INPUT = "unavailable_input"
    NOT_APPLICABLE = "not_applicable"
    MODEL_UNSTABLE = "model_unstable"
    COMPUTATION_FAILED = "computation_failed"

    @property
    def is_usable(self) -> bool:
        """Whether a severity from this state may be used at all."""
        return self in {EvidenceAvailability.AVAILABLE_VALID, EvidenceAvailability.MODEL_UNSTABLE}


class Direction(StrEnum):
    """Which tail of a metric is concerning."""

    LOWER_TAIL = "lower_tail"
    UPPER_TAIL = "upper_tail"


def _usable_mask(availability: pd.Series) -> pd.Series:
    """Per-cell boolean mask of usability, from a Series of availability values."""
    return availability.map(lambda state: EvidenceAvailability(state).is_usable).astype(bool)


@dataclass(frozen=True)
class AxisEvidence:
    """Severity in ``[0, 1]`` for one measurement axis, with its availability.

    Args:
        name: Metric name, matching the column that produced it.
        family: Evidence family this axis belongs to.
        direction: Concerning tail of the underlying metric.
        severity: Per-cell severity in ``[0, 1]``, NaN where not usable.
        availability: Per-cell :class:`EvidenceAvailability`, sharing severity's index.
        weight: Family-aggregation multiplier, for down-weighting a shaky fit.
        value: The raw measurement severity was derived from, when the producer has one to
            give. Carried so a reader can check a severity against the number behind it, and
            so the calibration figures can plot the metric rather than its transform — the
            figure spec puts the per-donor distributions on raw metrics for exactly that
            reason. ``None`` where the axis has no single underlying value.

    Raises:
        QCEvidenceError: On index mismatch, out-of-range severity, a usable NaN severity,
            or a non-positive weight.
    """

    name: str
    family: EvidenceFamily
    direction: Direction
    severity: pd.Series
    availability: pd.Series
    weight: float = 1.0
    value: pd.Series | None = None

    def __post_init__(self) -> None:
        if not self.severity.index.equals(self.availability.index):
            raise QCEvidenceError(
                f"Axis {self.name!r}: severity and availability must share an index."
            )

        contradictory = self.usable_mask() & self.severity.isna()
        if bool(contradictory.any()):
            raise QCEvidenceError(
                f"Axis {self.name!r}: {int(contradictory.sum())} cells are marked usable but "
                "have NaN severity. Mark them COMPUTATION_FAILED instead."
            )

        usable_values = self.severity[self.usable_mask()]
        if len(usable_values):
            low, high = float(usable_values.min()), float(usable_values.max())
            if low < 0.0 or high > 1.0:
                raise QCEvidenceError(
                    f"Axis {self.name!r}: severity must lie in [0, 1], got [{low:.3f}, {high:.3f}]."
                )

        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise QCEvidenceError(f"Axis {self.name!r}: weight must be positive.")

    def usable_mask(self) -> pd.Series:
        """Per-cell mask of whether this axis carries usable information."""
        return _usable_mask(self.availability)

    def effective_severity(self) -> pd.Series:
        """Weighted severity, NaN where not usable."""
        return (self.severity * self.weight).where(self.usable_mask()).clip(upper=1.0)


def build_axis(
    *,
    name: str,
    family: EvidenceFamily,
    direction: Direction,
    severity: pd.Series,
    availability: EvidenceAvailability | pd.Series,
    weight: float = 1.0,
    value: pd.Series | None = None,
) -> AxisEvidence:
    """Construct an :class:`AxisEvidence`, broadcasting a scalar availability.

    Args:
        name: Metric name.
        family: Evidence family.
        direction: Concerning tail.
        severity: Per-cell severity in ``[0, 1]``.
        availability: One state for every cell, or a per-cell Series.
        weight: Family-aggregation weight.
        value: Optional raw measurement behind the severity.
    """
    if isinstance(availability, EvidenceAvailability):
        availability = pd.Series(str(availability), index=severity.index, dtype=object)
    else:
        availability = availability.astype(str)

    severity = severity.where(_usable_mask(availability), other=np.nan)

    return AxisEvidence(
        name=name,
        family=family,
        direction=direction,
        severity=severity,
        availability=availability,
        weight=weight,
        value=None if value is None else value.reindex(severity.index).astype(float),
    )


@dataclass(frozen=True)
class EvidenceTable:
    """Every evidence axis for one dataset, rolled up to family level.

    Args:
        axes: Evidence axes, all sharing one cell index.
        obs_names: The cell index every axis is aligned to.

    Raises:
        QCEvidenceError: If no axes are given, or an axis is misaligned.
    """

    axes: tuple[AxisEvidence, ...]
    obs_names: pd.Index

    DAMAGE_FAMILIES: ClassVar[tuple[EvidenceFamily, ...]] = (
        EvidenceFamily.CAPTURE_COMPLEXITY,
        EvidenceFamily.NUCLEAR_INTEGRITY,
        EvidenceFamily.METABOLIC_STRESS,
        EvidenceFamily.AMBIENT_BACKGROUND,
    )

    def __post_init__(self) -> None:
        if not self.axes:
            raise QCEvidenceError("EvidenceTable requires at least one axis.")
        for axis in self.axes:
            if not axis.severity.index.equals(self.obs_names):
                raise QCEvidenceError(f"Axis {axis.name!r} is not aligned to obs_names.")

    def families_present(self) -> tuple[EvidenceFamily, ...]:
        """Families with at least one axis, in canonical order for stable columns."""
        present = {axis.family for axis in self.axes}
        return tuple(family for family in EvidenceFamily if family in present)

    def _axes_in(self, family: EvidenceFamily) -> list[AxisEvidence]:
        """Axes belonging to one family."""
        return [axis for axis in self.axes if axis.family is family]

    def family_severity(self) -> pd.DataFrame:
        """Per-cell severity per family; NaN only where no axis in it was usable."""
        severity_by_family = {
            str(family): pd.concat(
                [axis.effective_severity() for axis in self._axes_in(family)], axis=1
            ).max(axis=1, skipna=True)
            for family in self.families_present()
        }
        return pd.DataFrame(severity_by_family, index=self.obs_names)

    def family_usable(self) -> pd.DataFrame:
        """Per-cell boolean of whether each family carried any usable axis."""
        usable_by_family = {
            str(family): pd.concat(
                [axis.usable_mask() for axis in self._axes_in(family)], axis=1
            ).any(axis=1)
            for family in self.families_present()
        }
        return pd.DataFrame(usable_by_family, index=self.obs_names)

    def damage_family_severity(self) -> pd.DataFrame:
        """Family severity restricted to :attr:`DAMAGE_FAMILIES`."""
        damage_columns = [
            str(family) for family in self.families_present() if family in self.DAMAGE_FAMILIES
        ]
        return self.family_severity()[damage_columns]

    def evidence_coverage(self) -> pd.Series:
        """Fraction of *present* families that were measurable, per cell."""
        usable = self.family_usable()
        if usable.shape[1] == 0:
            return pd.Series(0.0, index=self.obs_names, dtype=float)
        return usable.sum(axis=1).astype(float) / float(usable.shape[1])

    def concordant_family_count(self, *, min_severity: float) -> pd.Series:
        """Count families reaching ``min_severity``, per cell.

        Args:
            min_severity: Bar in ``[0, 1]``. Caller-supplied; there is no default.

        Raises:
            QCEvidenceError: If ``min_severity`` is outside ``[0, 1]``.
        """
        if not 0.0 <= min_severity <= 1.0:
            raise QCEvidenceError(f"min_severity must lie in [0, 1], got {min_severity}.")
        return (self.family_severity() >= min_severity).sum(axis=1).astype(int)

    def to_obs_frame(self) -> pd.DataFrame:
        """Flatten to ``adata.obs`` columns, prefixed ``qc_ev_``."""
        columns: dict[str, pd.Series] = {}
        for axis in self.axes:
            columns[f"qc_ev_{axis.name}_severity"] = axis.severity
            columns[f"qc_ev_{axis.name}_availability"] = axis.availability.astype(str)

            if axis.value is not None:
                columns[f"qc_ev_{axis.name}_value"] = axis.value
        for family, severity in self.family_severity().items():
            columns[f"qc_ev_family_{family}_severity"] = severity
        for family, usable in self.family_usable().items():
            columns[f"qc_ev_family_{family}_usable"] = usable
        columns["qc_evidence_coverage"] = self.evidence_coverage()
        return pd.DataFrame(columns, index=self.obs_names)


SUPPORTING_FAMILIES: frozenset[EvidenceFamily] = frozenset({EvidenceFamily.METABOLIC_STRESS})


class QCStateInitial(StrEnum):
    """Provisional state assigned before any biological reference exists."""

    CORE = "core"

    BORDERLINE = "borderline"

    QUARANTINE = "quarantine"


class AdjudicationReason(StrEnum):
    """Which rule produced a cell's state, recorded per cell for auditability."""

    NO_CONCERN = "no_concern"
    UNINFORMATIVE_BARCODE = "uninformative_barcode"
    CONCORDANT_SEVERE_DAMAGE = "concordant_severe_damage"
    SINGLE_FAMILY_CONCERN = "single_family_concern"
    SUPPORTING_EVIDENCE_ONLY = "supporting_evidence_only"
    WITHHELD_LOW_COVERAGE = "withheld_low_coverage"
    PROBABLE_MULTIPLET = "probable_multiplet"


@dataclass(frozen=True)
class AdjudicationPolicy:
    """Calibrated bars controlling the initial adjudication.

    Args:
        concern_severity: Family severity at or above which a family is *concerning*, so
            the cell is at least borderline.
        severe_severity: Family severity at or above which a family is *severe*. Only
            severe families feed the concordance route to quarantine.
        min_concordant_families: How many independent damage families must be severe
            before quarantine is justified. Must be at least 2.
        uninformative_capture_severity: Capture severity at or above which the barcode
            carries no usable information, justifying quarantine on its own.
        min_coverage_for_quarantine: Evidence coverage below which quarantine is withheld
            in favour of borderline.
        multiplet_severity: Multiplet severity at or above which a cell is flagged a
            probable multiplet. Recorded separately; never quarantines.

    Raises:
        QCAdjudicationError: If a bar is outside ``[0, 1]``, if ``severe_severity`` is
            below ``concern_severity``, or if ``min_concordant_families`` is below 2.
    """

    concern_severity: float
    severe_severity: float
    min_concordant_families: int
    uninformative_capture_severity: float
    min_coverage_for_quarantine: float
    multiplet_severity: float

    def __post_init__(self) -> None:
        for field_name in (
            "concern_severity",
            "severe_severity",
            "uninformative_capture_severity",
            "min_coverage_for_quarantine",
            "multiplet_severity",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise QCAdjudicationError(f"{field_name} must lie in [0, 1], got {value}.")

        if self.severe_severity < self.concern_severity:
            raise QCAdjudicationError(
                f"severe_severity ({self.severe_severity}) must be >= concern_severity "
                f"({self.concern_severity})."
            )

        if self.min_concordant_families < 2:
            raise QCAdjudicationError(
                f"min_concordant_families must be >= 2, got {self.min_concordant_families}. "
                "One family would let a single statistical model condemn a cell."
            )


@dataclass(frozen=True)
class AdjudicationResult:
    """Per-cell initial states with the reasoning that produced them.

    Args:
        state: :class:`QCStateInitial` per cell.
        reason: :class:`AdjudicationReason` per cell.
        concerning_families: Damage families at or above the concern bar.
        severe_families: Damage families at or above the severe bar.
        primary_driver: Highest-severity concerning damage family, or ``""``. Emitted
            here because families overlap, so no consumer can honestly infer a primary
            cause from the flags alone.
        probable_multiplet: Whether multiplet severity reached its bar.
        coverage: Evidence coverage, carried through for reporting.
        confidence: Heuristic decision confidence in ``[0, 1]``.
    """

    state: pd.Series
    reason: pd.Series
    concerning_families: pd.Series
    severe_families: pd.Series
    primary_driver: pd.Series
    probable_multiplet: pd.Series
    coverage: pd.Series
    confidence: pd.Series

    def counts(self) -> dict[str, int]:
        """Cell count per state, including zero-count states for stable columns."""
        observed = self.state.value_counts()
        return {str(state): int(observed.get(str(state), 0)) for state in QCStateInitial}

    def to_obs_frame(self) -> pd.DataFrame:
        """Flatten to ``adata.obs`` columns."""
        return pd.DataFrame(
            {
                "qc_state_initial": self.state.astype(str),
                "qc_state_reason": self.reason.astype(str),
                "qc_concerning_families": self.concerning_families,
                "qc_severe_families": self.severe_families,
                "qc_primary_driver": self.primary_driver.astype(str),
                "qc_probable_multiplet": self.probable_multiplet,
                "qc_evidence_coverage": self.coverage,
                "qc_confidence": self.confidence,
            },
            index=self.state.index,
        )


def adjudicate_initial(
    evidence: EvidenceTable,
    policy: AdjudicationPolicy,
    *,
    called_doublets: pd.Series | None = None,
) -> AdjudicationResult:
    """Assign a provisional QC state to every cell.

    Args:
        evidence: Graded evidence for the dataset.
        policy: Configured concern and exclusion thresholds.
        called_doublets: Boolean detector-consensus calls indexed by every input cell.
            These retain their exclusion even when score-based evidence is weak.
    """
    cells = evidence.obs_names
    damage_severity = evidence.damage_family_severity()
    coverage = evidence.evidence_coverage()

    is_concerning = damage_severity >= policy.concern_severity
    is_severe = damage_severity >= policy.severe_severity
    n_concerning_families = is_concerning.sum(axis=1).astype(int)
    n_severe_families = is_severe.sum(axis=1).astype(int)

    establishing_columns = [
        column
        for column in damage_severity.columns
        if EvidenceFamily(column) not in SUPPORTING_FAMILIES
    ]
    n_severe_establishing = is_severe[establishing_columns].sum(axis=1)

    capture_column = str(EvidenceFamily.CAPTURE_COMPLEXITY)
    is_uninformative = (
        (damage_severity[capture_column] >= policy.uninformative_capture_severity).fillna(False)
        if capture_column in damage_severity.columns
        else pd.Series(False, index=cells)
    )

    is_concordant = (n_severe_families >= policy.min_concordant_families) & (
        n_severe_establishing >= 1
    )

    has_enough_coverage = coverage >= policy.min_coverage_for_quarantine
    meets_a_route = is_uninformative | is_concordant
    is_quarantined = meets_a_route & has_enough_coverage
    is_withheld = meets_a_route & ~has_enough_coverage

    state = pd.Series(str(QCStateInitial.CORE), index=cells, dtype=object)
    state[n_concerning_families > 0] = str(QCStateInitial.BORDERLINE)
    state[is_withheld] = str(QCStateInitial.BORDERLINE)
    state[is_quarantined] = str(QCStateInitial.QUARANTINE)

    reason = pd.Series(str(AdjudicationReason.NO_CONCERN), index=cells, dtype=object)
    supporting_columns = [
        column
        for column in damage_severity.columns
        if EvidenceFamily(column) in SUPPORTING_FAMILIES
    ]
    if supporting_columns:
        n_supporting_concerns = is_concerning[supporting_columns].sum(axis=1)
        concerning_only_on_support = (n_concerning_families > 0) & (
            n_supporting_concerns == n_concerning_families
        )
        reason[concerning_only_on_support] = str(AdjudicationReason.SUPPORTING_EVIDENCE_ONLY)
        reason[(n_concerning_families > 0) & ~concerning_only_on_support] = str(
            AdjudicationReason.SINGLE_FAMILY_CONCERN
        )
    else:
        reason[n_concerning_families > 0] = str(AdjudicationReason.SINGLE_FAMILY_CONCERN)
    reason[is_withheld] = str(AdjudicationReason.WITHHELD_LOW_COVERAGE)
    reason[is_quarantined & is_concordant] = str(AdjudicationReason.CONCORDANT_SEVERE_DAMAGE)
    reason[is_quarantined & is_uninformative] = str(AdjudicationReason.UNINFORMATIVE_BARCODE)

    all_severity = evidence.family_severity()
    multiplet_column = str(EvidenceFamily.MULTIPLET)
    is_probable_multiplet = (
        (all_severity[multiplet_column] >= policy.multiplet_severity).fillna(False)
        if multiplet_column in all_severity.columns
        else pd.Series(False, index=cells)
    )
    if called_doublets is not None:
        if (
            not pd.api.types.is_bool_dtype(called_doublets.dtype)
            or not called_doublets.index.is_unique
            or len(called_doublets) != len(cells)
            or not cells.isin(called_doublets.index).all()
        ):
            raise QCEvidenceError("Doublet calls must be boolean and cover every evidence cell.")
        is_probable_multiplet |= called_doublets.reindex(cells).fillna(False).astype(bool)

    reason[is_probable_multiplet & (n_concerning_families == 0) & ~is_quarantined] = str(
        AdjudicationReason.PROBABLE_MULTIPLET
    )

    has_concern = n_concerning_families > 0
    concerning_severity = damage_severity.where(is_concerning)
    primary_driver = pd.Series("", index=cells, dtype=object)
    if concerning_severity.shape[1] and bool(has_concern.any()):
        driver = concerning_severity.loc[has_concern].idxmax(axis=1).astype(object)
        primary_driver.loc[has_concern] = driver

    return AdjudicationResult(
        state=state,
        reason=reason,
        concerning_families=n_concerning_families,
        severe_families=n_severe_families,
        primary_driver=primary_driver,
        probable_multiplet=is_probable_multiplet,
        coverage=coverage,
        confidence=_decision_confidence(damage_severity, coverage, policy),
    )


def _decision_confidence(
    damage_severity: pd.DataFrame,
    coverage: pd.Series,
    policy: AdjudicationPolicy,
) -> pd.Series:
    """Heuristic confidence in a cell's adjudication, in ``[0, 1]``."""
    strongest_severity = damage_severity.max(axis=1, skipna=True)
    concern_bar = policy.concern_severity

    bar_span = max(concern_bar, 1.0 - concern_bar) or 1.0
    distance_from_bar = ((strongest_severity - concern_bar).abs() / bar_span).fillna(0.0)

    return (coverage * distance_from_bar.clip(0.0, 1.0)).clip(0.0, 1.0)


if TYPE_CHECKING:
    from anndata import AnnData


DEFAULT_HALF_SEVERITY_Z = 3.0


MAD_TO_SIGMA = 1.4826


QUANTILE_SCALE_FALLBACKS: tuple[tuple[float, float], ...] = (
    (0.75, 0.6745),
    (0.90, 1.2816),
    (0.99, 2.3263),
)


MIN_CELLS_FOR_NULL = 25


def _saturating_severity(z: pd.Series, half_severity_z: float) -> pd.Series:
    """Map a one-sided robust z to severity in ``[0, 1)``.

    Args:
        z: One-sided robust z; negative means "not concerning".
        half_severity_z: z at which severity is 0.5.
    """
    positive = z.clip(lower=0.0)
    return positive / (positive + half_severity_z)


@dataclass(frozen=True)
class RobustNull:
    """Per-cell location and scale of the healthy mode of its group.

    Args:
        location: Group median, broadcast per cell.
        scale: Robust sigma of the group's healthy mode; NaN where inestimable.
    """

    location: pd.Series
    scale: pd.Series

    def z(self, values: pd.Series, *, direction: Direction) -> pd.Series:
        """One-sided robust z, oriented so positive always means "concerning"."""
        signed = (values - self.location) / self.scale
        return signed if direction is Direction.UPPER_TAIL else -signed


def fit_robust_null(values: pd.Series, groups: pd.Series | None) -> RobustNull:
    """Estimate the healthy mode's location and scale, per group.

    Args:
        values: The metric, per cell.
        groups: Grouping to fit within, normally the cohort sample key. None pools, which
            is only correct for a single library.
    """
    if groups is None:
        groups = pd.Series("__pooled__", index=values.index)

    grouped = values.groupby(groups, observed=True)
    location = grouped.transform("median")

    absolute_deviation = (values - location).abs()
    scale = absolute_deviation.groupby(groups, observed=True).transform("median") * MAD_TO_SIGMA

    for quantile, normal_z in QUANTILE_SCALE_FALLBACKS:
        if bool((scale > 0).all()):
            break
        upper: pd.Series = grouped.transform(
            lambda group, q=quantile: float(group.quantile(q))  # type: ignore[arg-type,return-value]
        )
        scale = scale.where(scale > 0, (upper - location) / normal_z)

    n_cells = grouped.transform("size")
    scale = scale.where((scale > 0) & (n_cells >= MIN_CELLS_FOR_NULL))

    return RobustNull(location=location, scale=scale)


def tail_severity(
    values: pd.Series,
    groups: pd.Series | None,
    *,
    direction: Direction,
    log_scale: bool = False,
    half_severity_z: float = DEFAULT_HALF_SEVERITY_Z,
) -> pd.Series:
    """Severity for one metric: robust z against its healthy mode, saturated.

    Args:
        values: The metric, per cell.
        groups: Grouping to fit the null within.
        direction: Which tail is concerning.
        log_scale: Fit the null on ``log1p`` values. Correct for count-like metrics, whose
            healthy mode is right-skewed on the raw scale, so a symmetric robust z there
            would systematically over-flag the low side.
        half_severity_z: z at which severity is 0.5.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype(float)

    prepared = (
        pd.Series(np.log1p(numeric.clip(lower=0.0).to_numpy()), index=numeric.index, dtype=float)
        if log_scale
        else numeric
    )

    null = fit_robust_null(prepared, groups)
    return _saturating_severity(null.z(prepared, direction=direction), half_severity_z)


def nested_tail_severity(
    values: pd.Series,
    grouping: NullGrouping,
    *,
    direction: Direction,
    log_scale: bool = False,
    half_severity_z: float = DEFAULT_HALF_SEVERITY_Z,
) -> pd.Series:
    """Severity where each cell is scored against the reference class it was assigned.

    Args:
        values: The metric, per cell.
        grouping: Level assignment plus each level's keys for every cell.
        direction: Which tail is concerning.
        log_scale: Fit the null on ``log1p`` values, for count-like metrics.
        half_severity_z: z at which severity is 0.5.

    Returns:
        Severity per cell, NaN where the assigned level could not support a null.
    """
    severity = pd.Series(np.nan, index=values.index, dtype=float)
    for level in grouping.levels_used():
        assigned = (grouping.level == level).reindex(values.index, fill_value=False)
        if not bool(assigned.any()):
            continue
        at_level = tail_severity(
            values,
            grouping.keys[level].reindex(values.index),
            direction=direction,
            log_scale=log_scale,
            half_severity_z=half_severity_z,
        )
        severity[assigned] = at_level[assigned]
    return severity


def axis_from_severity(
    *,
    name: str,
    family: EvidenceFamily,
    direction: Direction,
    severity: pd.Series,
    weight: float = 1.0,
    value: pd.Series | None = None,
) -> AxisEvidence:
    """Build an axis whose availability follows from whether a severity was produced."""
    usable = severity.notna()
    return build_axis(
        name=name,
        family=family,
        direction=direction,
        severity=severity,
        availability=pd.Series(
            np.where(
                usable,
                str(EvidenceAvailability.AVAILABLE_VALID),
                str(EvidenceAvailability.COMPUTATION_FAILED),
            ),
            index=severity.index,
        ),
        weight=weight,
        value=value,
    )


NUCLEAR_RETAINED_GENE = "MALAT1"


DISSOCIATION_STRESS_GENES: tuple[str, ...] = (
    "FOS",
    "FOSB",
    "JUN",
    "JUNB",
    "JUND",
    "EGR1",
    "ATF3",
    "IER2",
    "HSPA1A",
    "HSPA1B",
    "HSPB1",
    "HSPH1",
    "DNAJB1",
    "DNAJA1",
    "SOCS3",
    "ZFP36",
    "DUSP1",
    "KLF6",
    "NR4A1",
    "PPP1R15A",
)


DOUBLET_SCORE_COLUMNS: tuple[str, ...] = (
    "doublet_score_scdblfinder",
    "doublet_score_scrublet",
    "doublet_score",
)


_CAPTURE_METRICS: tuple[str, ...] = ("n_genes_by_counts", "total_counts")


def gene_fraction(
    adata: AnnData,
    genes: tuple[str, ...],
    total: pd.Series,
    *,
    layer: str | None = None,
    use_raw: bool = False,
) -> pd.Series | None:
    """Fraction of a cell's counts falling in ``genes``, or None if none are present."""
    matrix, _ = resolve_qc_matrix(adata, layer=layer, use_raw=use_raw)
    names = adata.raw.var_names if use_raw else adata.var_names
    selected = names.isin(genes)
    if not selected.any():
        return None
    indicator = selected.astype(np.float64).reshape(-1, 1)
    summed = np.asarray(matrix @ indicator).ravel()

    fraction = pd.Series(summed, index=adata.obs_names, dtype=float).reindex(total.index) / total
    return fraction.replace([np.inf, -np.inf], np.nan)


def multiplet_agreement_severity(
    obs: pd.DataFrame,
    groups: pd.Series | None,
    *,
    half_severity_z: float = DEFAULT_HALF_SEVERITY_Z,
) -> pd.Series | None:
    """Multiplet severity requiring detectors to agree, on comparable scales."""
    present = [column for column in DOUBLET_SCORE_COLUMNS if column in obs.columns]

    present = [column for column in present if not obs[column].isna().all()]
    if not present:
        return None

    if len(present) > 1 and "doublet_score" in present:
        present = [column for column in present if column != "doublet_score"]

    severities = [
        tail_severity(
            obs[column],
            groups,
            direction=Direction.UPPER_TAIL,
            half_severity_z=half_severity_z,
        )
        for column in present
    ]

    return pd.concat(severities, axis=1).min(axis=1, skipna=False)


def build_evidence_table(
    adata: AnnData,
    cell_metrics: pd.DataFrame,
    *,
    group_key: str | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    mito_posterior: pd.Series | None = None,
    nuclear_axis_applicable: bool = True,
    half_severity_z: float = DEFAULT_HALF_SEVERITY_Z,
    grouping: NullGrouping | None = None,
    lineage_conditional: bool = False,
    expression_adata: AnnData | None = None,
) -> EvidenceTable:
    """Assemble every evidence axis this dataset supports.

    Args:
        adata: The QC AnnData, used for gene-level axes.
        cell_metrics: Per-cell QC metrics, indexed like ``adata.obs``.
        group_key: ``obs`` column to fit nulls within, normally the cohort sample key. Used
            only when ``grouping`` is not supplied.
        layer: Layer holding the counts the gene-fraction axes should use.
        use_raw: Read gene fractions from raw.X.
        expression_adata: Unfiltered count source matching the metric denominator.
            Observation metadata and doublet scores still come from adata.
        mito_posterior: Per-cell compromised probability from the mixture model. Used
            directly under the mixture model assumptions, without robust-tail rescaling.
            This is a model posterior, not empirical calibration against damage labels.
        nuclear_axis_applicable: False for single-nucleus assays.
        half_severity_z: Robust z at which severity is 0.5.
        grouping: Per-cell reference classes from
            :func:`cellquorum.stages.qc.lineage.resolve_null_groups`, with each level's null
            estimated over every cell at that level. Overrides ``group_key``.
        lineage_conditional: True when ``grouping`` carries cell identity as well as library.
            Recorded for provenance only — it deliberately changes no behaviour here. An earlier
            version used it to re-scale the mitochondrial posterior within lineage, which
            corrupted an already-calibrated probability; see the metabolic axis below. Calibrating
            the posterior for cell identity is the mixture model's job, not this module's.
    """
    expression_adata = adata if expression_adata is None else expression_adata
    obs = adata.obs
    groups = obs[group_key] if group_key and group_key in obs.columns else None

    def severity_of(
        values: pd.Series,
        *,
        direction: Direction,
        log_scale: bool = False,
    ) -> pd.Series:
        """Score one metric against each cell's reference class."""
        if grouping is not None:
            return nested_tail_severity(
                values,
                grouping,
                direction=direction,
                log_scale=log_scale,
                half_severity_z=half_severity_z,
            )
        return tail_severity(
            values,
            groups,
            direction=direction,
            log_scale=log_scale,
            half_severity_z=half_severity_z,
        )

    axes: list[AxisEvidence] = []

    def add(
        name: str,
        family: EvidenceFamily,
        direction: Direction,
        severity: pd.Series,
        weight: float = 1.0,
        value: pd.Series | None = None,
    ) -> None:
        axes.append(
            axis_from_severity(
                name=name,
                family=family,
                direction=direction,
                severity=severity,
                weight=weight,
                value=value,
            )
        )

    for metric in _CAPTURE_METRICS:
        if metric in cell_metrics:
            add(
                metric,
                EvidenceFamily.CAPTURE_COMPLEXITY,
                Direction.LOWER_TAIL,
                severity_of(
                    cell_metrics[metric],
                    direction=Direction.LOWER_TAIL,
                    log_scale=True,
                ),
            )

    if mito_posterior is not None:
        add(
            "mito_mixture_posterior",
            EvidenceFamily.METABOLIC_STRESS,
            Direction.UPPER_TAIL,
            mito_posterior.reindex(cell_metrics.index).astype(float),
        )
    elif "pct_counts_mito" in cell_metrics:
        add(
            "pct_counts_mito",
            EvidenceFamily.METABOLIC_STRESS,
            Direction.UPPER_TAIL,
            severity_of(
                cell_metrics["pct_counts_mito"],
                direction=Direction.UPPER_TAIL,
            ),
        )

    total_counts = cell_metrics.get("total_counts")
    if total_counts is not None:
        total_counts = total_counts.astype(float)

        stress = gene_fraction(
            expression_adata, DISSOCIATION_STRESS_GENES, total_counts, layer=layer, use_raw=use_raw
        )
        if stress is not None:
            add(
                "dissociation_stress",
                EvidenceFamily.METABOLIC_STRESS,
                Direction.UPPER_TAIL,
                severity_of(
                    stress,
                    direction=Direction.UPPER_TAIL,
                ),
                weight=0.6,
                value=stress,
            )

        if nuclear_axis_applicable:
            nuclear = gene_fraction(
                expression_adata,
                (NUCLEAR_RETAINED_GENE,),
                total_counts,
                layer=layer,
                use_raw=use_raw,
            )
            if nuclear is not None:
                add(
                    "malat1_fraction",
                    EvidenceFamily.NUCLEAR_INTEGRITY,
                    Direction.UPPER_TAIL,
                    severity_of(
                        nuclear,
                        direction=Direction.UPPER_TAIL,
                    ),
                    value=nuclear,
                )

    multiplet = multiplet_agreement_severity(obs, groups, half_severity_z=half_severity_z)
    if multiplet is not None:
        add("doublet_agreement", EvidenceFamily.MULTIPLET, Direction.UPPER_TAIL, multiplet)

    return EvidenceTable(axes=tuple(axes), obs_names=pd.Index(adata.obs_names))


__all__ = [
    "AdjudicationPolicy",
    "AdjudicationReason",
    "AdjudicationResult",
    "AxisEvidence",
    "Direction",
    "EvidenceAvailability",
    "EvidenceFamily",
    "EvidenceTable",
    "QCAdjudicationError",
    "QCEvidenceError",
    "QCStateInitial",
    "SUPPORTING_FAMILIES",
    "adjudicate_initial",
    "build_axis",
    "DEFAULT_HALF_SEVERITY_Z",
    "DISSOCIATION_STRESS_GENES",
    "DOUBLET_SCORE_COLUMNS",
    "MIN_CELLS_FOR_NULL",
    "NUCLEAR_RETAINED_GENE",
    "RobustNull",
    "axis_from_severity",
    "build_evidence_table",
    "fit_robust_null",
    "gene_fraction",
    "multiplet_agreement_severity",
    "tail_severity",
]
