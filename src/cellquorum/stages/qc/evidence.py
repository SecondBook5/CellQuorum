# Pipeline step (order=20): qc — graded technical evidence and initial adjudication.
"""Graded QC evidence, and the initial core/borderline/quarantine decision built on it.

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
from typing import ClassVar

import numpy as np
import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError

# ─── 1. Vocabulary ──────────────────────────────────────────────────────────────────


class QCEvidenceError(CellQuorumDataError):
    """Malformed evidence construction — a producer bug, not a data condition."""


class QCAdjudicationError(CellQuorumDataError):
    """An invalid adjudication policy."""


class EvidenceFamily(StrEnum):
    """Independent axes of technical evidence. Membership defines what corroborates.

    Multiplet is tracked here for bookkeeping but is *not* damage: a doublet can be an
    excellent library that simply is not one cell.
    """

    CAPTURE_COMPLEXITY = "capture_complexity"
    NUCLEAR_INTEGRITY = "nuclear_integrity"
    METABOLIC_STRESS = "metabolic_stress"
    AMBIENT_BACKGROUND = "ambient_background"
    CELL_CALLING = "cell_calling"
    MULTIPLET = "multiplet"


class EvidenceAvailability(StrEnum):
    """Why an axis does or does not inform a given cell.

    Five states because they demand different responses. ``NOT_APPLICABLE`` (high
    intronic fraction is expected in single-nucleus data) and ``COMPUTATION_FAILED``
    (should have worked, did not) are the pair most often wrongly collapsed — the second
    is a real blind spot, the first is not.
    """

    AVAILABLE_VALID = "available_valid"
    UNAVAILABLE_INPUT = "unavailable_input"
    NOT_APPLICABLE = "not_applicable"
    MODEL_UNSTABLE = "model_unstable"
    COMPUTATION_FAILED = "computation_failed"

    @property
    def is_usable(self) -> bool:
        """Whether a severity from this state may be used at all.

        ``MODEL_UNSTABLE`` is usable but deserves less weight, which is why
        :class:`AxisEvidence` carries ``weight`` separately.
        """
        return self in {EvidenceAvailability.AVAILABLE_VALID, EvidenceAvailability.MODEL_UNSTABLE}


class Direction(StrEnum):
    """Which tail of a metric is concerning.

    Two-sided bounds on every metric punish real biology: low detected genes may mean
    capture failure, high usually means a doublet or a large cell. Different mechanisms,
    different families — so a two-sided metric is split into two axes.
    """

    LOWER_TAIL = "lower_tail"
    UPPER_TAIL = "upper_tail"


def _usable_mask(availability: pd.Series) -> pd.Series:
    """Per-cell boolean mask of usability, from a Series of availability values."""
    return availability.map(lambda state: EvidenceAvailability(state).is_usable).astype(bool)


# ─── 2. One axis ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AxisEvidence:
    """Severity in ``[0, 1]`` for one measurement axis, with its availability.

    Severity is produced elsewhere — a MAD z-score, a miQC posterior, an ambient model.
    This class does not know how; it only guarantees severity is never readable without
    the availability that qualifies it.

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

        # A usable NaN is the contradiction this model exists to prevent: downstream code
        # would read it as "no concern".
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

        if self.weight <= 0.0:
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

    Most producers know one availability for the whole axis ("this dataset has no
    intronic counts"), so requiring a full Series invites length mistakes. A per-cell
    Series is still accepted where availability genuinely varies, such as a mixture model
    that converged for some samples only.

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

    # Blank unusable severity rather than trusting the caller to pass NaN — a leftover
    # zero here would read as "no concern".
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


# ─── 3. All axes ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvidenceTable:
    """Every evidence axis for one dataset, rolled up to family level.

    Answers three questions and refuses others: how severe is each family, how much was
    measurable, and — given a bar the *caller* supplies — how many independent families
    corroborate a concern. There is deliberately no default bar; choosing one is
    calibration, and a default would become policy nobody remembers deciding.

    Args:
        axes: Evidence axes, all sharing one cell index.
        obs_names: The cell index every axis is aligned to.

    Raises:
        QCEvidenceError: If no axes are given, or an axis is misaligned.
    """

    axes: tuple[AxisEvidence, ...]
    obs_names: pd.Index

    #: Families whose severity may, in concert, justify quarantine. Multiplet and
    #: cell-calling are excluded: "this is two cells" and "this may be an empty droplet"
    #: are different claims from "this cell is dying".
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
        """Per-cell severity per family; NaN only where no axis in it was usable.

        Aggregation is the **maximum** across usable axes, not the mean. A mean lets a
        healthy correlated metric dilute a genuine signal — averaging normal MALAT1
        against a severe intronic anomaly reports "mild", when the honest reading is "one
        axis says this cell is damaged".
        """
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
        """Fraction of *present* families that were measurable, per cell.

        Measured against the families this table has, not all six, so a pipeline that
        never collects splice data does not report permanently degraded coverage.
        """
        usable = self.family_usable()
        if usable.shape[1] == 0:
            return pd.Series(0.0, index=self.obs_names, dtype=float)
        return usable.sum(axis=1).astype(float) / float(usable.shape[1])

    def concordant_family_count(self, *, min_severity: float) -> pd.Series:
        """Count families reaching ``min_severity``, per cell.

        Counts corroborating *mechanisms*, not correlated measurements of one mechanism.
        NaN never counts — an unmeasured family is neither concerning nor reassuring, and
        that asymmetry lives in :meth:`evidence_coverage`.

        Args:
            min_severity: Bar in ``[0, 1]``. Caller-supplied; there is no default.

        Raises:
            QCEvidenceError: If ``min_severity`` is outside ``[0, 1]``.
        """
        if not 0.0 <= min_severity <= 1.0:
            raise QCEvidenceError(f"min_severity must lie in [0, 1], got {min_severity}.")
        return (self.family_severity() >= min_severity).sum(axis=1).astype(int)

    def to_obs_frame(self) -> pd.DataFrame:
        """Flatten to ``adata.obs`` columns, prefixed ``qc_ev_``.

        Availability is written beside each severity because the pair is what carries
        meaning; the prefix keeps evidence visually distinct from verdicts.
        """
        columns: dict[str, pd.Series] = {}
        for axis in self.axes:
            columns[f"qc_ev_{axis.name}_severity"] = axis.severity
            columns[f"qc_ev_{axis.name}_availability"] = axis.availability.astype(str)
            # The raw measurement, where the producer had one. Written because a severity
            # cannot be checked against anything on its own: MALAT1 fraction and the
            # dissociation-stress score were computed, converted to severity, and discarded,
            # so neither the number nor the calibration figure the spec asks for existed.
            if axis.value is not None:
                columns[f"qc_ev_{axis.name}_value"] = axis.value
        for family, severity in self.family_severity().items():
            columns[f"qc_ev_family_{family}_severity"] = severity
        for family, usable in self.family_usable().items():
            columns[f"qc_ev_family_{family}_usable"] = usable
        columns["qc_evidence_coverage"] = self.evidence_coverage()
        return pd.DataFrame(columns, index=self.obs_names)


# ─── 4. Adjudication ────────────────────────────────────────────────────────────────

#: Families that may only ever *support* a damage case, never establish one alone. In
#: inflamed or lesional tissue, elevated mitochondrial fraction and FOS/JUN/HSP stress
#: programmes are genuinely biology.
SUPPORTING_FAMILIES: frozenset[EvidenceFamily] = frozenset({EvidenceFamily.METABOLIC_STRESS})


class QCStateInitial(StrEnum):
    """Provisional state assigned before any biological reference exists.

    Deliberately three values. ``rescued`` and ``unresolved_borderline`` are *post*
    -reference conclusions belonging to ``qc_finalization``; emitting them here would
    mean pretending to know which questionable cells are recoverable before there is
    anything to recover them against.
    """

    #: Nothing reaches the concern bar. May fit the biological reference.
    CORE = "core"

    #: Off, but not condemned. Retained and marked, excluded from reference fitting,
    #: eligible for rescue at ``qc_finalization``. Not a soft delete.
    BORDERLINE = "borderline"

    #: Uninformative, or severely damaged on concordant independent evidence.
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

    Every field is required. These are properties of an assay and a tissue, read off
    calibration figures; a default here would silently become the policy for every
    dataset that never looked.

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

        # Requiring one family is exactly the failure this design prevents: it would let a
        # single model's posterior condemn a cell.
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
) -> AdjudicationResult:
    """Assign a provisional QC state to every cell.

    There are exactly two routes to quarantine, and neither is "one axis was extreme":

    1. **Uninformative barcode** — capture is so poor there is nothing to adjudicate. The
       one single-family route, justified because the claim is "this barcode carries no
       usable information", not "this cell is damaged".
    2. **Concordant severe damage** — several *independent* damage families agree, at
       least one of which can establish damage on its own terms.

    Anything else abnormal becomes borderline. Low coverage makes the adjudicator more
    conservative, never less: condemning a cell on evidence we mostly could not collect
    is the worst available error.

    Args:
        evidence: Graded evidence for the dataset.
        policy: Calibrated bars.
    """
    cells = evidence.obs_names
    damage_severity = evidence.damage_family_severity()
    coverage = evidence.evidence_coverage()

    # NaN never counts either way; the asymmetry lives in coverage, which gates below.
    is_concerning = damage_severity >= policy.concern_severity
    is_severe = damage_severity >= policy.severe_severity
    n_concerning_families = is_concerning.sum(axis=1).astype(int)
    n_severe_families = is_severe.sum(axis=1).astype(int)

    # Families that can establish damage alone, so "severe stress + severe mito" cannot
    # reach quarantine by itself even though both are damage families.
    establishing_columns = [
        column
        for column in damage_severity.columns
        if EvidenceFamily(column) not in SUPPORTING_FAMILIES
    ]
    n_severe_establishing = is_severe[establishing_columns].sum(axis=1)

    # Route 1: no usable information in the barcode at all.
    capture_column = str(EvidenceFamily.CAPTURE_COMPLEXITY)
    is_uninformative = (
        (damage_severity[capture_column] >= policy.uninformative_capture_severity).fillna(False)
        if capture_column in damage_severity.columns
        else pd.Series(False, index=cells)
    )

    # Route 2: independent damage families agree, severely.
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

    # Reasons, most specific assigned last so it wins.
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

    # Multiplet, tracked entirely separately from damage.
    all_severity = evidence.family_severity()
    multiplet_column = str(EvidenceFamily.MULTIPLET)
    is_probable_multiplet = (
        (all_severity[multiplet_column] >= policy.multiplet_severity).fillna(False)
        if multiplet_column in all_severity.columns
        else pd.Series(False, index=cells)
    )
    # An otherwise unremarkable multiplet still deserves a reason, since "core" would
    # misdescribe it — but it is not a damage state.
    reason[is_probable_multiplet & (n_concerning_families == 0) & ~is_quarantined] = str(
        AdjudicationReason.PROBABLE_MULTIPLET
    )

    # Only ask idxmax about rows that have a concern. Calling it on all-NA rows is
    # deprecated in pandas and will raise, and "" is the honest answer there anyway.
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
    """Heuristic confidence in a cell's adjudication, in ``[0, 1]``.

    Two things make a call trustworthy: how much of the evidence space was measurable,
    and how far the deciding severity sits from the bar it did or did not cross. A cell
    judged on two of six families, a hair above the concern bar, is the least trustworthy
    call available and should sort to the top of a review queue.

    Explicitly a triage aid, **not** a probability — it is calibrated against nothing and
    must never be thresholded as though it were.
    """
    strongest_severity = damage_severity.max(axis=1, skipna=True)
    concern_bar = policy.concern_severity

    # Normalise by the wider side so both directions scale into [0, 1].
    bar_span = max(concern_bar, 1.0 - concern_bar) or 1.0
    distance_from_bar = ((strongest_severity - concern_bar).abs() / bar_span).fillna(0.0)

    return (coverage * distance_from_bar.clip(0.0, 1.0)).clip(0.0, 1.0)


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
]
