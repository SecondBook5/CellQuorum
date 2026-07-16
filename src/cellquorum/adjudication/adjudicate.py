"""Rule-based adjudication of CellQuorum cluster and state claims.

This module is intentionally conservative. It turns available evidence into an
auditable taxonomy class without pretending that a cluster label is a biological
truth. The same data structures can later back a richer evidence graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TaxonomyClass = Literal[
    "reproducible_state",
    "condition_restricted_state",
    "donor_restricted_population",
    "continuous_program",
    "ambiguous_population",
    "unsupported_split",
    "technical_population",
    "validated_identity",
    "rare_replicated_population",
]

EvidencePolarity = Literal["supports", "weakens", "vetoes", "context"]


@dataclass(frozen=True)
class EvidenceItem:
    """
    Store one auditable evidence statement used during adjudication.

    Args:
        name: Stable evidence identifier.
        value: JSON-friendly evidence value.
        polarity: How the evidence affects the claim.
        reason: Human-readable explanation.
    """

    # Store a stable evidence identifier.
    name: str

    # Store the JSON-friendly evidence value.
    value: object

    # Store how the evidence affects the claim.
    polarity: EvidencePolarity

    # Store the human-readable rationale.
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly evidence dictionary."""

        return {
            "name": self.name,
            "value": self.value,
            "polarity": self.polarity,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClusterEvidence:
    """
    Store evidence available for one cluster, split, or candidate state.

    Args:
        cluster_id: Stable cluster or candidate identifier.
        n_cells: Number of cells assigned to the candidate.
        donor_counts: Cell counts per donor.
        condition_counts: Cell counts per condition.
        marker_support: Optional marker-program support score in [0, 1].
        reproducibility_score: Optional cross-donor/resampling support score in [0, 1].
        technical_score: Optional technical-artifact score in [0, 1].
        continuity_score: Optional continuum/program score in [0, 1].
        split_support: Optional formal split/subcluster support score in [0, 1].
        notes: Optional freeform notes from upstream diagnostics.
    """

    # Store the candidate id.
    cluster_id: str

    # Store the candidate size.
    n_cells: int

    # Store cell counts per donor.
    donor_counts: dict[str, int] = field(default_factory=dict)

    # Store cell counts per condition.
    condition_counts: dict[str, int] = field(default_factory=dict)

    # Store marker support in [0, 1], when available.
    marker_support: float | None = None

    # Store reproducibility support in [0, 1], when available.
    reproducibility_score: float | None = None

    # Store technical artifact score in [0, 1], when available.
    technical_score: float | None = None

    # Store continuum/program score in [0, 1], when available.
    continuity_score: float | None = None

    # Store formal split support in [0, 1], when available.
    split_support: float | None = None

    # Store upstream notes.
    notes: list[str] = field(default_factory=list)

    @property
    def n_donors(self) -> int:
        """Return the number of donors represented by at least one cell."""

        return sum(1 for count in self.donor_counts.values() if count > 0)

    @property
    def n_conditions(self) -> int:
        """Return the number of represented conditions."""

        return sum(1 for count in self.condition_counts.values() if count > 0)

    @property
    def dominant_donor_fraction(self) -> float:
        """Return the largest donor's cell fraction."""

        if self.n_cells <= 0 or not self.donor_counts:
            return 0.0
        return max(self.donor_counts.values()) / self.n_cells

    @property
    def dominant_condition_fraction(self) -> float:
        """Return the largest condition's cell fraction."""

        if self.n_cells <= 0 or not self.condition_counts:
            return 0.0
        return max(self.condition_counts.values()) / self.n_cells

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly evidence summary."""

        return {
            "cluster_id": self.cluster_id,
            "n_cells": self.n_cells,
            "donor_counts": dict(self.donor_counts),
            "condition_counts": dict(self.condition_counts),
            "n_donors": self.n_donors,
            "n_conditions": self.n_conditions,
            "dominant_donor_fraction": self.dominant_donor_fraction,
            "dominant_condition_fraction": self.dominant_condition_fraction,
            "marker_support": self.marker_support,
            "reproducibility_score": self.reproducibility_score,
            "technical_score": self.technical_score,
            "continuity_score": self.continuity_score,
            "split_support": self.split_support,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AdjudicationRuleConfig:
    """
    Store conservative thresholds for rule-based adjudication.

    Args:
        min_cells: Minimum cells for a supported claim.
        min_replicated_donors: Minimum donors for replicated claims.
        donor_dominance_veto: Dominant-donor fraction that triggers donor restriction.
        condition_restriction_fraction: Dominant-condition fraction for condition restriction.
        technical_veto_score: Technical score that triggers a technical veto.
        continuity_score: Continuity score that labels a continuous program.
        split_support_min: Minimum formal split support for a reproducible state.
        marker_support_min: Minimum marker support for identity/state support.
        reproducibility_min: Minimum reproducibility score for reproducible states.
    """

    # Store minimum cells for claims.
    min_cells: int = 30

    # Store minimum donors for replicated claims.
    min_replicated_donors: int = 3

    # Store dominant-donor fraction that triggers a donor-restricted label.
    donor_dominance_veto: float = 0.8

    # Store dominant-condition fraction that triggers condition restriction.
    condition_restriction_fraction: float = 0.9

    # Store technical score that triggers a technical label.
    technical_veto_score: float = 0.75

    # Store continuity score that triggers a continuous-program label.
    continuity_score: float = 0.7

    # Store minimum formal split support.
    split_support_min: float = 0.6

    # Store minimum marker support.
    marker_support_min: float = 0.5

    # Store minimum reproducibility score.
    reproducibility_min: float = 0.6


@dataclass(frozen=True)
class AdjudicationResult:
    """
    Store the adjudicated taxonomy class and its evidence trail.

    Args:
        cluster_id: Stable cluster or candidate identifier.
        taxonomy_class: Assigned conservative taxonomy class.
        confidence: Rule-derived confidence in [0, 1].
        reasons: Human-readable reason list.
        vetoes: Veto evidence that prevented stronger claims.
        evidence: Full evidence trail considered by the rules.
    """

    # Store candidate id.
    cluster_id: str

    # Store assigned taxonomy class.
    taxonomy_class: TaxonomyClass

    # Store conservative confidence.
    confidence: float

    # Store concise human-readable reasons.
    reasons: list[str] = field(default_factory=list)

    # Store vetoing evidence.
    vetoes: list[EvidenceItem] = field(default_factory=list)

    # Store all evidence considered.
    evidence: list[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly adjudication dictionary."""

        return {
            "cluster_id": self.cluster_id,
            "taxonomy_class": self.taxonomy_class,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "vetoes": [item.to_dict() for item in self.vetoes],
            "evidence": [item.to_dict() for item in self.evidence],
        }


def adjudicate_cluster(
    evidence: ClusterEvidence,
    *,
    config: AdjudicationRuleConfig | None = None,
) -> AdjudicationResult:
    """
    Adjudicate one cluster or candidate state from currently available evidence.

    Args:
        evidence: Cluster-level evidence summary.
        config: Optional rule thresholds.

    Returns:
        Conservative adjudication result with reasons and vetoes.
    """

    rules = config or AdjudicationRuleConfig()
    items = _build_evidence_items(evidence, rules)
    vetoes = [item for item in items if item.polarity == "vetoes"]

    # Hard vetoes come first so stronger biological claims cannot leak through.
    if evidence.n_cells < rules.min_cells:
        return _result(
            evidence,
            taxonomy_class="unsupported_split",
            confidence=0.2,
            reasons=[f"Only {evidence.n_cells} cells; minimum is {rules.min_cells}."],
            evidence_items=items,
            vetoes=vetoes,
        )

    if _score_at_least(evidence.technical_score, rules.technical_veto_score):
        return _result(
            evidence,
            taxonomy_class="technical_population",
            confidence=_bounded(evidence.technical_score),
            reasons=["Technical-artifact evidence vetoed a biological claim."],
            evidence_items=items,
            vetoes=vetoes,
        )

    if evidence.dominant_donor_fraction >= rules.donor_dominance_veto:
        return _result(
            evidence,
            taxonomy_class="donor_restricted_population",
            confidence=evidence.dominant_donor_fraction,
            reasons=[
                "Candidate is dominated by one donor "
                f"({evidence.dominant_donor_fraction:.2f} of cells)."
            ],
            evidence_items=items,
            vetoes=vetoes,
        )

    # Then identify evidence-backed but still conservative biological classes.
    if (
        evidence.dominant_condition_fraction >= rules.condition_restriction_fraction
        and evidence.n_donors >= rules.min_replicated_donors
    ):
        return _result(
            evidence,
            taxonomy_class="condition_restricted_state",
            confidence=evidence.dominant_condition_fraction,
            reasons=[
                "Candidate is condition-restricted while represented across "
                f"{evidence.n_donors} donors."
            ],
            evidence_items=items,
            vetoes=vetoes,
        )

    if _score_at_least(evidence.continuity_score, rules.continuity_score):
        return _result(
            evidence,
            taxonomy_class="continuous_program",
            confidence=_bounded(evidence.continuity_score),
            reasons=[
                "Continuity evidence supports a program or gradient rather than " "a discrete type."
            ],
            evidence_items=items,
            vetoes=vetoes,
        )

    if (
        evidence.n_donors >= rules.min_replicated_donors
        and _score_at_least(evidence.reproducibility_score, rules.reproducibility_min)
        and _score_at_least(evidence.split_support, rules.split_support_min)
    ):
        marker_bonus = _bounded(evidence.marker_support) * 0.1
        confidence = min(
            1.0,
            max(
                evidence.reproducibility_score or 0.0,
                evidence.split_support or 0.0,
            )
            + marker_bonus,
        )
        return _result(
            evidence,
            taxonomy_class="reproducible_state",
            confidence=confidence,
            reasons=[
                "Candidate has donor replication, reproducibility support, and "
                "formal split support."
            ],
            evidence_items=items,
            vetoes=vetoes,
        )

    if evidence.n_donors >= rules.min_replicated_donors and _score_at_least(
        evidence.marker_support,
        rules.marker_support_min,
    ):
        return _result(
            evidence,
            taxonomy_class="rare_replicated_population",
            confidence=max(0.4, _bounded(evidence.marker_support)),
            reasons=[
                "Candidate has marker support and donor replication but lacks "
                "stronger split evidence."
            ],
            evidence_items=items,
            vetoes=vetoes,
        )

    return _result(
        evidence,
        taxonomy_class="ambiguous_population",
        confidence=0.3,
        reasons=["Available evidence is insufficient for a stronger taxonomy claim."],
        evidence_items=items,
        vetoes=vetoes,
    )


def _build_evidence_items(
    evidence: ClusterEvidence,
    rules: AdjudicationRuleConfig,
) -> list[EvidenceItem]:
    """Build the evidence trail used by the adjudication rules."""

    items = [
        EvidenceItem(
            name="n_cells",
            value=evidence.n_cells,
            polarity="supports" if evidence.n_cells >= rules.min_cells else "vetoes",
            reason=f"Candidate has {evidence.n_cells} cells.",
        ),
        EvidenceItem(
            name="n_donors",
            value=evidence.n_donors,
            polarity="supports" if evidence.n_donors >= rules.min_replicated_donors else "weakens",
            reason=f"Candidate is represented in {evidence.n_donors} donor(s).",
        ),
        EvidenceItem(
            name="dominant_donor_fraction",
            value=evidence.dominant_donor_fraction,
            polarity="vetoes"
            if evidence.dominant_donor_fraction >= rules.donor_dominance_veto
            else "context",
            reason="Dominant donor fraction measures donor restriction.",
        ),
        EvidenceItem(
            name="dominant_condition_fraction",
            value=evidence.dominant_condition_fraction,
            polarity="supports"
            if evidence.dominant_condition_fraction >= rules.condition_restriction_fraction
            else "context",
            reason="Dominant condition fraction measures condition restriction.",
        ),
    ]

    _append_score(items, "marker_support", evidence.marker_support, rules.marker_support_min)
    _append_score(
        items,
        "reproducibility_score",
        evidence.reproducibility_score,
        rules.reproducibility_min,
    )
    _append_score(
        items,
        "technical_score",
        evidence.technical_score,
        rules.technical_veto_score,
        veto=True,
    )
    _append_score(items, "continuity_score", evidence.continuity_score, rules.continuity_score)
    _append_score(items, "split_support", evidence.split_support, rules.split_support_min)

    for note in evidence.notes:
        items.append(
            EvidenceItem(
                name="note",
                value=note,
                polarity="context",
                reason="Upstream diagnostic note.",
            )
        )

    return items


def _append_score(
    items: list[EvidenceItem],
    name: str,
    value: float | None,
    threshold: float,
    *,
    veto: bool = False,
) -> None:
    """Append an optional score evidence item."""

    if value is None:
        return
    polarity: EvidencePolarity = "supports" if value >= threshold else "weakens"
    if veto and value >= threshold:
        polarity = "vetoes"
    items.append(
        EvidenceItem(
            name=name,
            value=value,
            polarity=polarity,
            reason=f"{name}={value:.3g}; threshold={threshold:.3g}.",
        )
    )


def _result(
    evidence: ClusterEvidence,
    *,
    taxonomy_class: TaxonomyClass,
    confidence: float,
    reasons: list[str],
    evidence_items: list[EvidenceItem],
    vetoes: list[EvidenceItem],
) -> AdjudicationResult:
    """Build a bounded adjudication result."""

    return AdjudicationResult(
        cluster_id=evidence.cluster_id,
        taxonomy_class=taxonomy_class,
        confidence=_bounded(confidence),
        reasons=reasons,
        vetoes=vetoes,
        evidence=evidence_items,
    )


def _score_at_least(value: float | None, threshold: float) -> bool:
    """Return whether an optional score passes a threshold."""

    return value is not None and value >= threshold


def _bounded(value: float | None) -> float:
    """Bound an optional numeric score to [0, 1]."""

    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))
