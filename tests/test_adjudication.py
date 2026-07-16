"""Tests for cluster/state adjudication rules."""

from __future__ import annotations

from cellquorum.adjudication import ClusterEvidence, adjudicate_cluster


def test_adjudicate_cluster_labels_reproducible_state():
    evidence = ClusterEvidence(
        cluster_id="c1",
        n_cells=250,
        donor_counts={"d1": 50, "d2": 60, "d3": 70, "d4": 70},
        condition_counts={"case": 130, "control": 120},
        marker_support=0.8,
        reproducibility_score=0.75,
        split_support=0.7,
    )

    result = adjudicate_cluster(evidence)

    assert result.taxonomy_class == "reproducible_state"
    assert result.confidence > 0.75
    assert not result.vetoes


def test_adjudicate_cluster_vetoes_technical_population():
    evidence = ClusterEvidence(
        cluster_id="c2",
        n_cells=200,
        donor_counts={"d1": 50, "d2": 50, "d3": 50, "d4": 50},
        condition_counts={"case": 100, "control": 100},
        marker_support=0.9,
        reproducibility_score=0.8,
        split_support=0.8,
        technical_score=0.9,
    )

    result = adjudicate_cluster(evidence)

    assert result.taxonomy_class == "technical_population"
    assert any(item.name == "technical_score" for item in result.vetoes)


def test_adjudicate_cluster_labels_donor_restricted_population():
    evidence = ClusterEvidence(
        cluster_id="c3",
        n_cells=100,
        donor_counts={"d1": 85, "d2": 10, "d3": 5},
        condition_counts={"case": 50, "control": 50},
        marker_support=0.8,
        reproducibility_score=0.8,
        split_support=0.8,
    )

    result = adjudicate_cluster(evidence)

    assert result.taxonomy_class == "donor_restricted_population"
    assert any(item.name == "dominant_donor_fraction" for item in result.vetoes)


def test_adjudicate_cluster_labels_condition_restricted_state():
    evidence = ClusterEvidence(
        cluster_id="c4",
        n_cells=120,
        donor_counts={"d1": 30, "d2": 30, "d3": 30, "d4": 30},
        condition_counts={"case": 115, "control": 5},
    )

    result = adjudicate_cluster(evidence)

    assert result.taxonomy_class == "condition_restricted_state"


def test_adjudicate_cluster_labels_unsupported_split_for_small_candidate():
    evidence = ClusterEvidence(
        cluster_id="c5",
        n_cells=10,
        donor_counts={"d1": 5, "d2": 5},
        condition_counts={"case": 5, "control": 5},
    )

    result = adjudicate_cluster(evidence)

    assert result.taxonomy_class == "unsupported_split"
