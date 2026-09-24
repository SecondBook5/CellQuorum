"""Tests for the PerturbationConfig model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.stages.gene_regulation.perturbation.config import PerturbationConfig


def test_defaults() -> None:
    c = PerturbationConfig()
    assert c.enabled is True
    assert c.method == "celloracle"
    assert c.layer == "counts"
    assert c.organism == "human"
    assert c.cluster_key is None
    assert c.cluster_labels is None
    assert c.embedding_key is None
    assert c.rep_key is None
    assert c.condition_key is None
    assert c.healthy_label is None
    assert c.tf_list is None
    assert c.n_top_targets == 20
    assert c.knn_n_neighbors == 200
    assert c.max_cells == 0
    assert c.sampled_fraction == pytest.approx(1.0)
    assert c.filter_p == pytest.approx(0.001)
    assert c.n_propagation == 3
    assert c.min_cells_total == 200
    assert c.seed == 0
    assert c.env_name == "celloracle_env"
    assert c.launcher == "micromamba"
    assert c.timeout_seconds == 10800


def test_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        PerturbationConfig(bogus=1)


def test_tf_list_accepts_list() -> None:
    c = PerturbationConfig(tf_list=["PROX1", "PIEZO1"])
    assert c.tf_list == ["PROX1", "PIEZO1"]


def test_cluster_labels_accepts_dict() -> None:
    c = PerturbationConfig(cluster_labels={"0": "Perivascular", "1": "Mechanosensory"})
    assert c.cluster_labels == {"0": "Perivascular", "1": "Mechanosensory"}
