"""Tests for AnnotationConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.config.models import CellQuorumConfig
from cellquorum.stages.annotation.config import AnnotationConfig


def test_annotation_defaults():
    c = AnnotationConfig()
    assert c.method == "marker_vote"
    assert c.cluster_key == "leiden"
    assert c.key_added == "cell_type"
    assert c.n_permutations == 100
    assert c.significance_alpha == 0.05


def test_annotation_rejects_non_positive_n_permutations():
    with pytest.raises(ValidationError):
        AnnotationConfig(n_permutations=0)


def test_annotation_rejects_significance_alpha_outside_open_unit_interval():
    with pytest.raises(ValidationError):
        AnnotationConfig(significance_alpha=0.0)
    with pytest.raises(ValidationError):
        AnnotationConfig(significance_alpha=1.0)


def test_annotation_strict():
    with pytest.raises(ValidationError):
        AnnotationConfig(bogus=1)


def test_annotation_accepts_methods_list():
    """Test that AnnotationConfig accepts a methods list via pydantic validation."""
    config = CellQuorumConfig.model_validate(
        {
            "annotation": {
                "methods": [
                    {"method": "marker_vote", "key_added": "cell_type_mv"},
                    {
                        "method": "celltypist",
                        "key_added": "cell_type_ct",
                        "model": "Immune_All_Low.pkl",
                    },
                ]
            }
        }
    )
    assert config.annotation.methods == [
        {"method": "marker_vote", "key_added": "cell_type_mv"},
        {"method": "celltypist", "key_added": "cell_type_ct", "model": "Immune_All_Low.pkl"},
    ]


def test_annotation_scalar_method_has_empty_methods_list():
    """Test that scalar method configs have an empty methods list by default."""
    config = CellQuorumConfig.model_validate({"annotation": {"method": "marker_vote"}})
    assert config.annotation.method == "marker_vote"
    assert config.annotation.methods == []
