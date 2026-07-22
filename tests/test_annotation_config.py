"""Tests for AnnotationConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.annotation.config import AnnotationConfig
from cellquorum.config.models import CellQuorumConfig


def test_annotation_defaults():
    c = AnnotationConfig()
    assert c.method == "marker_vote"
    assert c.cluster_key == "leiden"
    assert c.key_added == "cell_type"


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
