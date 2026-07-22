"""Tests for AnnotationConsensusConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellquorum.annotation_consensus.config import AnnotationConsensusConfig


def test_defaults():
    c = AnnotationConsensusConfig()
    assert c.enabled is True
    assert c.method_label_keys == []
    assert c.key_added == "cell_type"
    assert c.confidence_key == "annotation_confidence"
    assert c.needs_review_key == "needs_review"
    assert c.min_agree_fraction == 0.5


def test_accepts_fields():
    c = AnnotationConsensusConfig(
        method_label_keys=["cell_type_markers", "cell_type_celltypist", "ref_state"],
        backbone_aliases={"T cell": "T/NK", "keratinocyte": "Keratinocytes"},
        granular_source_key="ref_state",
    )
    assert len(c.method_label_keys) == 3
    assert c.backbone_aliases["T cell"] == "T/NK"
    assert c.granular_source_key == "ref_state"


def test_rejects_unknown_key():
    with pytest.raises(ValidationError):
        AnnotationConsensusConfig(nonsense=True)
