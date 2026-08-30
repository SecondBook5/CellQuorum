"""Tests for subclustering configuration."""

from __future__ import annotations

import pytest

from cellquorum.stages.clustering.subclustering.config import SubclusteringConfig


def test_subclustering_config_defaults() -> None:
    """Verify SubclusteringConfig has correct generic defaults."""
    # Build default config.
    config = SubclusteringConfig()

    # Verify enabled flag defaults to False.
    assert config.enabled is False

    # Verify focus defaults are generic (not KC-specific).
    assert config.focus.label_key == "cell_type"
    assert config.focus.labels == []

    # Verify group_filter defaults are None (no default group).
    assert config.group_filter.group_key is None
    assert config.group_filter.min_cells is None

    # Verify counts_layer default.
    assert config.counts_layer == "counts"

    # Verify reembed sub-block exists (placeholder for Tasks 2-3).
    assert hasattr(config, "reembed")
    assert config.reembed.representations == ["batch_aware"]

    # Verify partition sub-block exists.
    assert hasattr(config, "partition")
    assert config.partition.method == "choir"
    assert config.partition.seeds == [0]

    # Verify formal_test sub-block exists.
    assert hasattr(config, "formal_test")
    assert config.formal_test.method == "scshc"
    assert config.formal_test.alpha == 0.05

    # Verify donor_gate sub-block exists.
    assert hasattr(config, "donor_gate")
    assert config.donor_gate.group_key is None
    assert config.donor_gate.min_groups == 3

    # Verify diagnostics sub-block exists.
    assert hasattr(config, "diagnostics")
    assert config.diagnostics.clustree is True
    assert config.diagnostics.stability_curve is True

    # Verify action defaults to flag (not drop).
    assert config.action == "flag"

    # Verify key_added default.
    assert config.key_added == "subcluster"


def test_subclustering_config_lung_focus_validates() -> None:
    """Verify a LUNG-focus config validates (proves no KC hardcoding)."""
    # Build a lung-lineage config (not KC).
    config = SubclusteringConfig(
        enabled=True,
        focus={
            "label_key": "cell_type",
            "labels": ["AT2"],
        },
        group_filter={
            "group_key": "sample",
            "min_cells": 50,
        },
    )

    # Verify focus is lung (not KC).
    assert config.focus.label_key == "cell_type"
    assert config.focus.labels == ["AT2"]

    # Verify group_filter uses sample (not patient_id).
    assert config.group_filter.group_key == "sample"
    assert config.group_filter.min_cells == 50

    # Verify enabled flag is respected.
    assert config.enabled is True


def test_subclustering_config_rejects_typos() -> None:
    """Verify StrictBaseModel rejects typo fields."""
    from pydantic import ValidationError

    # Attempt to build a config with a typo field.
    with pytest.raises(ValidationError):
        SubclusteringConfig(enabeld=True)  # typo: enabeld → enabled


def test_subclustering_config_multi_label_focus() -> None:
    """Verify multi-label focus (e.g., multiple KC states) validates."""
    # Build a multi-label focus config.
    config = SubclusteringConfig(
        focus={
            "label_key": "cell_type",
            "labels": ["KC1", "KC2", "KC3"],
        }
    )

    # Verify multi-label focus.
    assert config.focus.labels == ["KC1", "KC2", "KC3"]


def test_subclustering_config_empty_labels_no_op() -> None:
    """Verify empty labels list is valid (no-op focus extraction)."""
    # Build a config with empty labels (keep all cells).
    config = SubclusteringConfig(
        focus={
            "label_key": "cell_type",
            "labels": [],
        }
    )

    # Verify empty labels is valid.
    assert config.focus.labels == []
