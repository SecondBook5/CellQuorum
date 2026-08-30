"""Tests for top-level CellQuorum QC configuration integration."""

from __future__ import annotations

# Import Path for YAML config fixture paths.
from pathlib import Path

# Import SimpleNamespace for lightweight stage-resolution context tests.
from types import SimpleNamespace

# Import pytest for exception assertions.
import pytest

# Import Pydantic validation error for strict base model tests.
from pydantic import ValidationError

# Import the shared strict base model.
from cellquorum.config.base import StrictBaseModel

# Import configuration loading utilities.
from cellquorum.config.loader import ConfigLoadError, load_config, validate_config_dict

# Import the top-level CellQuorum configuration model.
from cellquorum.config.models import CellQuorumConfig

# Import the QC configuration model.
from cellquorum.stages.qc.config import QCConfig

# Import QC stage config-resolution helpers.
from cellquorum.stages.qc.stage import is_qc_stage_enabled, resolve_qc_config


class TinyStrictConfig(StrictBaseModel):
    """
    Tiny strict config model for testing StrictBaseModel behavior.

    Args:
        value: Example integer value.
    """

    # Store an example value.
    value: int = 1


def test_strict_base_model_forbids_unknown_fields() -> None:
    """
    Verify the moved StrictBaseModel still forbids unknown fields.

    This protects the central reason for the shared base model refactor.
    """

    # Confirm supported fields validate normally.
    config = TinyStrictConfig(value=2)

    # Confirm the supported field was retained.
    assert config.value == 2

    # Confirm unknown fields fail validation.
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TinyStrictConfig(value=2, unknown=True)  # type: ignore[call-arg]


def test_cellquorum_config_default_includes_qc_config() -> None:
    """
    Verify the default top-level config includes QC settings.

    This is the bridge from standalone QCStage to config-driven CellQuorum runs.
    """

    # Build a default top-level config.
    config = CellQuorumConfig()

    # Confirm QC config is present.
    assert isinstance(config.qc, QCConfig)

    # Confirm key QC defaults are available from the top-level config.
    assert config.qc.enabled is True
    assert config.qc.mode == "flag_no_drop"
    assert config.qc.threshold_strategy == "fixed_and_mad"
    assert config.qc.basic.min_genes_per_cell == 200
    assert config.qc.basic.min_cells_per_gene == 3


def test_cellquorum_config_accepts_qc_mapping_directly() -> None:
    """
    Verify CellQuorumConfig can coerce a QC mapping into QCConfig.

    Programmatic callers should be able to pass nested dictionaries directly.
    """

    # Build a top-level config with a QC mapping.
    config = CellQuorumConfig(
        qc={
            "enabled": True,
            "mode": "filter",
            "threshold_strategy": "fixed",
            "mad": {
                "enabled": False,
            },
            "basic": {
                "min_genes_per_cell": 123,
                "min_cells_per_gene": 4,
                "max_mito_percent": 9.5,
            },
        }
    )

    # Confirm the QC mapping was converted into QCConfig.
    assert isinstance(config.qc, QCConfig)

    # Confirm nested QC values were retained.
    assert config.qc.mode == "filter"
    assert config.qc.threshold_strategy == "fixed"
    assert config.qc.mad.enabled is False
    assert config.qc.basic.min_genes_per_cell == 123
    assert config.qc.basic.min_cells_per_gene == 4
    assert config.qc.basic.max_mito_percent == 9.5


def test_validate_config_dict_accepts_qc_block() -> None:
    """
    Verify validate_config_dict accepts a top-level qc block.

    This is the main dictionary-to-runtime-config path used by notebooks and API
    calls.
    """

    # Validate a dictionary with a QC block.
    config = validate_config_dict(
        {
            "project": {
                "name": "qc_dict_project",
            },
            "qc": {
                "enabled": True,
                "mode": "both",
                "threshold_strategy": "fixed",
                "mad": {
                    "enabled": False,
                },
                "basic": {
                    "min_genes_per_cell": 150,
                    "min_cells_per_gene": 5,
                    "max_mito_percent": 12.0,
                },
            },
        }
    )

    # Confirm top-level config was returned.
    assert isinstance(config, CellQuorumConfig)

    # Confirm QC config was parsed.
    assert isinstance(config.qc, QCConfig)
    assert config.qc.mode == "both"
    assert config.qc.threshold_strategy == "fixed"
    assert config.qc.basic.min_genes_per_cell == 150
    assert config.qc.basic.min_cells_per_gene == 5
    assert config.qc.basic.max_mito_percent == 12.0


def test_validate_config_dict_rejects_unknown_qc_keys() -> None:
    """
    Verify unknown nested QC keys still fail strict validation.

    Adding qc to the top-level config must not weaken strict config validation.
    """

    # Confirm unknown keys inside qc fail validation.
    with pytest.raises(ConfigLoadError, match="Invalid CellQuorum configuration"):
        validate_config_dict(
            {
                "project": {
                    "name": "bad_qc_project",
                },
                "qc": {
                    "not_a_real_qc_key": True,
                },
            }
        )


def test_load_config_accepts_yaml_qc_block(tmp_path: Path) -> None:
    """
    Verify YAML config files can include a top-level qc block.

    This is the future CLI path for configuring QC.
    """

    # Build a config file path.
    config_path = tmp_path / "config.yaml"

    # Write a minimal YAML config with QC settings.
    config_path.write_text(
        """
project:
  name: yaml_qc_project
compute:
  backend: cpu
  prefer_gpu: false
qc:
  enabled: true
  mode: filter
  threshold_strategy: fixed
  mad:
    enabled: false
  basic:
    min_genes_per_cell: 175
    min_cells_per_gene: 6
    max_mito_percent: 11.0
""",
        encoding="utf-8",
    )

    # Load the config file.
    config = load_config(config_path)

    # Confirm the top-level config loaded.
    assert isinstance(config, CellQuorumConfig)

    # Confirm QC settings loaded.
    assert isinstance(config.qc, QCConfig)
    assert config.project.name == "yaml_qc_project"
    assert config.qc.mode == "filter"
    assert config.qc.threshold_strategy == "fixed"
    assert config.qc.mad.enabled is False
    assert config.qc.basic.min_genes_per_cell == 175
    assert config.qc.basic.min_cells_per_gene == 6
    assert config.qc.basic.max_mito_percent == 11.0


def test_qc_stage_resolves_top_level_config_qc_block() -> None:
    """
    Verify QCStage config resolution uses CellQuorumConfig.qc automatically.

    This proves the stage no longer requires a manual QCStage(config=...) override
    when running from the top-level CellQuorum config.
    """

    # Build a top-level config with custom QC settings.
    config = CellQuorumConfig(
        qc={
            "mode": "both",
            "threshold_strategy": "fixed",
            "mad": {
                "enabled": False,
            },
        }
    )

    # Build a lightweight context carrying the top-level config.
    context = SimpleNamespace(config=config)

    # Resolve QC config through the stage helper.
    resolved_qc_config = resolve_qc_config(context)

    # Confirm the top-level QC config was used directly.
    assert resolved_qc_config is config.qc
    assert resolved_qc_config.mode == "both"
    assert resolved_qc_config.threshold_strategy == "fixed"


def test_qc_stage_enablement_uses_stage_flag_and_qc_flag() -> None:
    """
    Verify QC stage enablement respects both top-level and QC-level flags.

    The top-level stages.qc flag controls whether QC is allowed to run, while
    qc.enabled controls the QC module itself.
    """

    # Build config with top-level QC stage disabled.
    top_level_disabled = CellQuorumConfig(
        stages={
            "qc": False,
        },
        qc={
            "enabled": True,
        },
    )

    # Confirm top-level stage selection disables QC.
    assert (
        is_qc_stage_enabled(
            SimpleNamespace(config=top_level_disabled),
            top_level_disabled.qc,
        )
        is False
    )

    # Build config with QC module disabled.
    qc_disabled = CellQuorumConfig(
        stages={
            "qc": True,
        },
        qc={
            "enabled": False,
        },
    )

    # Confirm module-level QC flag disables QC.
    assert (
        is_qc_stage_enabled(
            SimpleNamespace(config=qc_disabled),
            qc_disabled.qc,
        )
        is False
    )

    # Build config with both flags enabled.
    qc_enabled = CellQuorumConfig(
        stages={
            "qc": True,
        },
        qc={
            "enabled": True,
        },
    )

    # Confirm QC is enabled when both flags allow it.
    assert (
        is_qc_stage_enabled(
            SimpleNamespace(config=qc_enabled),
            qc_enabled.qc,
        )
        is True
    )
