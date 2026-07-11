"""The shipped mast config must load and plan the full backbone."""

from __future__ import annotations

from pathlib import Path

from cellquorum.config.loader import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mast_cell.yaml"


def test_mast_config_loads_and_enables_backbone(tmp_path):
    config = load_config(CONFIG)
    assert config.integration.batch_key == "batch"
    assert config.ambient_correction.enabled is True
    assert "Mast cells" in config.annotation.marker_panels
