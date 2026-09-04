"""The shipped mast config must load and plan the full backbone."""

from __future__ import annotations

from pathlib import Path

import pytest
from _external_data import stub_config_env

from cellquorum.config.loader import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mast_cell.yaml"


def test_mast_config_loads_and_enables_backbone(tmp_path, monkeypatch: pytest.MonkeyPatch):
    # The config names its Cell Ranger root via ${oc.env:...} rather than hardcoding one
    # machine's external drive, so the interpolation needs *a* value to resolve. This
    # test only checks that the config validates and enables the right stages — nothing
    # is read from disk — so a throwaway path is exactly right, and it keeps the test
    # passing identically in CI and on any contributor's machine.
    stub_config_env(monkeypatch, tmp_path)

    config = load_config(CONFIG)
    assert config.integration.batch_key == "batch"
    assert config.ambient_correction.enabled is True
    assert "Mast cells" in config.annotation.marker_panels
