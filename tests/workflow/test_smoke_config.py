from __future__ import annotations

from pathlib import Path

import yaml

from cellquorum.config.loader import validate_config_dict

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_smoke_config_is_cpu_and_valid() -> None:
    doc = yaml.safe_load((REPO_ROOT / "docker" / "smoke" / "smoke.yaml").read_text())
    cfg = validate_config_dict(doc)
    assert cfg.compute.backend == "cpu"
