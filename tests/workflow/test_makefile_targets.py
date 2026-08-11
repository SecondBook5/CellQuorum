from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_makefile_declares_required_targets() -> None:
    text = (REPO_ROOT / "Makefile").read_text()
    targets = set(re.findall(r"^([a-zA-Z0-9_-]+):", text, re.MULTILINE))
    assert {"image", "image-gpu", "lock", "smoke", "matrix"} <= targets


def test_smoke_target_runs_three_checks() -> None:
    text = (REPO_ROOT / "Makefile").read_text()
    # version, plan, and env-list assertions must all appear in the smoke recipe.
    assert "--version" in text
    assert "plan --config docker/smoke/smoke.yaml" in text
    for env in ("celloracle_env", "pyscenic_env", "hdwgcna_env", "scclr", "sccoda_env"):
        assert env in text
