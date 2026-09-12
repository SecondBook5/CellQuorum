from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_makefile_declares_required_targets() -> None:
    text = (REPO_ROOT / "Makefile").read_text()
    targets = set(re.findall(r"^([a-zA-Z0-9_-]+):", text, re.MULTILINE))
    assert {"image", "image-gpu", "lock", "smoke", "smoke-gpu", "matrix"} <= targets


def test_smoke_target_runs_three_checks() -> None:
    """The CPU smoke target checks version, plan, and the CPU target's own two
    required environments (cellquorum-core, cellquorum-r) — the optional isolated
    backend environments (celloracle_env, pyscenic_env, ...) moved to the separate
    `backends` Docker target and are not part of the default image's contract."""
    text = (REPO_ROOT / "Makefile").read_text()
    assert "--version" in text
    assert "plan --config docker/smoke/smoke.yaml" in text
    assert "REQUIRED_ENVS ?= cellquorum-core cellquorum-r" in text


def test_smoke_gpu_target_checks_the_gpu_environment_and_device_probes() -> None:
    """smoke-gpu must add cellquorum-gpu to the required-env check and run the
    device-verification script, not just re-run the CPU smoke checks."""
    text = (REPO_ROOT / "Makefile").read_text()
    assert "smoke-gpu:" in text
    assert "cellquorum-gpu" in text
    assert "scripts/verify_gpu.py" in text
