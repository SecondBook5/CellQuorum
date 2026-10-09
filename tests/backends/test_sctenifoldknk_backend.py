"""Isolated-runtime dispatch and planner-visible backend settings."""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest

from cellquorum.backends.registry import build_default_backend_registry
from cellquorum.backends.sctenifoldknk_backend import ScTenifoldKnkBackend
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.planner import PipelinePlanner


def test_status_probes_selected_python(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    backend = ScTenifoldKnkBackend(python_executable="/isolated/python")
    assert backend.status().available
    assert calls[0][0] == "/isolated/python"
    assert "from scTenifold import virtual_knockout" in calls[0]


def test_missing_python_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("missing interpreter")

    monkeypatch.setattr(subprocess, "run", run)
    status = ScTenifoldKnkBackend().status()
    assert not status.available and "missing interpreter" in status.details["error"]


def test_runner_preserves_nonzero_exit_and_caps_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=9, stdout="", stderr="fit failed")

    monkeypatch.setattr(subprocess, "run", run)
    result = ScTenifoldKnkBackend(python_executable="/isolated/python").run_script(
        ["--input", "input.csv"], timeout=12
    )
    assert result.returncode == 9
    assert calls[0][1]["timeout"] == 12
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "1"
    assert calls[0][1]["env"]["PATH"] == os.environ["PATH"]


def test_registry_threads_isolated_python() -> None:
    backend = build_default_backend_registry(sctenifoldknk_python="/isolated/python").get(
        "sctenifoldknk"
    )
    assert backend.python_executable == "/isolated/python"


def test_planner_uses_configured_python() -> None:
    config = CellQuorumConfig()
    config.perturbation.sctenifoldknk.python_executable = "/isolated/python"
    planner = PipelinePlanner(config)
    assert planner.backend_registry.get("sctenifoldknk").python_executable == "/isolated/python"
