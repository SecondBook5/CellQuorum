"""The shared external-environment backend: launcher lookup, probe and script runner."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import pytest

from cellquorum.backends.external_env import ExternalEnvBackend


@dataclass
class _ToyBackend(ExternalEnvBackend):
    name: str = "toy"
    env_name: str = "toy_env"
    module_name = "toytool"


def test_missing_launcher_is_reported_without_probing(monkeypatch):
    monkeypatch.setattr("cellquorum.backends.external_env.resolve_launcher", lambda _n: None)
    status = _ToyBackend().status()
    assert not status.available
    assert status.missing == ["micromamba"]


def test_missing_module_is_reported(monkeypatch):
    monkeypatch.setattr("cellquorum.backends.external_env.resolve_launcher", lambda n: f"/x/{n}")
    monkeypatch.setattr(
        "cellquorum.backends.external_env.env_python_module_available", lambda *a: False
    )
    status = _ToyBackend().status()
    assert not status.available
    assert status.missing == ["toytool"]


def test_available_when_launcher_and_module_present(monkeypatch):
    monkeypatch.setattr("cellquorum.backends.external_env.resolve_launcher", lambda n: f"/x/{n}")
    monkeypatch.setattr(
        "cellquorum.backends.external_env.env_python_module_available", lambda *a: True
    )
    assert _ToyBackend().status().available


def test_run_script_builds_the_environment_command(monkeypatch, tmp_path):
    script = tmp_path / "helper.py"
    script.write_text("print(1)\n")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("cellquorum.backends.external_env.resolve_launcher", lambda n: f"/x/{n}")
    monkeypatch.setattr("cellquorum.backends.external_env.subprocess.run", fake_run)
    _ToyBackend(script_timeout_seconds=7).run_script(script, ["a", "b"])
    assert seen["cmd"] == ["/x/micromamba", "run", "-n", "toy_env", "python", str(script), "a", "b"]
    assert seen["kwargs"]["timeout"] == 7
    assert seen["kwargs"]["check"] is False
    assert seen["kwargs"]["capture_output"] is True


def test_run_script_timeout_override(monkeypatch, tmp_path):
    script = tmp_path / "helper.py"
    script.write_text("")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("cellquorum.backends.external_env.resolve_launcher", lambda n: f"/x/{n}")
    monkeypatch.setattr("cellquorum.backends.external_env.subprocess.run", fake_run)
    _ToyBackend().run_script(script, timeout=3)
    assert seen["timeout"] == 3


def test_run_script_requires_script_and_launcher(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="toy helper script not found"):
        _ToyBackend().run_script(tmp_path / "absent.py")
    script = tmp_path / "helper.py"
    script.write_text("")
    monkeypatch.setattr("cellquorum.backends.external_env.resolve_launcher", lambda _n: None)
    with pytest.raises(FileNotFoundError, match="was not found on PATH"):
        _ToyBackend().run_script(script)


@pytest.mark.parametrize("bad", ["", "1abc", "a b", "a;rm", "a-b"])
def test_module_name_is_validated_before_anything_runs(bad):
    with pytest.raises(ValueError, match="Invalid Python module name"):
        _ToyBackend()._py_module_available(bad)
