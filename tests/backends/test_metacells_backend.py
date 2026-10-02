"""Tests for the MetaCell2 isolated-env subprocess backend."""

from __future__ import annotations

import subprocess

import pytest

from cellquorum.backends.metacells_backend import (
    RUN_METACELL2_PY,
    build_metacells_backend,
)


def test_backend_identity_defaults() -> None:
    b = build_metacells_backend()
    assert b.name == "metacells"
    assert b.kind == "external"
    assert b.env_name == "metacells_env"
    assert b.launcher == "micromamba"


def test_run_script_builds_micromamba_python_argv(monkeypatch, tmp_path) -> None:
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n")
    b = build_metacells_backend()
    monkeypatch.setattr(
        "cellquorum.backends.metacells_backend.resolve_launcher",
        lambda _name: "/usr/bin/micromamba",
    )

    captured = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    b.run_script(script, ["--flag", "value"], timeout=123)

    assert captured["cmd"] == [
        "/usr/bin/micromamba",
        "run",
        "-n",
        "metacells_env",
        "python",
        str(script),
        "--flag",
        "value",
    ]
    assert captured["kwargs"]["check"] is False
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["timeout"] == 123


def test_run_script_missing_launcher_raises(monkeypatch, tmp_path) -> None:
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n")
    b = build_metacells_backend()
    monkeypatch.setattr(b, "_launcher_available", lambda: False)
    with pytest.raises(FileNotFoundError):
        b.run_script(script, [])


def test_run_script_missing_script_raises(tmp_path) -> None:
    b = build_metacells_backend()
    with pytest.raises(FileNotFoundError):
        b.run_script(tmp_path / "does_not_exist.py", [])


def test_status_unavailable_when_launcher_missing(monkeypatch) -> None:
    b = build_metacells_backend()
    monkeypatch.setattr(b, "_launcher_available", lambda: False)
    st = b.status()
    assert st.available is False
    assert "micromamba" in st.missing


def test_bundled_script_path_points_into_metacells_scripts() -> None:
    assert RUN_METACELL2_PY.name == "run_metacell2.py"
    assert RUN_METACELL2_PY.parent.name == "metacells_scripts"
