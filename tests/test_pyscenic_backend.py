"""Tests for the pySCENIC isolated-env subprocess backend."""

from __future__ import annotations

import subprocess

import pytest

from cellquorum.backends.pyscenic_backend import (
    PYSCENIC_AUCELL_PY,
    PYSCENIC_GRN_PY,
    build_pyscenic_backend,
)


def test_backend_identity_defaults() -> None:
    b = build_pyscenic_backend()
    assert b.name == "pyscenic"
    assert b.kind == "external"
    assert b.env_name == "pyscenic_env"
    assert b.launcher == "micromamba"


def test_run_script_builds_micromamba_python_argv(monkeypatch, tmp_path) -> None:
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n")
    b = build_pyscenic_backend()
    monkeypatch.setattr(b, "_launcher_available", lambda: True)

    captured = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    b.run_script(script, ["--flag", "value"], timeout=123)

    assert captured["cmd"] == [
        "micromamba",
        "run",
        "-n",
        "pyscenic_env",
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
    b = build_pyscenic_backend()
    monkeypatch.setattr(b, "_launcher_available", lambda: False)
    with pytest.raises(FileNotFoundError):
        b.run_script(script, [])


def test_run_script_missing_script_raises(tmp_path) -> None:
    b = build_pyscenic_backend()
    with pytest.raises(FileNotFoundError):
        b.run_script(tmp_path / "does_not_exist.py", [])


def test_status_unavailable_when_launcher_missing(monkeypatch) -> None:
    b = build_pyscenic_backend()
    monkeypatch.setattr(b, "_launcher_available", lambda: False)
    st = b.status()
    assert st.available is False
    assert "micromamba" in st.missing


def test_bundled_script_paths_point_into_pyscenic_scripts() -> None:
    assert PYSCENIC_GRN_PY.name == "pyscenic_grn.py"
    assert PYSCENIC_AUCELL_PY.name == "pyscenic_aucell.py"
    assert PYSCENIC_GRN_PY.parent.name == "pyscenic_scripts"
