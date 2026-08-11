"""Tests for the CellOracle subprocess backend (no real micromamba/celloracle)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cellquorum.backends.celloracle_backend import (
    CELLORACLE_KO_PY,
    build_celloracle_backend,
)


def test_build_defaults() -> None:
    b = build_celloracle_backend()
    assert b.name == "celloracle"
    assert b.kind == "external"
    assert b.env_name == "celloracle_env"
    assert b.launcher == "micromamba"
    assert b.script_timeout_seconds == 10800


def test_run_script_builds_micromamba_argv(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("print('hi')\n")
    b = build_celloracle_backend()

    captured = {}

    def fake_which(_name):  # noqa: ANN001, ANN202
        return "/usr/bin/micromamba"

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003, ANN202
        captured["cmd"] = cmd
        captured["check"] = kwargs.get("check")

        class R:  # noqa: D401
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr("cellquorum.backends.celloracle_backend.shutil.which", fake_which)
    monkeypatch.setattr("cellquorum.backends.celloracle_backend.subprocess.run", fake_run)

    b.run_script(script, ["--h5ad", "x.h5ad"], timeout=123)
    assert captured["cmd"][:5] == ["micromamba", "run", "-n", "celloracle_env", "python"]
    assert str(script) in captured["cmd"]
    assert captured["cmd"][-2:] == ["--h5ad", "x.h5ad"]
    assert captured["check"] is False


def test_run_script_missing_script_raises(tmp_path: Path) -> None:
    b = build_celloracle_backend()
    with pytest.raises(FileNotFoundError):
        b.run_script(tmp_path / "does_not_exist.py", [])


def test_run_script_missing_launcher_raises(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("x=1\n")
    b = build_celloracle_backend()
    monkeypatch.setattr("cellquorum.backends.celloracle_backend.shutil.which", lambda _n: None)
    with pytest.raises(FileNotFoundError):
        b.run_script(script, [])


def test_invalid_module_name_rejected() -> None:
    b = build_celloracle_backend()
    with pytest.raises(ValueError):
        b._py_module_available("bad name; rm -rf")


def test_ko_script_path_points_into_scripts_dir() -> None:
    assert CELLORACLE_KO_PY.name == "celloracle_ko.py"
    assert CELLORACLE_KO_PY.parent.name == "celloracle_scripts"
