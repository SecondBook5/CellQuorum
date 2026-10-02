"""Tests for the PyUCell subprocess backend (no real micromamba/pyucell)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cellquorum.backends.pyucell_backend import (
    PYUCELL_SCORE_PY,
    build_pyucell_backend,
)


def test_build_defaults() -> None:
    b = build_pyucell_backend()
    assert b.name == "pyucell"
    assert b.kind == "external"
    # Deliberately reuses celloracle_env rather than a dedicated env: pyucell's
    # own numpy<2 pin is already satisfied there (verified live), and every
    # other dependency it needs (anndata, scanpy>=1.9, scipy, joblib) was
    # already installed -- a `pip install pyucell` there added nothing else.
    assert b.env_name == "celloracle_env"
    assert b.launcher == "micromamba"


def test_run_script_builds_micromamba_argv(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("print('hi')\n")
    b = build_pyucell_backend()

    captured = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003, ANN202
        captured["cmd"] = cmd
        captured["check"] = kwargs.get("check")

        class R:  # noqa: D401
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(
        "cellquorum.backends.pyucell_backend.resolve_launcher",
        lambda _name: "/usr/bin/micromamba",
    )
    monkeypatch.setattr("cellquorum.backends.pyucell_backend.subprocess.run", fake_run)

    b.run_script(script, ["--h5ad", "x.h5ad"], timeout=123)
    assert captured["cmd"][:5] == [
        "/usr/bin/micromamba",
        "run",
        "-n",
        "celloracle_env",
        "python",
    ]
    assert str(script) in captured["cmd"]
    assert captured["cmd"][-2:] == ["--h5ad", "x.h5ad"]
    assert captured["check"] is False


def test_run_script_missing_script_raises(tmp_path: Path) -> None:
    b = build_pyucell_backend()
    with pytest.raises(FileNotFoundError):
        b.run_script(tmp_path / "does_not_exist.py", [])


def test_run_script_missing_launcher_raises(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("x=1\n")
    b = build_pyucell_backend()
    monkeypatch.setattr("cellquorum.backends.pyucell_backend.resolve_launcher", lambda _n: None)
    with pytest.raises(FileNotFoundError):
        b.run_script(script, [])


def test_invalid_module_name_rejected() -> None:
    b = build_pyucell_backend()
    with pytest.raises(ValueError):
        b._py_module_available("bad name; rm -rf")


def test_score_script_path_points_into_scripts_dir() -> None:
    assert PYUCELL_SCORE_PY.name == "ucell_score.py"
    assert PYUCELL_SCORE_PY.parent.name == "pyucell_scripts"
