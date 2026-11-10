"""Tests for the PyUCell subprocess backend (no real micromamba/pyucell)."""

from __future__ import annotations

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


def test_score_script_path_points_into_scripts_dir() -> None:
    assert PYUCELL_SCORE_PY.name == "ucell_score.py"
    assert PYUCELL_SCORE_PY.parent.name == "pyucell_scripts"
