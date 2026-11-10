"""Tests for the MetaCell2 isolated-env subprocess backend."""

from __future__ import annotations

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


def test_bundled_script_path_points_into_metacells_scripts() -> None:
    assert RUN_METACELL2_PY.name == "run_metacell2.py"
    assert RUN_METACELL2_PY.parent.name == "metacells_scripts"
