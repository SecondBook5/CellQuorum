"""Tests for the pySCENIC isolated-env subprocess backend."""

from __future__ import annotations

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


def test_bundled_script_paths_point_into_pyscenic_scripts() -> None:
    assert PYSCENIC_GRN_PY.name == "pyscenic_grn.py"
    assert PYSCENIC_AUCELL_PY.name == "pyscenic_aucell.py"
    assert PYSCENIC_GRN_PY.parent.name == "pyscenic_scripts"
