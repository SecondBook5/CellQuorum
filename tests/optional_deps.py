"""Skip marks for tests that need an optional third-party package.

The package is not installed in every environment (the CI test job installs only ``.[dev]``), and
a test that needs it should say so and skip, not fail with ModuleNotFoundError. Markers deselect;
skipif handles availability, which is what this is.
"""

from __future__ import annotations

import importlib.util

import pytest


def _missing(*modules: str) -> list[str]:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def requires(*modules: str) -> pytest.MarkDecorator:
    """Skip the test unless every named top-level module is importable."""
    missing = _missing(*modules)
    return pytest.mark.skipif(
        bool(missing), reason=f"optional dependency not installed: {', '.join(missing)}"
    )


def requires_parquet() -> pytest.MarkDecorator:
    """Skip the test unless pandas has a parquet engine (pyarrow or fastparquet)."""
    absent = bool(_missing("pyarrow")) and bool(_missing("fastparquet"))
    return pytest.mark.skipif(absent, reason="optional dependency not installed: pyarrow")
