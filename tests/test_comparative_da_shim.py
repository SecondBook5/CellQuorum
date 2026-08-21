"""Move 3: differential_abundance -> comparative.differential_abundance.

The old top-level import paths must still resolve to the exact objects now living
under cellquorum.comparative.
"""

from __future__ import annotations

import importlib

import pytest

_MODULE_PAIRS = [
    ("cellquorum.differential_abundance", "cellquorum.comparative.differential_abundance"),
    (
        "cellquorum.differential_abundance.aggregation",
        "cellquorum.comparative.differential_abundance.aggregation",
    ),
    (
        "cellquorum.differential_abundance.config",
        "cellquorum.comparative.differential_abundance.config",
    ),
    (
        "cellquorum.differential_abundance.milo_method",
        "cellquorum.comparative.differential_abundance.milo_method",
    ),
    (
        "cellquorum.differential_abundance.propeller_method",
        "cellquorum.comparative.differential_abundance.propeller_method",
    ),
    (
        "cellquorum.differential_abundance.proportion_ttest_method",
        "cellquorum.comparative.differential_abundance.proportion_ttest_method",
    ),
    (
        "cellquorum.differential_abundance.sccoda_method",
        "cellquorum.comparative.differential_abundance.sccoda_method",
    ),
]


@pytest.mark.parametrize("old_path,new_path", _MODULE_PAIRS)
def test_old_path_reexports_public_api(old_path, new_path):
    old_mod = importlib.import_module(old_path)
    new_mod = importlib.import_module(new_path)
    assert new_mod.__all__, f"{new_path} defines no public __all__"
    assert old_mod.__all__ == new_mod.__all__
    for name in new_mod.__all__:
        assert getattr(old_mod, name) is getattr(new_mod, name)
