"""Move 3: differential_expression -> comparative.differential_expression.

The old top-level import paths must still resolve to the exact objects now living
under cellquorum.comparative, so pre-#187 analysis scripts run unchanged.
"""

from __future__ import annotations

import importlib

import pytest

# (old public module path, canonical module path). stage.py modules are intentionally
# absent: they have no out-of-repo consumer and are not shimmed.
_MODULE_PAIRS = [
    ("cellquorum.differential_expression", "cellquorum.comparative.differential_expression"),
    (
        "cellquorum.differential_expression.config",
        "cellquorum.comparative.differential_expression.config",
    ),
    (
        "cellquorum.differential_expression.pseudobulk",
        "cellquorum.comparative.differential_expression.pseudobulk",
    ),
    (
        "cellquorum.differential_expression.pseudobulk_edger_method",
        "cellquorum.comparative.differential_expression.pseudobulk_edger_method",
    ),
    (
        "cellquorum.differential_expression.viz",
        "cellquorum.comparative.differential_expression.viz",
    ),
    (
        "cellquorum.differential_expression.viz.config",
        "cellquorum.comparative.differential_expression.viz.config",
    ),
    (
        "cellquorum.differential_expression.viz.plots",
        "cellquorum.comparative.differential_expression.viz.plots",
    ),
    (
        "cellquorum.differential_expression.viz.volcano_viz",
        "cellquorum.comparative.differential_expression.viz.volcano_viz",
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


def test_key_downstream_symbol_identity():
    # The path one analysis script imports directly.
    from cellquorum.comparative.differential_expression.pseudobulk import (
        aggregate_pseudobulk as new,
    )
    from cellquorum.differential_expression.pseudobulk import aggregate_pseudobulk as old

    assert old is new
