"""Move 3: multicellular_programs -> comparative.multicellular_programs.

The old top-level import paths must still resolve to the exact objects now living
under cellquorum.comparative.
"""

from __future__ import annotations

import importlib

import pytest

_MODULE_PAIRS = [
    ("cellquorum.multicellular_programs", "cellquorum.comparative.multicellular_programs"),
    (
        "cellquorum.multicellular_programs.config",
        "cellquorum.comparative.multicellular_programs.config",
    ),
    (
        "cellquorum.multicellular_programs.dialogue_method",
        "cellquorum.comparative.multicellular_programs.dialogue_method",
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
