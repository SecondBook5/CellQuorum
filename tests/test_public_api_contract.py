"""Freezes the documented public API surface so consolidation moves cannot
silently drop or rename an exported symbol.

The contract is each namespace's declared ``__all__`` — the surface the
maintainer intends users to import, and the guarantee that "usage stays
identical" across the consolidation. ``__all__`` is frozen rather than
``dir()`` on purpose:

  * ``dir()`` on the top-level ``cellquorum`` package is import-order
    dependent — a submodule name leaks into the namespace the moment any
    prior test imports it (e.g. ``cellquorum.reports``), so a ``dir()``-based
    freeze passes in isolation but fails in the full suite.
  * ``dir()`` also captures incidental imports (``Path``, ``annotations``,
    ``TYPE_CHECKING``, backend/config classes re-imported for convenience)
    that are NOT declared public API and which the consolidation folds are
    free to rearrange behind an unchanged facade.

``__all__`` is the static, declared contract: leakage-immune, and stable
across folds that relocate implementations behind the same facade. The
consolidation deletes/merges internal top-level packages (``trajectory_viz``,
``feature_selection``, ``ambient_correction``, …) that were never part of
``__all__``; freezing ``__all__`` protects the real public surface without
freezing implementation-detail package names the plan is restructuring.
"""

import importlib

import pytest

# Declared public surface (``__all__``) captured at baseline commit
# 11a81032e32d1445fecea9675aa63edeb952a3a9.
EXPECTED = {
    "cellquorum": {
        "__version__",
        "diag",
        "evidence",
        "pp",
        "run_pipeline",
        "tl",
        "utils",
    },
    "cellquorum.api": {
        "PipelineRunResult",
        "run_pipeline",
    },
    "cellquorum.utils": {
        "PriorFetchError",
        "PseudobulkResult",
        "aggregate_pseudobulk",
        "de_table_to_ranking",
        "get_net",
    },
    "cellquorum.pp": {
        "correct_ambient",
        "normalize",
        "qc",
        "select_features",
    },
    "cellquorum.tl": {
        "adjudicate",
        "annotate",
        "cluster",
        "integrate",
        "population_identity",
        "reduce_dimensions",
        "reference_map",
        "subcluster",
    },
}


@pytest.mark.parametrize("module_name", sorted(EXPECTED))
def test_public_surface_is_stable(module_name):
    """Each public namespace's declared ``__all__`` is frozen exactly.

    Bidirectional set-equality catches both dropped/renamed symbols (a
    consolidation move that loses an export) and invented ones (surface
    creep). Frozen against ``__all__`` — not ``dir()`` — so it is immune to
    submodule-attribute leakage and to the folds relocating implementations
    behind an unchanged facade.
    """
    module = importlib.import_module(module_name)
    declared = set(getattr(module, "__all__", []))
    expected = EXPECTED[module_name]
    assert declared == expected, (
        f"{module_name}.__all__ changed:\n"
        f"  dropped: {sorted(expected - declared)}\n"
        f"  added:   {sorted(declared - expected)}"
    )


@pytest.mark.parametrize("module_name", sorted(EXPECTED))
def test_declared_symbols_are_accessible(module_name):
    """Every name a namespace declares in ``__all__`` must actually resolve.

    Guards against an ``__all__`` that advertises a symbol the module no
    longer provides — the export would be listed but ``from pkg import name``
    (and ``from pkg import *``) would fail at runtime.
    """
    module = importlib.import_module(module_name)
    missing = sorted(name for name in EXPECTED[module_name] if not hasattr(module, name))
    assert not missing, f"{module_name} declares but does not provide: {missing}"
