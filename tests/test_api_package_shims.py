"""Move 5 (consolidation #187): the public API now lives in ``cellquorum.api``.

These tests prove the relocation is behavior-identical from the caller's side:

  * every former top-level public module (``cellquorum.tl``, ``.pp``, ``.diag``,
    ``.evidence``, ``._notebook``) still imports and re-exports the SAME objects
    as its new ``cellquorum.api.*`` home, and
  * the top-level ``cq.*`` re-exports resolve to those same objects.

Paired with ``tests/test_public_api_contract.py`` (which freezes each
namespace's ``__all__``), this guarantees the move dropped or renamed nothing.
"""

from __future__ import annotations

import importlib

import pytest

# (old top-level path, new api-package path) for each relocated module.
RELOCATIONS = [
    ("cellquorum.tl", "cellquorum.api.tl"),
    ("cellquorum.pp", "cellquorum.api.pp"),
    ("cellquorum.diag", "cellquorum.api.diag"),
    ("cellquorum.evidence", "cellquorum.api.evidence"),
    ("cellquorum._notebook", "cellquorum.api._notebook"),
]


@pytest.mark.parametrize(("old_path", "new_path"), RELOCATIONS)
def test_old_path_reexports_same_objects_as_new_home(old_path, new_path):
    """Each shim re-exports every ``__all__`` symbol as the identical object."""
    old = importlib.import_module(old_path)
    new = importlib.import_module(new_path)
    # The shim advertises exactly the moved module's public surface.
    assert set(getattr(old, "__all__", [])) == set(getattr(new, "__all__", []))
    # And each advertised name is the SAME object, not a copy.
    for name in new.__all__:
        assert getattr(old, name) is getattr(new, name), f"{old_path}.{name} diverged"


def test_api_package_is_the_run_pipeline_home():
    """``cellquorum.api`` is now a package; ``run_pipeline`` resolves through it."""
    import cellquorum
    import cellquorum.api
    import cellquorum.api.pipeline

    assert (
        cellquorum.run_pipeline
        is cellquorum.api.run_pipeline
        is cellquorum.api.pipeline.run_pipeline
    )
    assert cellquorum.api.PipelineRunResult is cellquorum.api.pipeline.PipelineRunResult


@pytest.mark.parametrize("name", ["diag", "evidence", "pp", "tl"])
def test_top_level_namespace_exposes_api_objects(name):
    """``cq.tl`` (etc.) exposes the same public objects as ``cellquorum.api.tl``.

    A physical shim module exists at ``cellquorum/<name>.py`` so that the
    submodule-import form ``import cellquorum.tl`` keeps resolving; once that
    runs, ``cellquorum.<name>`` is that shim (not the api submodule itself), so
    identity is asserted on the re-exported symbols, not the module object.
    """
    import cellquorum
    import cellquorum.api

    top = getattr(cellquorum, name)
    api_mod = getattr(cellquorum.api, name)
    assert set(top.__all__) == set(api_mod.__all__)
    for sym in api_mod.__all__:
        assert getattr(top, sym) is getattr(api_mod, sym)


@pytest.mark.parametrize(
    "old_path", ["cellquorum.tl", "cellquorum.pp", "cellquorum.diag", "cellquorum.evidence"]
)
def test_submodule_import_form_still_resolves(old_path):
    """``import cellquorum.tl`` (submodule form, not attribute access) works."""
    assert importlib.import_module(old_path) is not None
