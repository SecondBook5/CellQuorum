"""Move 2: qc.ambient -> ambient_correction relocation keeps old import paths valid."""

from __future__ import annotations


def test_old_config_path_resolves_to_moved_object():
    from cellquorum.ambient_correction.config import AmbientCorrectionConfig as New
    from cellquorum.qc.ambient.config import AmbientCorrectionConfig as Old

    assert Old is New


def test_old_stage_path_resolves_to_moved_object():
    from cellquorum.ambient_correction.stage import AmbientCorrectionStage as New
    from cellquorum.qc.ambient.stage import AmbientCorrectionStage as Old

    assert Old is New


def test_old_package_reexports_stage():
    from cellquorum.ambient_correction import AmbientCorrectionStage as New
    from cellquorum.qc.ambient import AmbientCorrectionStage as Old

    assert Old is New


def test_old_soupx_symbols_resolve_to_moved_objects():
    from cellquorum.ambient_correction import soupx as new_mod
    from cellquorum.qc.ambient import soupx as old_mod

    for name in new_mod.__all__:
        assert getattr(old_mod, name) is getattr(new_mod, name)
