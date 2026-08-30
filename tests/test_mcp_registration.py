"""Registration test for the multicellular_programs (DIALOGUE) method."""

from __future__ import annotations


def test_dialogue_registered():
    import cellquorum.stages.comparative.multicellular_programs  # noqa: F401
    from cellquorum.methods.registry import METHOD_REGISTRY

    assert METHOD_REGISTRY.has("multicellular_programs", "dialogue")
