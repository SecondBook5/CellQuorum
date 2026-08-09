from __future__ import annotations


def test_methods_registered():
    import cellquorum.cell_cell_communication  # noqa: F401
    from cellquorum.methods.registry import METHOD_REGISTRY

    assert METHOD_REGISTRY.has("cell_cell_communication", "liana")
    assert METHOD_REGISTRY.has("cell_cell_communication", "tensor_c2c")
