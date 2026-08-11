from cellquorum.backends.registry import build_default_backend_registry


def test_hdwgcna_backend_registered() -> None:
    registry = build_default_backend_registry()
    assert registry.has("hdwgcna_r")
    assert "hdwgcna_r" in registry.names()
