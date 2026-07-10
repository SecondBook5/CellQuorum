"""Tests for resolving a stage's config sub-block from a context."""

from __future__ import annotations

from cellquorum.methods.context_access import resolve_stage_config


class _PydanticLike:
    # Minimal stand-in for a pydantic sub-model (has model_dump).
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return dict(self._data)


class _Config:
    def __init__(self):
        self.dimensionality = _PydanticLike({"n_pcs": "auto"})


class _CtxPydantic:
    config = _Config()


class _CtxDict:
    config = {"dimensionality": {"n_pcs": 30}}


class _CtxNone:
    config = None


def test_resolve_from_pydantic_submodel():
    assert resolve_stage_config(_CtxPydantic(), "dimensionality") == {"n_pcs": "auto"}


def test_resolve_from_dict():
    assert resolve_stage_config(_CtxDict(), "dimensionality") == {"n_pcs": 30}


def test_resolve_missing_returns_empty():
    assert resolve_stage_config(_CtxPydantic(), "clustering") == {}
    assert resolve_stage_config(_CtxDict(), "clustering") == {}
    assert resolve_stage_config(_CtxNone(), "dimensionality") == {}
