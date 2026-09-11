"""Shared GPU self-gating, used by both scVI and scANVI so they cannot drift on it."""

from __future__ import annotations

import pytest

from cellquorum.core.exceptions import CellQuorumStageError
from cellquorum.stages.integration._gpu_gate import require_gpu


class _Registry:
    def __init__(self, gpu: bool):
        self._gpu = gpu

    def available(self, name: str) -> bool:
        return name == "gpu" and self._gpu


class _Context:
    def __init__(self, registry):
        self.backend_registry = registry


def test_raises_when_no_gpu_backend_available():
    with pytest.raises(CellQuorumStageError, match="scANVI integration requires a GPU"):
        require_gpu(_Context(_Registry(gpu=False)), method_name="scANVI")


def test_passes_silently_when_gpu_backend_available():
    require_gpu(_Context(_Registry(gpu=True)), method_name="scVI")


def test_treats_a_missing_registry_as_no_gpu():
    class _Bare:
        pass

    with pytest.raises(CellQuorumStageError, match="requires a GPU"):
        require_gpu(_Bare(), method_name="scVI")
