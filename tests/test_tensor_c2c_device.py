"""Tests for tensor_c2c device resolution (compute.prefer_gpu -> cuda/cpu).

These lock down the wiring that lets the tensor decomposition use the GPU:
an explicit config device wins, otherwise compute.prefer_gpu decides, and a
CUDA request always degrades to CPU when no usable GPU is present.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cellquorum.stages.cell_cell_communication.tensor_c2c_method import TensorCell2CellMethod


def _context(prefer_gpu: bool):
    return SimpleNamespace(config=SimpleNamespace(compute=SimpleNamespace(prefer_gpu=prefer_gpu)))


def _set_cuda(monkeypatch, available: bool):
    """Force torch.cuda.is_available() to a known value."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def test_prefer_gpu_false_resolves_cpu(monkeypatch):
    _set_cuda(monkeypatch, True)  # GPU present but not preferred
    m = TensorCell2CellMethod()
    assert m._resolve_device({}, _context(prefer_gpu=False)) == "cpu"


def test_prefer_gpu_true_with_cuda_resolves_cuda(monkeypatch):
    _set_cuda(monkeypatch, True)
    m = TensorCell2CellMethod()
    assert m._resolve_device({}, _context(prefer_gpu=True)) == "cuda"


def test_prefer_gpu_true_without_cuda_falls_back_to_cpu(monkeypatch):
    _set_cuda(monkeypatch, False)
    m = TensorCell2CellMethod()
    assert m._resolve_device({}, _context(prefer_gpu=True)) == "cpu"


def test_explicit_cpu_overrides_prefer_gpu(monkeypatch):
    _set_cuda(monkeypatch, True)
    m = TensorCell2CellMethod()
    assert m._resolve_device({"device": "cpu"}, _context(prefer_gpu=True)) == "cpu"


def test_explicit_gpu_alias_resolves_cuda_when_available(monkeypatch):
    _set_cuda(monkeypatch, True)
    m = TensorCell2CellMethod()
    assert m._resolve_device({"device": "gpu"}, _context(prefer_gpu=False)) == "cuda"


def test_missing_compute_config_defaults_to_cpu():
    """No compute block (prefer_gpu unknown) must not crash; defaults to CPU."""
    m = TensorCell2CellMethod()
    ctx = SimpleNamespace(config=SimpleNamespace())
    assert m._resolve_device({}, ctx) == "cpu"
