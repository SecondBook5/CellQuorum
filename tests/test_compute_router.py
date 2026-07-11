"""Tests for the GPU compute router (capability + config decision)."""

from __future__ import annotations

from cellquorum.compute.router import (
    gpu_compute_available,
    resolve_compute,
    should_use_gpu,
)


class _Compute:
    def __init__(self, backend="auto", prefer_gpu=True, fallback_to_cpu=True):
        self.backend = backend
        self.prefer_gpu = prefer_gpu
        self.fallback_to_cpu = fallback_to_cpu


class _Ctx:
    def __init__(self, compute):
        self.config = type("C", (), {"compute": compute})()


def test_gpu_available_probe_never_raises():
    # Whatever the environment, the probe returns a bool and does not raise.
    assert isinstance(gpu_compute_available(), bool)


def test_cpu_backend_forces_cpu_even_with_gpu(monkeypatch):
    # The escape hatch: backend="cpu" -> never use GPU, regardless of hardware.
    monkeypatch.setattr("cellquorum.compute.router.gpu_compute_available", lambda: True)
    assert should_use_gpu(_Ctx(_Compute(backend="cpu"))) is False


def test_auto_prefers_gpu_when_available(monkeypatch):
    monkeypatch.setattr("cellquorum.compute.router.gpu_compute_available", lambda: True)
    assert should_use_gpu(_Ctx(_Compute(backend="auto", prefer_gpu=True))) is True


def test_auto_no_gpu_falls_to_cpu(monkeypatch):
    monkeypatch.setattr("cellquorum.compute.router.gpu_compute_available", lambda: False)
    assert should_use_gpu(_Ctx(_Compute(backend="auto", prefer_gpu=True))) is False


def test_explicit_gpu_backend(monkeypatch):
    monkeypatch.setattr("cellquorum.compute.router.gpu_compute_available", lambda: True)
    assert should_use_gpu(_Ctx(_Compute(backend="gpu"))) is True


def test_missing_config_defaults_to_auto_prefer_gpu(monkeypatch):
    monkeypatch.setattr("cellquorum.compute.router.gpu_compute_available", lambda: True)

    class _Bare:
        config = None

    assert should_use_gpu(_Bare()) is True


def test_resolve_compute_shape(monkeypatch):
    monkeypatch.setattr("cellquorum.compute.router.gpu_compute_available", lambda: False)
    r = resolve_compute(_Ctx(_Compute(backend="auto", fallback_to_cpu=True)))
    assert r == {"use_gpu": False, "fallback_to_cpu": True}
