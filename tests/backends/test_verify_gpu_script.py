"""Environment verification must fail when a requested GPU capability fails."""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest
from repo_paths import REPO_ROOT


@pytest.fixture
def verifier():
    path = REPO_ROOT / "scripts" / "verify_gpu.py"
    spec = importlib.util.spec_from_file_location("verify_gpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scvi_import_failure_is_failure(verifier, monkeypatch):
    torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="12"),
        backends=SimpleNamespace(cuda=SimpleNamespace(is_built=lambda: True)),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "test GPU"),
        zeros=lambda *args, **kwargs: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    # Supply a successful device operation independently of scVI importability.
    class Tensor:
        def __add__(self, other):
            return self

        def sum(self):
            return self

        def item(self):
            return 8

    torch.zeros = lambda *args, **kwargs: Tensor()
    monkeypatch.setitem(sys.modules, "scvi", None)
    assert verifier.verify_scvi() == 1


def test_rapids_kernel_failure_is_failure(verifier, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("CUDA headers unavailable")

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(arange=fail, float32="float32"))
    monkeypatch.setitem(sys.modules, "rapids_singlecell", SimpleNamespace())
    assert verifier.verify_rapids() == 1


def test_all_profiles_run_even_when_first_fails(verifier, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["verify_gpu.py", "--profile", "all"])
    monkeypatch.setattr(verifier, "verify_rapids", lambda: calls.append("rapids") or 1)
    monkeypatch.setattr(verifier, "verify_scvi", lambda: calls.append("scvi") or 0)
    assert verifier.main() == 1
    assert calls == ["rapids", "scvi"]
