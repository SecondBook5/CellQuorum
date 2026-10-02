from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod


class _RDummy(RAnalysisMethod):
    name = "rdummy"
    stage_category = "test"
    r_package = "edgeR"

    def input_contract(self, config): ...
    def _run(self, adata, config, context): ...


class _Reg:
    def __init__(self, backend):
        self._b = backend

    def get(self, name):
        if name == "rscript":
            return self._b
        raise KeyError(name)


class _Ctx:
    def __init__(self, backend):
        self.backend_registry = _Reg(backend)


class _Backend:
    """Stand-in Rscript backend exposing the interface _resolve depends on.

    Mirrors the real ``RscriptBackend`` primitives: ``_rscript_available()``
    (Rscript reachable at the configured path) and ``_r_package_available()``
    (the R package installed).
    """

    def __init__(self, ok=True, avail=True):
        self._ok = ok
        self._avail = avail
        self.rscript_path = "Rscript"

    def _rscript_available(self):
        return self._avail

    def _r_package_available(self, pkg):
        return self._ok


def test_resolves_backend_when_all_available():
    backend = _Backend(ok=True, avail=True)
    b, skip = _RDummy()._resolve_rscript_backend(_Ctx(backend))
    assert b is backend and skip is None


def test_skips_when_rscript_absent():
    # Rscript not reachable at the configured path -> _rscript_available() False.
    b, skip = _RDummy()._resolve_rscript_backend(_Ctx(_Backend(ok=True, avail=False)))
    assert b is None and isinstance(skip, MethodSkip)
    assert "rscript" in skip.reason.lower()


def test_skips_when_r_package_absent():
    # Rscript reachable, but the R package is missing.
    b, skip = _RDummy()._resolve_rscript_backend(_Ctx(_Backend(ok=False, avail=True)))
    assert b is None and isinstance(skip, MethodSkip)
    assert skip.details.get("r_package") == "edgeR"


def test_config_overrides_r_package():
    seen = {}

    class _RecBackend:
        rscript_path = "Rscript"

        def _rscript_available(self):
            return True

        def _r_package_available(self, pkg):
            seen["pkg"] = pkg
            return False

    b, skip = _RDummy()._resolve_rscript_backend(_Ctx(_RecBackend()), {"r_package": "DESeq2"})
    assert seen["pkg"] == "DESeq2"
    assert b is None and skip.details.get("r_package") == "DESeq2"


def test_availability_honors_configured_rscript_path(monkeypatch):
    """A configured rscript_path is honored end-to-end.

    Regression guard for the dead-``r.rscript_path`` bug: the availability gate
    must check the backend's *configured* Rscript path, not a hardcoded bare
    ``Rscript`` on PATH. Here bare ``Rscript`` is absent from PATH but an
    absolute configured path resolves; the real backend must report available
    and the method must clear the availability gate against it.
    """
    import shutil as _shutil

    from cellquorum.backends.rscript import build_rscript_backend

    custom = "/opt/envs/cellquorum-r/bin/Rscript"
    # PATH has NO bare 'Rscript'; only the configured absolute path resolves.
    monkeypatch.setattr(_shutil, "which", lambda name: custom if name == custom else None)

    backend = build_rscript_backend(rscript_path=custom)
    # The primitive the availability gate uses honors the configured path.
    assert backend._rscript_available(), "configured rscript_path must be honored"

    # Stub the R-package probe so the fake path is never exec'd; the point is
    # that the method clears the *availability* gate (no 'Rscript unavailable'
    # skip) and only skips later on the package.
    monkeypatch.setattr(backend, "_r_package_available", lambda pkg: False)
    b, skip = _RDummy()._resolve_rscript_backend(_Ctx(backend))
    assert b is None and isinstance(skip, MethodSkip)
    assert "rscript unavailable" not in skip.reason.lower()
    assert skip.details.get("r_package") == "edgeR"
