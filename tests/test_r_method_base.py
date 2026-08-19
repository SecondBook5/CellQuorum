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
    def __init__(self, ok):
        self._ok = ok

    def _r_package_available(self, pkg):
        return self._ok


def test_resolves_backend_when_all_available(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/Rscript")
    backend = _Backend(ok=True)
    b, skip = _RDummy()._resolve_rscript_backend(_Ctx(backend))
    assert b is backend and skip is None


def test_skips_when_rscript_absent(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    b, skip = _RDummy()._resolve_rscript_backend(_Ctx(_Backend(True)))
    assert b is None and isinstance(skip, MethodSkip)
    assert "rscript" in skip.reason.lower()


def test_skips_when_r_package_absent(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/Rscript")
    b, skip = _RDummy()._resolve_rscript_backend(_Ctx(_Backend(ok=False)))
    assert b is None and isinstance(skip, MethodSkip)
    assert skip.details.get("r_package") == "edgeR"


def test_config_overrides_r_package(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/Rscript")
    seen = {}

    class _RecBackend:
        def _r_package_available(self, pkg):
            seen["pkg"] = pkg
            return False

    b, skip = _RDummy()._resolve_rscript_backend(_Ctx(_RecBackend()), {"r_package": "DESeq2"})
    assert seen["pkg"] == "DESeq2"
    assert b is None and skip.details.get("r_package") == "DESeq2"
