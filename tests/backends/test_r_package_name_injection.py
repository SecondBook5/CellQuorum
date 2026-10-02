"""An R package name comes from configuration, so it must never be able to inject R code.

Each backend builds ``requireNamespace('<name>')`` for ``Rscript -e``. A name such as
``x'); system('...'); ('`` would otherwise close the string and run its own code, which
SECURITY.md lists as in scope. The names are validated before any process is started.
"""

from __future__ import annotations

import subprocess

import pytest

from cellquorum.backends.hdwgcna_backend import HdwgcnaBackend
from cellquorum.backends.r import RBackend
from cellquorum.backends.rscript import RscriptBackend

INJECTIONS = [
    "x'); system('touch /tmp/pwned'); ('",
    'x"); system("id"); ("',
    "edgeR; quit()",
    "../../etc/passwd",
    "edge R",
    "1abc",
    "",
]


def _forbid_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError(f"a process was started for an invalid package name: {args}")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)


@pytest.mark.parametrize("backend_class", [RscriptBackend, RBackend, HdwgcnaBackend])
@pytest.mark.parametrize("name", INJECTIONS)
def test_an_invalid_package_name_is_rejected_before_any_process_starts(
    backend_class, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_processes(monkeypatch)
    with pytest.raises(ValueError, match="Invalid R package name"):
        backend_class()._r_package_available(name)


@pytest.mark.parametrize("backend_class", [RscriptBackend, RBackend, HdwgcnaBackend])
@pytest.mark.parametrize("name", ["edgeR", "miloR", "SingleCellExperiment", "data.table", "R6"])
def test_ordinary_bioconductor_and_cran_names_are_accepted(backend_class, name: str) -> None:
    assert backend_class._valid_r_package_name(name)
