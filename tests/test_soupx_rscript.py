"""Smoke test the SoupX R script if SoupX is installed (else skip)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

RSCRIPT = shutil.which("Rscript")


def _soupx_available() -> bool:
    if RSCRIPT is None:
        return False
    r = subprocess.run(
        [RSCRIPT, "--vanilla", "-e", 'quit(status=!requireNamespace("SoupX", quietly=TRUE))'],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not _soupx_available(), reason="Rscript+SoupX unavailable")


def test_soupx_script_exists_and_is_syntactically_valid():
    # The script must exist and parse under R (catch syntax errors early without
    # needing full CellRanger inputs).
    script = (
        Path(__file__).parent.parent
        / "src"
        / "cellquorum"
        / "backends"
        / "r_scripts"
        / "soupx_per_library.R"
    )
    assert script.is_file()
    r = subprocess.run(
        [RSCRIPT, "--vanilla", "-e", f'invisible(parse("{script}"))'],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
