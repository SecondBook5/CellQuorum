"""Tests for the Rscript run_script adapter."""

from __future__ import annotations

import shutil

import pytest

from cellquorum.backends.rscript import RscriptBackend

RSCRIPT = shutil.which("Rscript")
pytestmark = pytest.mark.skipif(RSCRIPT is None, reason="Rscript not available")


def test_run_script_executes_and_passes_args(tmp_path):
    # A tiny R script that writes its first arg to a file.
    script = tmp_path / "echo.R"
    script.write_text("args <- commandArgs(trailingOnly=TRUE)\n" "writeLines(args[1], args[2])\n")
    out = tmp_path / "out.txt"
    backend = RscriptBackend()
    result = backend.run_script(script, ["hello", str(out)])
    assert result.returncode == 0
    assert out.read_text().strip() == "hello"


def test_run_script_missing_script_raises(tmp_path):
    backend = RscriptBackend()
    with pytest.raises(FileNotFoundError, match="script"):
        backend.run_script(tmp_path / "nope.R", [])


def test_run_script_nonzero_returncode_is_returned(tmp_path):
    script = tmp_path / "fail.R"
    script.write_text('stop("boom")\n')
    backend = RscriptBackend()
    result = backend.run_script(script, [])
    # Non-zero returncode is returned (not raised) so the caller can inspect stderr.
    assert result.returncode != 0
    assert "boom" in result.stderr
