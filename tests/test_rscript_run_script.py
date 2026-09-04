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


def test_run_script_does_not_litter_the_callers_directory(tmp_path, monkeypatch):
    """R's default device writes ``Rplots.pdf`` into the working directory.

    Several packages we call draw as a side effect, so a script that inherits
    this process's directory drops an ``Rplots.pdf`` wherever CellQuorum was
    launched from — the repository root during a test run, the hypothesis repo
    beside the manifests during a real one. This is where that gets absorbed.
    """
    script = tmp_path / "draws.R"
    script.write_text("plot(1:10)\n")

    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = RscriptBackend().run_script(script, [])

    assert result.returncode == 0, result.stderr
    assert list(workdir.iterdir()) == []


def test_run_script_still_sees_absolute_paths_from_the_scratch_directory(tmp_path):
    """The scratch CWD must not break the file exchange the adapter is built on.

    Every R script here receives absolute paths as arguments, so moving the
    working directory is invisible to them — this pins that assumption.
    """
    script = tmp_path / "roundtrip.R"
    script.write_text(
        "args <- commandArgs(trailingOnly=TRUE)\n"
        "writeLines(readLines(args[1]), args[2])\n"
        "stopifnot(!file.exists(basename(args[1])))\n"
    )
    src = tmp_path / "in.txt"
    src.write_text("payload\n")
    dst = tmp_path / "out.txt"

    result = RscriptBackend().run_script(script, [str(src), str(dst)])

    assert result.returncode == 0, result.stderr
    assert dst.read_text().strip() == "payload"
