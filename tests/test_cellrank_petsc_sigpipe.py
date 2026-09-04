"""PETSc must not turn a closed pipe into a process abort.

``PetscInitialize`` installs its own signal handlers, and its SIGPIPE handler
calls ``MPI_Abort``. Python ignores SIGPIPE by default precisely so that a closed
downstream reader surfaces as ``BrokenPipeError`` in Python; under PETSc's handler
the same benign event kills the process from C, past every ``except``. Observed
twice: ``cellquorum run | head`` died as an MPI abort, and the full test suite
aborted during teardown before pytest could print its summary or its exit code.

These run in subprocesses on purpose. The failure mode IS process death, so it
cannot be observed from inside the process under test.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

petsc4py = pytest.importorskip("petsc4py")
slepc4py = pytest.importorskip("slepc4py")

# Raise SIGPIPE at ourselves rather than building a real broken pipe: same signal,
# no dependence on when a reader happens to close.
_RAISE_SIGPIPE = """
import os, signal
{setup}
os.kill(os.getpid(), signal.SIGPIPE)
print("SURVIVED")
"""


def _run(setup: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", _RAISE_SIGPIPE.format(setup=setup)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_sigpipe_survives_the_probe_then_cellranks_own_petsc_import() -> None:
    """The real sequence: engine probes, then CellRank imports PETSc itself.

    Probing alone was never the problem -- it imports ``petsc4py`` but not
    ``petsc4py.PETSc``, so nothing initializes. The handler arrives moments later
    when CellRank (via pyGPCCA) does its own ``from petsc4py import PETSc``, which
    is what this replays. For the restore to be possible at all, the probe has to
    have initialized PETSc already, so that this second import is a no-op.
    """
    done = _run(
        "from cellquorum.stages.trajectory._cellrank import _slepc_available\n"
        "assert _slepc_available() is True\n"
        "from petsc4py import PETSc  # what CellRank/pyGPCCA does\n"
        "PETSc.Sys.getVersion()"
    )
    assert "SURVIVED" in done.stdout, f"stdout={done.stdout!r} stderr={done.stderr!r}"
    assert "MPI_Abort" not in done.stderr
    assert done.returncode == 0


def test_probe_alone_leaves_sigpipe_ignorable() -> None:
    """And the probe must not leave the handler installed on its own either."""
    done = _run(
        "from cellquorum.stages.trajectory._cellrank import _slepc_available\n"
        "assert _slepc_available() is True"
    )
    assert "SURVIVED" in done.stdout, f"stdout={done.stdout!r} stderr={done.stderr!r}"
    assert "MPI_Abort" not in done.stderr
    assert done.returncode == 0


def test_unguarded_petsc_import_is_why_the_probe_has_to_do_this() -> None:
    """The hazard is real: a bare PETSc import alone installs the handler.

    This test keeps the one above from being vacuous. If it ever FAILS, petsc4py
    has stopped hijacking SIGPIPE and the restore in ``_slepc_available`` can be
    deleted -- a failure here is good news, not a regression.
    """
    done = _run("from petsc4py import PETSc\nPETSc.Sys.getVersion()")
    assert "SURVIVED" not in done.stdout
    assert "MPI_Abort" in done.stderr
