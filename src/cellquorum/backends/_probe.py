"""Process-level cache for subprocess backend availability probes.

Availability probes are expensive and repeated. Asking "is celloracle importable in
``celloracle_env``?" means running ``micromamba run -n celloracle_env python -c
'import celloracle'``, and importing celloracle costs about **7.7 seconds** — it
pulls scanpy, gimmemotifs, and a stack of the rest. Nothing cached the answer, so
every caller paid it again:

* ``BackendRegistry.to_status_table()`` measured 8.5s, 7.3s, and 7.0s on three
  consecutive calls in one process.
* ``cellquorum plan`` took ~10s, nearly all of it this one probe.
* Planner and CLI tests (``test_planner``, ``test_cli``, ``test_backend_registry``,
  ``test_stage_catalog``, ``test_ambient_config``, ``test_reference_mapping_config``)
  each took 7–8s for the same reason, despite doing no analysis.

The result is safe to cache for the lifetime of the process: a conda environment does
not appear or vanish partway through a run, and a run that installed one mid-flight
would be non-reproducible for much worse reasons. The cache is keyed on everything
that can change the answer, so two backends pointing at different environments do not
share an entry.

The cache lives here — at module scope — rather than on the backend instances,
because ``build_default_backend_registry()`` constructs fresh backend objects on every
call. An instance-level cache would be discarded exactly when it was about to help.

Backends keep their own ``_py_module_available`` methods as thin wrappers over these
functions: those methods are the documented seam that tests monkeypatch, and argument
validation stays in the caller so an invalid module name still raises before anything
is cached.
"""

from __future__ import annotations

import subprocess
from functools import cache

# Note on scope: only the *subprocess* probe is cached here. The sibling
# `_launcher_available` checks on each backend use `shutil.which`, which measured
# 0.00s, so caching them would add indirection for no gain.


@cache
def env_python_module_available(
    launcher: str,
    env_name: str,
    module_name: str,
    timeout_seconds: int,
) -> bool:
    """
    Return whether a Python module imports inside a launcher-managed environment.

    The result is cached for the lifetime of the process, keyed on all four
    arguments. Callers must validate ``module_name`` before calling: this function
    interpolates it into an ``import`` expression.

    Args:
        launcher: Launcher executable used to enter the environment.
        env_name: Environment name to run inside.
        module_name: Python module to attempt to import.
        timeout_seconds: Per-probe subprocess timeout.

    Returns:
        True if the import succeeded, False if it failed, timed out, or the launcher
        is missing.
    """

    # A missing launcher, a failed import, and a timeout are all "not available".
    # Returning False rather than raising is what lets a stage skip with a recorded
    # reason instead of crashing the run.
    try:
        result = subprocess.run(
            [
                launcher,
                "run",
                "-n",
                env_name,
                "python",
                "-c",
                f"import {module_name}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    # A zero exit status means the module imported cleanly.
    return result.returncode == 0


def clear_probe_cache() -> None:
    """
    Drop every cached probe result.

    Intended for tests that need a probe to run again — for example, one asserting
    that a backend reports unavailable after its environment is monkeypatched away.
    Production code should not need this: backend availability is fixed for the
    lifetime of a run.
    """

    # Clear each cache explicitly so adding a new probe here without clearing it fails
    # loudly in review rather than silently leaking state between tests.
    env_python_module_available.cache_clear()
