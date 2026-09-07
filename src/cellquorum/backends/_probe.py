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

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from functools import cache
from pathlib import Path

# Note on scope: only the *subprocess* probe is cached here. The sibling
# `_launcher_available` checks on each backend use `shutil.which`, which measured
# 0.00s, so caching them would add indirection for no gain.


# Where a conda-family launcher lives when it is not on PATH. Ordered most-specific first.
#
# `micromamba` installs itself as a shell FUNCTION — `micromamba shell hook` defines one so that
# `micromamba activate` can modify the calling shell. A function is invisible to
# `shutil.which()`, so on a machine whose PATH never reaches the binary's directory the probe
# concludes the launcher is absent while `command -v micromamba` in the same terminal says it is
# right there. That is exactly what happened: a run died with "The scclr backend is unavailable
# (missing: micromamba)" on a machine with micromamba installed and a working scclr env.
_LAUNCHER_SEARCH_DIRS: tuple[str, ...] = (
    "$MAMBA_ROOT_PREFIX/bin",
    "$MICROMAMBA_ROOT_PREFIX/bin",
    "$CONDA_PREFIX/bin",
    "~/micromamba/bin",
    "~/.local/bin",
    "~/miniforge3/bin",
    "~/mambaforge/bin",
    "~/miniconda3/bin",
    "~/anaconda3/bin",
    "/opt/homebrew/bin",
    "/usr/local/bin",
)


@cache
def resolve_launcher(name: str) -> str | None:
    """Locate an environment launcher, tolerating one that is not on PATH.

    ``shutil.which`` first, because an explicit PATH entry is the caller's intent. Failing that,
    the conventional install directories — so a launcher that exists is found rather than
    reported missing. Six backends each called ``shutil.which(self.launcher)`` directly; they now
    share this, so a launcher findable for one is findable for all.

    Args:
        name: Launcher name (``micromamba``/``mamba``/``conda``), or an absolute path, which is
            returned unchanged when it is executable.

    Returns:
        Absolute path to the launcher, or None when it genuinely is not installed.
    """
    candidate = Path(name).expanduser()
    if candidate.is_absolute():
        return str(candidate) if os.access(candidate, os.X_OK) else None

    found = shutil.which(name)
    if found:
        return found

    for raw in _LAUNCHER_SEARCH_DIRS:
        expanded = os.path.expandvars(raw)
        # An unset variable leaves the literal `$NAME` behind; skip rather than stat it.
        if "$" in expanded:
            continue
        binary = Path(expanded).expanduser() / name
        if binary.is_file() and os.access(binary, os.X_OK):
            return str(binary)
    return None


@cache
def existing_env_names(launcher: str) -> frozenset[str]:
    """Names of the environments the launcher can see.

    Cached for the process: a launcher does not gain environments mid-run, and the listing
    costs a subprocess.

    Args:
        launcher: Launcher executable, ideally already resolved by :func:`resolve_launcher`.

    Returns:
        The environment names, or an empty set when the listing fails.
    """
    try:
        result = subprocess.run(
            [launcher, "env", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return frozenset()
    if result.returncode != 0:
        return frozenset()
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        return frozenset()
    return frozenset(str(path).rstrip("/").rsplit("/", 1)[-1] for path in payload.get("envs", []))


def resolve_env(launcher: str, candidates: Sequence[str]) -> str | None:
    """Return the first candidate environment that actually exists.

    The same job as :func:`resolve_launcher`, one level up: a backend names the environment it
    wants, and this decides which of the acceptable names is present here.

    It exists because environment names are a fact about a MACHINE, not about an analysis. The
    container built by ``docker/Dockerfile`` creates ``pyscenic_env`` and ``hdwgcna_env`` -- the
    names the code defaults to -- while this workstation has ``scenic_env`` and ``lekc_hubs``.
    Pointing the config at the local names made both stages work here and would have broken
    them in the container, so the analysis config had become machine-bound. Listing candidates
    instead keeps one config correct in both places.

    Args:
        launcher: Launcher executable.
        candidates: Acceptable environment names, most-preferred first.

    Returns:
        The first name present, the first candidate when the listing is unavailable (so a
        probe failure degrades to the old single-name behaviour rather than reporting the
        backend missing), or None when no candidate was given.
    """
    names = [str(name) for name in candidates if str(name)]
    if not names:
        return None
    available = existing_env_names(launcher)
    if not available:
        return names[0]
    for name in names:
        if name in available:
            return name
    return None


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
