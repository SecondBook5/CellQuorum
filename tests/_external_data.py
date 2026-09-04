"""External-dataset opt-in and cached tool probes for the test suite.

Import these directly; ``tests/`` is on ``sys.path`` under pytest:

```python
from _external_data import require_cellranger_library, require_r_package
```

## External datasets

A few tests are genuine integration tests against real, multi-gigabyte data — real
Cell Ranger matrices, a real reference-mapped ``.h5ad``. That data cannot live in the
repository, so those tests have to locate it somehow.

They used to locate it by **hardcoding the maintainer's absolute paths** — an external
drive mount for the Cell Ranger matrices, a Windows user directory for the cohort
``.h5ad``. That is a problem well beyond untidiness:

* For every other contributor, and in CI, those paths never exist, so the tests are
  permanently skipped — dead weight that looks like coverage.
* The skip reason ("data unavailable") is indistinguishable from a genuine
  misconfiguration, so nobody can tell whether the test *could* run.
* A test suite that only fully runs on one laptop cannot gate a release.

Instead, each external dataset is now named by an **environment variable**. Unset means
the test skips with a message naming the variable and what to point it at, so the skip
is actionable rather than mysterious. Nothing is assumed about any machine's layout.

To run these locally, point the variables at your own copies:

```bash
export CELLQUORUM_TEST_CELLRANGER_ROOT=/path/to/cellranger_output
export CELLQUORUM_TEST_KC_H5AD=/path/to/le_kc_keratinocyte_refmapped.h5ad
```

These tests also carry the ``integration`` marker (plus ``r`` and ``slow`` where they
shell out to R or take minutes), so they can be deselected wholesale even on a machine
that *does* have the data:

```bash
pytest -m "not integration"
```

Markers and ``skipif`` answer different questions and the suite needs both: ``skipif``
handles "the data is absent", a marker handles "I do not want to run this now". Only the
marker can deselect a test on a machine where the data *is* present.

## Cached tool probes

Asking "is the SoupX R package installed?" costs an ``Rscript`` subprocess. Eighteen
test modules each roll their own version of that probe, and several evaluate it at
**module scope** — so the subprocess runs during collection, for every session, even
when the test is deselected. :func:`r_package_available` memoizes the answer for the
process, which is safe because an R library does not gain packages mid-session.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

# Environment variable naming the Cell Ranger output root: the directory holding
# per-run subdirectories such as `Set1_norm_LE/LE1_v8/outs`.
ENV_CELLRANGER_ROOT = "CELLQUORUM_TEST_CELLRANGER_ROOT"

# Environment variable naming a reference-mapped keratinocyte .h5ad used by the
# data-contract regression test.
ENV_KC_H5AD = "CELLQUORUM_TEST_KC_H5AD"

# Environment variables that the shipped study configs interpolate with
# ``${oc.env:...}`` so they name no machine's filesystem. A test that only checks a
# config VALIDATES (rather than running it) has no business depending on the developer's
# environment, so it stubs these instead — see :func:`stub_config_env`.
CONFIG_ENV_VARS = (
    "CELLQUORUM_CELLRANGER_ROOT",
    "CELLQUORUM_KC_H5AD",
    "CELLQUORUM_AD_ATLAS_H5AD",
    "CELLQUORUM_KC_ATLAS_H5AD",
    "CELLQUORUM_SMOKE_H5AD",
)


def stub_config_env(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """
    Point every config-interpolated environment variable at a throwaway location.

    Config *validation* only needs the interpolations to resolve to something
    path-shaped; nothing is read. Stubbing them keeps such tests passing identically on
    a contributor's laptop and in CI, where none of the real datasets exist.

    Args:
        monkeypatch: pytest fixture used to set the variables for one test.
        root: Directory to hang the stub paths off, typically ``tmp_path``.
    """

    # Give each variable a distinct path under the throwaway root, so a config that
    # accidentally reuses one variable for two purposes is still visible.
    #
    # The suffix matters: several config fields validate their extension (input.h5ad
    # rejects anything not ending in `.h5ad`), so a generic ".stub" value would fail
    # validation for reasons that have nothing to do with what a test is checking.
    for name in CONFIG_ENV_VARS:
        suffix = ".h5ad" if name.endswith("_H5AD") else ""
        monkeypatch.setenv(name, f"{root / name.lower()}{suffix}")


def external_path(env_var: str) -> Path | None:
    """
    Return the filesystem path an environment variable names, if it is set.

    Args:
        env_var: Name of the environment variable to read.

    Returns:
        The expanded path, or None when the variable is unset or blank.
    """

    # Treat blank and whitespace-only values as unset, so `export VAR=` disables a
    # dataset rather than resolving to the current directory.
    raw = os.environ.get(env_var, "").strip()

    # Expand `~` so a home-relative path works as a user would expect.
    return Path(raw).expanduser() if raw else None


def require_external_dir(env_var: str, *, what: str) -> Path:
    """
    Return an existing directory named by an environment variable, or skip.

    Args:
        env_var: Environment variable expected to hold the directory path.
        what: Human-readable description of the dataset, used in the skip message.

    Returns:
        The resolved directory path.
    """

    # Resolve the configured location, if any.
    path = external_path(env_var)

    # Skip with an actionable message naming both the variable and the dataset.
    if path is None:
        pytest.skip(f"{env_var} is not set; point it at {what} to run this test")

    # Distinguish "configured but wrong" from "not configured" — a typo in the
    # variable should not read the same as opting out.
    if not path.is_dir():
        pytest.skip(f"{env_var}={path} is not a directory (expected {what})")

    # Hand back the validated directory.
    return path


def require_external_file(env_var: str, *, what: str) -> Path:
    """
    Return an existing file named by an environment variable, or skip.

    Args:
        env_var: Environment variable expected to hold the file path.
        what: Human-readable description of the dataset, used in the skip message.

    Returns:
        The resolved file path.
    """

    # Resolve the configured location, if any.
    path = external_path(env_var)

    # Skip with an actionable message naming both the variable and the dataset.
    if path is None:
        pytest.skip(f"{env_var} is not set; point it at {what} to run this test")

    # Distinguish "configured but wrong" from "not configured".
    if not path.is_file():
        pytest.skip(f"{env_var}={path} is not a file (expected {what})")

    # Hand back the validated file.
    return path


def require_cellranger_library(*relative_parts: str, needs: tuple[str, ...] = ()) -> Path:
    """
    Return a Cell Ranger ``outs`` directory under the configured root, or skip.

    Args:
        *relative_parts: Path components under the Cell Ranger root, e.g.
            ``"Set1_norm_LE", "LE1_v8", "outs"``.
        needs: Matrix file names that must exist inside the directory, e.g.
            ``("raw_feature_bc_matrix.h5",)``. Checked explicitly because a
            half-populated library otherwise fails deep inside R with an opaque error.

    Returns:
        The resolved library directory.
    """

    # Locate the configured Cell Ranger root, skipping when it is absent.
    root = require_external_dir(
        ENV_CELLRANGER_ROOT,
        what="a Cell Ranger output root (the directory holding per-run outs/ dirs)",
    )

    # Descend to the requested library.
    library = root.joinpath(*relative_parts)

    # Skip when this particular library is missing: a root can legitimately hold a
    # different cohort than the one a given test names.
    if not library.is_dir():
        pytest.skip(f"Cell Ranger library not found under {ENV_CELLRANGER_ROOT}: {library}")

    # Verify the specific matrices the caller needs, so a missing file skips here with
    # a clear reason instead of failing later inside the R backend.
    missing = [name for name in needs if not (library / name).is_file()]
    if missing:
        pytest.skip(f"Cell Ranger library {library} is missing {', '.join(missing)}")

    # Hand back the validated library directory.
    return library


@cache
def rscript_available() -> bool:
    """
    Return whether ``Rscript`` is on ``PATH``, caching the answer.

    Returns:
        True when Rscript resolves on PATH.
    """

    # Cached because it is asked once per R-backed test module.
    return shutil.which("Rscript") is not None


@cache
def r_package_available(package: str) -> bool:
    """
    Return whether an R package is installed, caching the answer for the process.

    Each call costs an ``Rscript`` subprocess, and eighteen test modules ask
    variations of this question — several at module scope, where the cost lands on
    collection for every session regardless of what was selected. An R library does
    not gain packages mid-session, so memoizing is safe.

    Args:
        package: R package name to probe, e.g. ``"SoupX"``.

    Returns:
        True when Rscript is present and the package can be loaded.
    """

    # No Rscript means no R package, and saves spawning a doomed subprocess.
    if not rscript_available():
        return False

    # `quit(status=!requireNamespace(...))` exits 0 when the package is loadable.
    # --vanilla keeps a user's ~/.Rprofile from changing the answer.
    try:
        result = subprocess.run(
            [
                "Rscript",
                "--vanilla",
                "-e",
                f'quit(status=!requireNamespace("{package}", quietly=TRUE))',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        # A broken or hanging R installation reads as "package unavailable" so the
        # test skips rather than erroring during collection.
        return False

    # Exit status 0 means requireNamespace succeeded.
    return result.returncode == 0


def require_r_package(package: str) -> None:
    """
    Skip the calling test unless an R package is installed.

    Args:
        package: R package name required by the test.
    """

    # Skip with a message naming the specific package, not just "R unavailable".
    if not r_package_available(package):
        pytest.skip(f"Rscript with the {package} R package is required for this test")
