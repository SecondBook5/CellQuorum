"""Guard against machine-specific absolute paths reappearing in the repository.

Shipped configs and tests used to hardcode the maintainer's filesystem: an external
drive for the Cell Ranger matrices, a Windows user directory for a cohort ``.h5ad``, and
an absolute path into one particular checkout for run outputs. ``configs/config.yaml`` is
the file the CLI loads when no ``--config`` is given, so a new user's very first
``cellquorum run`` tried to write to a drive that did not exist on their machine.

Paths like these are easy to reintroduce — you set one while debugging your own data and
it rides along in the commit. Nothing about that fails a test, so this is the test.

The rule: a path that names a *specific machine's* layout does not belong in tracked
source or config. Point at the data through an environment variable instead
(``${oc.env:VAR}`` in YAML, since the loader resolves OmegaConf interpolations; the
helpers in ``tests/_external_data.py`` for tests), or use a repo-relative path.

Only **git-tracked** files are scanned. Untracked and gitignored files are a developer's
own business — study-specific scratch configs live there by design.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from _external_data import stub_config_env

# Repository root, derived from this file's location rather than the CWD so the test
# behaves the same however pytest is invoked.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories whose tracked contents must stay machine-independent. docs/ is excluded:
# prose legitimately shows example paths, and a doc cannot break a run.
SCANNED_DIRS = ("configs", "src", "tests", "scripts")

# File types worth scanning. Anything a run reads or imports.
SCANNED_SUFFIXES = (".py", ".yaml", ".yml", ".R", ".csv")

# Absolute path roots that name one machine's layout. Each is a path *prefix* appearing
# inside the file, so `/mnt/e/data` is caught while "home" in prose is not. machine-path-ok
#
# System roots (/usr, /opt, /etc, /var) and /tmp are deliberately absent: those are
# portable across POSIX machines, and /tmp is the conventional place for scratch
# fixtures.
MACHINE_SPECIFIC_PREFIXES = (
    "/mnt/",  # machine-path-ok - this is the pattern list, not a path
    "/home/",  # machine-path-ok
    "/Users/",  # machine-path-ok
    "/media/",  # machine-path-ok
    "/Volumes/",  # machine-path-ok
)

# Windows drive letters, e.g. a `C:` or `D:` prefix. Kept separate because it needs a
# pattern rather than a literal prefix.
WINDOWS_DRIVE = re.compile(r"\b[A-Za-z]:[\\/]")

# Lines exempt from the check, by marker. A line ending in this comment opts out, which
# keeps the rule enforceable without becoming a blocker for a genuine exception.
ALLOW_MARKER = "machine-path-ok"


def _tracked_files() -> list[Path]:
    """
    List git-tracked files in the scanned directories.

    Returns:
        Absolute paths of tracked files with a scanned suffix.
    """

    # Ask git rather than walking the tree, so gitignored study configs and untracked
    # scratch files are excluded — those are a developer's own business.
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", *SCANNED_DIRS],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        pytest.skip(f"git is unavailable, cannot determine tracked files: {error}")

    # A non-zero status means this is not a git checkout (e.g. an extracted sdist).
    if completed.returncode != 0:
        pytest.skip("not a git checkout, cannot determine tracked files")

    # Keep only the file types worth scanning.
    return [
        REPO_ROOT / line
        for line in completed.stdout.splitlines()
        if line.endswith(SCANNED_SUFFIXES)
    ]


def _offending_lines(path: Path) -> list[tuple[int, str]]:
    """
    Find lines in one file that embed a machine-specific absolute path.

    Args:
        path: File to scan.

    Returns:
        List of (1-based line number, stripped line) for each offending line.
    """

    # A file removed between `git ls-files` and now should not fail the suite.
    if not path.is_file():
        return []

    # Binary or oddly encoded files are not our concern.
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    offenders: list[tuple[int, str]] = []

    for number, line in enumerate(text.splitlines(), start=1):
        # Honor an explicit per-line opt-out.
        if ALLOW_MARKER in line:
            continue

        # Flag any machine-specific POSIX root or Windows drive reference.
        if any(prefix in line for prefix in MACHINE_SPECIFIC_PREFIXES) or WINDOWS_DRIVE.search(
            line
        ):
            offenders.append((number, line.strip()))

    return offenders


def test_no_tracked_file_embeds_a_machine_specific_path() -> None:
    """No tracked config, source, or test file may hardcode one machine's paths."""

    # Collect every offending line across the scanned, tracked files.
    findings: list[str] = []
    for path in _tracked_files():
        findings.extend(
            f"  {path.relative_to(REPO_ROOT)}:{number}: {line}"
            for number, line in _offending_lines(path)
        )

    # Report every offender at once: fixing these one failure at a time is miserable.
    assert not findings, (
        "Machine-specific absolute paths found in tracked files.\n\n"
        + "\n".join(findings)
        + "\n\nThese make the repository depend on one computer's filesystem. Use an "
        "environment variable instead — `${oc.env:VAR}` in YAML (the config loader "
        "resolves OmegaConf interpolations), or the helpers in "
        "tests/_external_data.py for tests — or a repo-relative path. If a literal "
        f"path is genuinely required, append a `{ALLOW_MARKER}` comment to that line."
    )


def test_default_config_is_portable() -> None:
    """`configs/config.yaml` must load and resolve an output dir on any machine.

    This is the config the CLI reads when no ``--config`` is passed, so it is the first
    thing a new user exercises. It must not require editing, and must not depend on any
    environment variable being set.
    """

    from cellquorum.config.loader import load_config
    from cellquorum.core.pipeline import resolve_output_dir

    # Load the shipped default exactly as the CLI would.
    config = load_config(REPO_ROOT / "configs" / "config.yaml")

    # An output directory must be derivable without any env var or override, or the
    # first `cellquorum run` fails with "could not resolve an output directory".
    assert resolve_output_dir(config) is not None

    # The roots that remain configured must be relative, so they land under the
    # directory the user runs from rather than a path only one machine has.
    for field in ("data_root", "run_root", "scratch_root", "output_dir"):
        value = getattr(config.paths, field)
        assert value is None or not value.is_absolute(), (
            f"configs/config.yaml paths.{field} is the absolute path {value}; the "
            f"shipped default must be null or repo-relative"
        )


def _tracked_config_files() -> list[Path]:
    """
    List the tracked YAML configs at the top level of ``configs/``.

    Returns:
        Absolute paths of shipped, user-facing config files.
    """

    # Reuse the tracked-file listing, then keep only top-level configs. Nested
    # directories (profiles/, backends/, qc/, …) are Hydra config *groups* — fragments
    # meant to be composed, not loaded standalone, so they will not validate alone.
    return sorted(
        path
        for path in _tracked_files()
        if path.suffix in {".yaml", ".yml"} and path.parent == REPO_ROOT / "configs"
    )


def test_every_shipped_config_validates() -> None:
    """Every tracked top-level config must load and validate against the schema.

    Configs drift from the schema silently: a field gets renamed in ``config/models.py``
    and the example configs keep the old name until someone tries to run one. This walks
    every shipped config so that drift fails here instead of in a user's first run.

    Environment interpolations are stubbed, because the point is schema validity, not
    whether this machine holds the datasets.
    """

    # Collect the configs to check, skipping cleanly outside a git checkout.
    configs = _tracked_config_files()

    # A pass that silently checked nothing would be worse than a failure.
    assert configs, "no tracked configs found under configs/"

    # The monkeypatch fixture is function-scoped, but this loop wants env vars set and
    # torn down around each individual load. pytest.MonkeyPatch.context() is the public
    # API for that.
    from cellquorum.config.loader import ConfigLoadError, load_config

    failures: list[str] = []
    for path in configs:
        with pytest.MonkeyPatch.context() as patch:
            # Give every ${oc.env:...} reference something path-shaped to resolve to.
            stub_config_env(patch, REPO_ROOT / "build" / "config_env_stub")

            # Record rather than raise, so one broken config does not hide the others.
            try:
                load_config(path)
            except (ConfigLoadError, ValueError, TypeError) as error:
                failures.append(f"  {path.relative_to(REPO_ROOT)}: {type(error).__name__}: {error}")

    # Report every broken config at once.
    assert not failures, "Shipped configs failed to validate:\n" + "\n".join(failures)
