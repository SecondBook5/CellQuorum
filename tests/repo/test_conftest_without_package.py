"""The lightweight CI jobs run tests without installing cellquorum; conftest must not need it."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from repo_paths import REPO_ROOT

_BLOCKER = textwrap.dedent(
    """
    import importlib.abc
    import sys


    class _Hide(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "cellquorum" or name.startswith("cellquorum."):
                raise ModuleNotFoundError(f"No module named {name!r}", name="cellquorum")
            return None


    sys.meta_path.insert(0, _Hide())
    for key in [k for k in sys.modules if k == "cellquorum" or k.startswith("cellquorum.")]:
        del sys.modules[key]
    """
)


def test_lock_integrity_tests_run_when_cellquorum_is_not_installed(tmp_path: Path) -> None:
    """The environment-lock job installs only pytest, PyYAML and packaging, so the autouse
    matplotlib-style fixture in tests/conftest.py has to step aside when the package is absent."""
    (tmp_path / "hide_cellquorum.py").write_text(_BLOCKER)
    env_path = f"{tmp_path}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/repo/test_environment_locks.py",
            "-p",
            "hide_cellquorum",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-500:]
