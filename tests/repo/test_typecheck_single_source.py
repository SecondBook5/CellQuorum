"""The list of type-checked packages lives in one place, pyproject.toml.

CI and `make typecheck` once each spelled their own list, and the Makefile's fell behind by
seven packages without anyone noticing. A setting written in two places is a bug waiting to
happen, so this pins that neither of them names a path.
"""

from __future__ import annotations

import re
import tomllib

from repo_paths import REPO_ROOT


def test_the_gated_packages_are_listed_once_in_pyproject() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    files = config["tool"]["mypy"]["files"]
    assert files, "[tool.mypy] files is empty, so mypy would check nothing"
    for path in files:
        assert (REPO_ROOT / path).is_dir(), f"[tool.mypy] files names a missing folder: {path}"


def test_ci_and_make_run_a_bare_mypy() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    commands = re.findall(r"^\s*run:\s*>?\s*\n?\s*(mypy\b[^\n]*(?:\n\s+src/\S+)*)", ci, re.M)
    assert commands, "no mypy step found in ci.yml"
    assert all(command.strip() == "mypy" for command in commands), commands

    make = (REPO_ROOT / "Makefile").read_text()
    recipe = re.search(r"^typecheck:\n((?:\t.*\n)+)", make, re.M)
    assert recipe is not None
    assert recipe.group(1).strip() == "mypy"
