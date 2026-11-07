"""The locking wrapper leaves each package with one owner, conda or pip, never both."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from repo_paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "lock_environments", REPO_ROOT / "scripts" / "lock_environments.py"
)
assert _spec is not None and _spec.loader is not None
lock_environments = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lock_environments)

LOCK = """# header comment kept
metadata:
  platforms:
  - linux-64
package:
- name: python-igraph
  version: 0.11.9
  manager: conda
  platform: linux-64
  url: https://example.test/python-igraph.conda
- name: igraph
  version: 1.0.0
  manager: pip
  platform: linux-64
  url: https://example.test/igraph-1.0.0.whl
- name: palantir
  version: 1.4.4
  manager: pip
  platform: linux-64
  url: https://example.test/palantir.whl
"""


def test_a_pip_package_that_conda_already_provides_is_dropped(tmp_path: Path) -> None:
    lock = tmp_path / "x.conda-lock.yml"
    lock.write_text(LOCK)
    assert lock_environments.drop_pip_packages_provided_by_conda(lock) == ["igraph"]
    text = lock.read_text()
    assert "# header comment kept" in text
    assert "name: python-igraph" in text and "name: palantir" in text
    assert "igraph-1.0.0" not in text


def test_the_pip_package_is_kept_when_conda_does_not_provide_it(tmp_path: Path) -> None:
    lock = tmp_path / "x.conda-lock.yml"
    lock.write_text(LOCK.replace("python-igraph", "something-else"))
    assert lock_environments.drop_pip_packages_provided_by_conda(lock) == []
    assert "igraph-1.0.0" in lock.read_text()
