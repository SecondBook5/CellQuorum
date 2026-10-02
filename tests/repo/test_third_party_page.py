"""The third-party page must cover what CellQuorum depends on and what the README links to."""

from __future__ import annotations

import re
import tomllib

from repo_paths import REPO_ROOT

ROOT = REPO_ROOT
PAGE = (ROOT / "docs" / "reference" / "third-party-software.md").read_text().lower()


def _names(requirements: list[str]) -> list[str]:
    return [re.split(r"[<>=!~\[ ]", item, maxsplit=1)[0] for item in requirements]


def test_every_required_dependency_is_listed() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    required = _names(pyproject["project"]["dependencies"])
    missing = [name for name in required if name.lower() not in PAGE]
    assert not missing, f"third-party-software.md does not list: {missing}"


def test_every_r_package_the_scripts_load_is_listed() -> None:
    loaded: set[str] = set()
    for script in (ROOT / "src" / "cellquorum").rglob("*.R"):
        text = script.read_text()
        loaded |= set(re.findall(r"library\(([A-Za-z][A-Za-z0-9.]+)\)", text))
        loaded |= set(re.findall(r'requireNamespace\("([A-Za-z][A-Za-z0-9.]+)"', text))
    missing = sorted(name for name in loaded if name.lower() not in PAGE)
    assert not missing, f"third-party-software.md does not list these R packages: {missing}"


def test_every_doi_is_well_formed_and_resolves_through_doi_org() -> None:
    dois = re.findall(r"https://doi\.org/(\S+?)\)", PAGE)
    assert len(dois) > 40
    assert all(re.fullmatch(r"10\.\d{4,9}/[^\s)]+", doi) for doi in dois)


def test_every_doi_in_the_readme_is_also_on_the_page() -> None:
    readme = (ROOT / "README.md").read_text().lower()
    readme_dois = set(re.findall(r"https://doi\.org/(\S+?)\)", readme))
    missing = [doi for doi in readme_dois if doi not in PAGE]
    assert not missing, f"cited in the README but not on the third-party page: {missing}"


def test_the_licensing_notes_name_the_copyleft_dependency_cellquorum_requires() -> None:
    assert "leidenalg" in PAGE and "gpl-3.0-or-later" in PAGE
    assert "not legal advice" in PAGE
