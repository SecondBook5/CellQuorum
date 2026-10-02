"""The README is the front door, so what it states must stay true.

Each claim here was once wrong (the stage count was 32 while the code had 40; a Python version
was advertised that CI never tested), so a test now compares the README against the code.
"""

from __future__ import annotations

import re

import pytest
from repo_paths import REPO_ROOT

ROOT = REPO_ROOT
README = (ROOT / "README.md").read_text()


def test_the_stage_count_matches_the_catalog() -> None:
    from cellquorum.core.stages import all_stage_specs

    implemented = sum(1 for spec in all_stage_specs() if spec.is_implemented)
    claimed = {int(n) for n in re.findall(r"\*\*(\d+) stages", README)}
    assert claimed == {implemented}, f"README says {claimed}, the catalog implements {implemented}"


def test_the_method_count_is_a_floor_the_registry_meets() -> None:
    from cellquorum.core.stages import all_stage_specs  # registers every method
    from cellquorum.methods.registry import METHOD_REGISTRY

    all_stage_specs()
    registered = sum(len(methods) for methods in METHOD_REGISTRY._methods.values())
    claimed = {int(n) for n in re.findall(r"\*\*(\d+)\+ selectable methods", README)}
    assert len(claimed) == 1 and registered >= claimed.pop(), registered


def test_the_python_versions_advertised_are_the_ones_ci_tests() -> None:
    badge = re.search(r"badge/python-([\d.%A-Z7C]+)-blue", README)
    assert badge, "the Python badge is missing"
    advertised = set(re.findall(r"\d+\.\d+", badge.group(1).replace("%20%7C%20", " ")))
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    matrix = re.search(r"python-version:\s*\[([^\]]+)\]", ci)
    assert matrix, "the CI test matrix could not be found"
    tested = set(re.findall(r"\d+\.\d+", matrix.group(1)))
    assert advertised == tested, f"README advertises {advertised} but CI tests {tested}"


def test_every_local_link_and_image_in_the_readme_exists() -> None:
    targets = re.findall(r"\]\(([^)\s]+)\)", README) + re.findall(r'src="([^"]+)"', README)
    missing = []
    for target in targets:
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if not (ROOT / target.split("#")[0]).exists():
            missing.append(target)
    assert not missing, f"README points at files that do not exist: {missing}"


def test_the_quickstart_uses_commands_that_exist() -> None:
    from typer.main import get_command

    from cellquorum.cli.app import app

    section = README.split("## See it work in two minutes")[1].split("\n## ")[0]
    block = re.search(r"```bash\n(.*?)```", section, re.S)
    assert block, "the quickstart has no bash block"
    commands = set(get_command(app).commands)
    used = re.findall(r"^cellquorum\s+(\w+)", block.group(1), re.M)
    assert {"demo", "plan", "run"} <= set(used)
    assert set(used) <= commands, f"README uses {set(used) - commands}, which the CLI lacks"


@pytest.mark.parametrize("name", ["CHANGELOG.md", "CITATION.cff", "LICENSE", "CONTRIBUTING.md"])
def test_the_project_files_a_visitor_expects_are_present(name: str) -> None:
    assert (ROOT / name).is_file()
