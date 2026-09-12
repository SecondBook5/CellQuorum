"""Structural release checks for the workstation dependency locks."""

import re
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ENVS = Path(__file__).resolve().parents[1] / "envs"


@pytest.mark.parametrize("profile", ["core", "gpu"])
def test_workstation_lock_integrity(profile):
    recipe_path = ENVS / f"cellquorum-{profile}.yml"
    recipe = yaml.safe_load(recipe_path.read_text())
    lock = yaml.safe_load(recipe_path.with_suffix(".conda-lock.yml").read_text())
    assert lock["metadata"]["platforms"] == ["linux-64"]
    assert lock["metadata"]["sources"] == [recipe_path.name]
    packages = lock["package"]
    identities = [(p["manager"], p["name"], p["platform"]) for p in packages]
    assert len(identities) == len(set(identities)), "Duplicate package records"
    managers = {}
    for package in packages:
        name = canonicalize_name(package["name"])
        managers.setdefault(name, set()).add(package["manager"])
        assert package["platform"] == "linux-64"
        url = package["url"]
        if url.startswith("git+https://"):
            revision = url.rsplit("@", 1)[-1]
            assert re.fullmatch(r"[0-9a-f]{40}", revision), package["name"]
            assert package["hash"]["sha256"] == revision
        else:
            assert url.startswith("https://"), package["name"]
            assert re.fullmatch(r"[0-9a-f]{64}", package["hash"]["sha256"]), name
    assert all(len(owners) == 1 for owners in managers.values()), managers

    by_name = {canonicalize_name(p["name"]): p for p in packages}
    for dependency in recipe["dependencies"]:
        if not isinstance(dependency, dict):
            continue
        for entry in dependency.get("pip", []):
            requirement = Requirement(entry)
            package = by_name[canonicalize_name(requirement.name)]
            assert requirement.specifier.contains(package["version"]), entry
            if requirement.url:
                expected, _, fragment = requirement.url.partition("#")
                assert unquote(package["url"]) == unquote(expected), entry
                if fragment:
                    algorithm, digest = fragment.split("=", 1)
                    assert package["hash"][algorithm] == digest, entry

    torch = by_name["torch"]
    assert torch["version"].endswith("+cpu" if profile == "core" else "+cu129")
    assert by_name["python"]["version"].startswith("3.12.")
    if profile == "core":
        assert not any(name.startswith("nvidia-") for name in by_name)
    else:
        assert {"cupy", "cudf", "cuml", "cugraph", "cuvs"} <= by_name.keys()
