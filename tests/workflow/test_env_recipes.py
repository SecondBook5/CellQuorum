from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVS_DIR = REPO_ROOT / "envs"
BACKENDS_DIR = REPO_ROOT / "src" / "cellquorum" / "backends"

# The five isolated backend envs the image must bake, keyed to their yml file.
BACKEND_ENV_FILES = {
    "celloracle_env": "celloracle_env.yml",
    "pyscenic_env": "pyscenic_env.yml",
    "hdwgcna_env": "hdwgcna_env.yml",
    "scclr": "scclr.yml",
    "sccoda_env": "sccoda_env.yml",
}


def _declared_env_names() -> set[str]:
    """Every env_name string hardcoded in the backend modules."""
    names: set[str] = set()
    pattern = re.compile(r'env_name:\s*str\s*=\s*"([^"]+)"')
    for path in BACKENDS_DIR.glob("*_backend.py"):
        names.update(pattern.findall(path.read_text()))
    return names


def test_every_declared_backend_env_has_a_source_yml() -> None:
    declared = _declared_env_names()
    # Every env name the backends hardcode must be one we ship a recipe for.
    assert declared == set(
        BACKEND_ENV_FILES
    ), f"declared backend envs {declared} != recipe set {set(BACKEND_ENV_FILES)}"


def test_each_recipe_file_exists_and_name_matches() -> None:
    for env_name, filename in BACKEND_ENV_FILES.items():
        path = ENVS_DIR / filename
        assert path.exists(), f"missing env recipe {path}"
        doc = yaml.safe_load(path.read_text())
        assert doc["name"] == env_name, f"{filename} name={doc['name']!r} != {env_name!r}"
        assert doc["dependencies"], f"{filename} has no dependencies"
