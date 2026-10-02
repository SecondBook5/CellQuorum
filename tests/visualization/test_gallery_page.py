"""The docs gallery must list every baseline image, so a new figure cannot be left out."""

from __future__ import annotations

import importlib.util

from repo_paths import REPO_ROOT

ROOT = REPO_ROOT


def _script():
    spec = importlib.util.spec_from_file_location(
        "update_gallery_page", ROOT / "scripts" / "update_gallery_page.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_baseline_image_appears_on_the_gallery_page() -> None:
    page = (ROOT / "docs" / "gallery" / "index.md").read_text()
    missing = [
        path.name
        for path in sorted((ROOT / "docs" / "gallery" / "assets").glob("*.png"))
        if f"assets/{path.name}" not in page
    ]
    assert not missing, f"not on the gallery page: {missing}; run scripts/update_gallery_page.py"


def test_the_generated_index_is_up_to_date() -> None:
    module = _script()
    page = (ROOT / "docs" / "gallery" / "index.md").read_text()
    generated = page[page.index(module.MARKER) :]
    assert generated == module.build_index(), "run scripts/update_gallery_page.py"
