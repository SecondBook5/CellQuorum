"""Figures use Arial or a font with its metrics, and fall back to DejaVu only as a last resort."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest
from matplotlib import font_manager

matplotlib.use("Agg")

from cellquorum.visualization import figstyle  # noqa: E402

STYLES = ("compact", "notebook", "publication")
ARIAL_LIKE = ("arial", "liberationsans", "arimo")


def _arial_like_installed() -> bool:
    roots = [Path("/usr/share/fonts"), Path.home() / ".fonts", Path.home() / ".local/share/fonts"]
    for root in roots:
        if root.exists() and any(
            any(tag in p.name.lower().replace("-", "").replace("_", "") for tag in ARIAL_LIKE)
            for p in root.rglob("*.ttf")
        ):
            return True
    return False


@pytest.mark.parametrize("style", STYLES)
def test_arial_like_fonts_are_tried_before_dejavu(style: str) -> None:
    figstyle.apply_theme(style)
    stack = matplotlib.rcParams["font.sans-serif"]
    assert stack[0] == "Arial"
    assert stack.index("Liberation Sans") < stack.index("DejaVu Sans")


@pytest.mark.skipif(not _arial_like_installed(), reason="no Arial-compatible font on this machine")
@pytest.mark.parametrize("style", STYLES)
def test_text_resolves_to_an_arial_like_font_when_one_is_installed(style: str) -> None:
    figstyle.apply_theme(style)
    path = font_manager.findfont(font_manager.FontProperties(family=["sans-serif"]))
    name = Path(path).name.lower().replace("-", "").replace("_", "")
    assert any(tag in name for tag in ARIAL_LIKE), path
