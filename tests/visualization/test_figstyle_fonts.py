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


@pytest.mark.parametrize("style", STYLES)
def test_times_like_fonts_are_tried_before_dejavu_serif(style: str) -> None:
    """The few figures that ask for a serif face get Times metrics, not DejaVu Serif."""
    figstyle.apply_theme(style)
    stack = matplotlib.rcParams["font.serif"]
    assert stack[0] == "Times New Roman"
    assert stack.index("Liberation Serif") < stack.index("DejaVu Serif")


@pytest.mark.skipif(not _arial_like_installed(), reason="no Arial-compatible font on this machine")
@pytest.mark.parametrize("style", STYLES)
def test_text_resolves_to_an_arial_like_font_when_one_is_installed(style: str) -> None:
    figstyle.apply_theme(style)
    path = font_manager.findfont(font_manager.FontProperties(family=["sans-serif"]))
    name = Path(path).name.lower().replace("-", "").replace("_", "")
    assert any(tag in name for tag in ARIAL_LIKE), path


def _arial_installed() -> bool:
    roots = [Path("/usr/share/fonts"), Path.home() / ".fonts", Path.home() / ".local/share/fonts"]
    return any(p.name.lower() == "arial.ttf" for r in roots if r.exists() for p in r.rglob("*.ttf"))


@pytest.mark.skipif(not _arial_installed(), reason="Arial itself is not installed on this machine")
@pytest.mark.parametrize("style", STYLES)
def test_arial_itself_is_used_when_installed_even_with_a_stale_font_cache(style: str) -> None:
    from matplotlib import font_manager

    stale = [f for f in font_manager.fontManager.ttflist if f.name != "Arial"]
    original = font_manager.fontManager.ttflist
    font_manager.fontManager.ttflist = stale
    figstyle._register_arial_compatible_fonts.cache_clear()
    try:
        figstyle.apply_theme(style)
        path = font_manager.findfont(font_manager.FontProperties(family=["sans-serif"]))
        assert Path(path).name.lower() == "arial.ttf", path
    finally:
        font_manager.fontManager.ttflist = original
