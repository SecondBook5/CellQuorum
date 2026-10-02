"""Adapter for figure functions that write files instead of returning a Figure."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path


def via_files(
    writer: Callable[[Path], Iterable[Path | str]], *, pick: str = ""
) -> Callable[[], Path]:
    """Wrap ``writer(out_dir) -> written paths`` as a builder that returns one PNG.

    The gallery baselines the PNG; the PDF a stage writes beside it is ignored. When a
    writer emits several PNGs, ``pick`` selects the one whose file name contains it.
    """

    def build() -> Path:
        out_dir = Path(tempfile.mkdtemp(prefix="gallery_"))
        pngs = [
            Path(p) for p in writer(out_dir) if Path(p).suffix == ".png" and pick in Path(p).name
        ]
        if len(pngs) != 1:
            raise RuntimeError(f"expected exactly one PNG, got {[p.name for p in pngs]}")
        return pngs[0]

    return build
