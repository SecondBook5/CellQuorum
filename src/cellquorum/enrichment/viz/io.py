"""Backward-compatibility shim — pre-#187 import path.

Canonical location: :mod:`cellquorum.comparative.enrichment.viz.io`.

CellQuorum's four comparative analyses (differential expression, differential
abundance, enrichment, multicellular programs) were consolidated under the
``cellquorum.comparative`` package in #187. This module re-exports the public
API from its new home so pre-consolidation imports keep working unchanged.
New code should import from the canonical location above.
"""

from __future__ import annotations

from cellquorum.comparative.enrichment.viz.io import (
    apply_theme,
    collections_from_glob,
    figure_artifacts,
    save_figure,
)

__all__ = ["collections_from_glob", "apply_theme", "save_figure", "figure_artifacts"]
