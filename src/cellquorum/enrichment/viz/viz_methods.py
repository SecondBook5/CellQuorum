"""Backward-compatibility shim — pre-#187 import path.

Canonical location: :mod:`cellquorum.comparative.enrichment.viz.viz_methods`.

CellQuorum's four comparative analyses (differential expression, differential
abundance, enrichment, multicellular programs) were consolidated under the
``cellquorum.comparative`` package in #187. This module re-exports the public
API from its new home so pre-consolidation imports keep working unchanged.
New code should import from the canonical location above.
"""

from __future__ import annotations

from cellquorum.comparative.enrichment.viz.viz_methods import (
    ActivityVizMethod,
    GseaVizMethod,
    GsvaVizMethod,
    OraVizMethod,
)

__all__ = ["ActivityVizMethod", "GseaVizMethod", "GsvaVizMethod", "OraVizMethod"]
