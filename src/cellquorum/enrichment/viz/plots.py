"""Backward-compatibility shim — pre-#187 import path.

Canonical location: :mod:`cellquorum.comparative.enrichment.viz.plots`.

CellQuorum's four comparative analyses (differential expression, differential
abundance, enrichment, multicellular programs) were consolidated under the
``cellquorum.comparative`` package in #187. This module re-exports the public
API from its new home so pre-consolidation imports keep working unchanged.
New code should import from the canonical location above.
"""

from __future__ import annotations

from cellquorum.comparative.enrichment.viz.plots import (
    activity_dotplot,
    annotated_clustermap,
    cross_group_dotplot,
    diverging_bar,
    ora_barplot,
    ora_dotplot,
    pvalue_to_stars,
    running_es_curve,
    select_top_bottom,
    signed_norm,
)

__all__ = [
    "signed_norm",
    "pvalue_to_stars",
    "select_top_bottom",
    "diverging_bar",
    "activity_dotplot",
    "running_es_curve",
    "ora_barplot",
    "ora_dotplot",
    "annotated_clustermap",
    "cross_group_dotplot",
]
