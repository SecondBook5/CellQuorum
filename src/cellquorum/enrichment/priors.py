"""Backward-compatibility shim — pre-#187 import path.

Canonical location: :mod:`cellquorum.comparative.enrichment.priors`.

CellQuorum's four comparative analyses (differential expression, differential
abundance, enrichment, multicellular programs) were consolidated under the
``cellquorum.comparative`` package in #187. This module re-exports the public
API from its new home so pre-consolidation imports keep working unchanged.
New code should import from the canonical location above.
"""

from __future__ import annotations

from cellquorum.comparative.enrichment.priors import PriorFetchError, get_net

__all__ = ["PriorFetchError", "get_net"]
