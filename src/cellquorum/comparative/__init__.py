"""Comparative analyses: given groups, compute what differs between them.

This package groups CellQuorum's four "compare groups" analysis stages —
differential expression, differential abundance, functional enrichment, and
multicellular (cross-cell-type) programs — as submodules:

- :mod:`cellquorum.comparative.differential_expression`
- :mod:`cellquorum.comparative.differential_abundance`
- :mod:`cellquorum.comparative.enrichment`
- :mod:`cellquorum.comparative.multicellular_programs`

Each submodule keeps the uniform stage layout (``stage.py``, ``config.py``, one
method module per method). This ``__init__`` deliberately performs no eager
submodule imports: importing :mod:`cellquorum.comparative` alone pulls in no heavy
optional dependency, preserving the skip-not-crash invariant. Import a submodule to
use it, exactly as before the #187 consolidation.
"""

from __future__ import annotations
