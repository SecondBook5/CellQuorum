"""Public analytical building blocks for power-user reuse.

CellQuorum's primary interfaces are the CLI (``cellquorum run``) and the notebook
namespaces under :mod:`cellquorum.api`. But a handful of the engine's internal
helpers are genuinely reusable on their own — analysis scripts across the science
repos already import them directly — so the consolidation design (Move 5) promoted
them to this single, documented, versioned surface instead of leaving downstream
code to reach into deep module paths:

* :func:`de_table_to_ranking` — turn an edgeR/DE table into a preranked contrast
  vector for GSEA (signed ``-log10(p)`` metric).
* :func:`get_net` — fetch a long-format prior-knowledge net (hallmark, Reactome,
  CollecTRI, PROGENy, DoRothEA, a ``.gmt`` …) via decoupler/OmniPath.
* :func:`aggregate_pseudobulk` — sum single cells to donor × condition pseudobulk
  counts for differential expression.

The companion types each function returns or raises — :class:`PseudobulkResult`
and :class:`PriorFetchError` — are exported alongside them so a caller never has to
reach back into the internal modules to type-annotate or handle results.

Study-agnostic *statistical* primitives that sit on top of stage outputs (donor-aware
LMM effect sizes, PERMANOVA-by-group, signature-argmax subtyping, the signed
program-contrast index, leading-edge concordance, program correlations) live in their
own shallow surface, :mod:`cellquorum.stats` — import them from there.

These names are **re-exports of the canonical implementations** in
:mod:`cellquorum.stages.comparative`, not copies: a fix to the engine is a fix here. The
pre-consolidation deep-import paths (``cellquorum.enrichment.ranking`` etc.) have
been removed — import from :mod:`cellquorum.utils` or :mod:`cellquorum.stages.comparative`
instead. Importing this module pulls in no heavy optional
dependency — ``get_net`` lazy-imports ``decoupler`` only when called — preserving
the engine-wide skip-not-crash invariant.
"""

from __future__ import annotations

from cellquorum.stages.comparative.differential_expression.pseudobulk import (
    PseudobulkResult,
    aggregate_pseudobulk,
)
from cellquorum.stages.comparative.enrichment.priors import PriorFetchError, get_net
from cellquorum.stages.comparative.enrichment.ranking import de_table_to_ranking

__all__ = [
    "PriorFetchError",
    "PseudobulkResult",
    "aggregate_pseudobulk",
    "de_table_to_ranking",
    "get_net",
]
