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
instead.

Resolution is **lazy** (:pep:`562`): the re-exports below are wired through
``__getattr__`` so that importing this module costs nothing, and the module holding
a given name is imported on first attribute access. Eagerly re-exporting them used
to pull ``anndata`` (and transitively ``dask.array``) into every process that so much
as ran ``cellquorum --version``, which is what the paragraph above *claimed* was not
happening. ``from cellquorum.utils import get_net`` still works unchanged, because
``from X import Y`` falls back to ``getattr(X, "Y")``. ``get_net`` additionally
lazy-imports ``decoupler`` only when called, preserving the engine-wide
skip-not-crash invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export every public name from its canonical module without importing that
# module yet. Keeping this table beside __all__ means a new export is one line in
# two places, and `test_import_cost.py` asserts the two stay in sync.
_LAZY_EXPORTS: dict[str, str] = {
    "PseudobulkResult": "cellquorum.stages.comparative.differential_expression.pseudobulk",
    "aggregate_pseudobulk": "cellquorum.stages.comparative.differential_expression.pseudobulk",
    "PriorFetchError": "cellquorum.stages.comparative.enrichment.priors",
    "get_net": "cellquorum.stages.comparative.enrichment.priors",
    "de_table_to_ranking": "cellquorum.stages.comparative.enrichment.ranking",
}

# Give type checkers, IDEs, and mkdocstrings the real bindings. This block never
# runs, so it costs nothing at import time, but without it every re-export below
# would look undefined to static analysis.
if TYPE_CHECKING:
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


def __getattr__(name: str) -> object:
    """
    Resolve a lazily re-exported public name on first access (:pep:`562`).

    Args:
        name: Attribute name requested on the ``cellquorum.utils`` module.

    Returns:
        The object of that name from its canonical implementation module.

    Raises:
        AttributeError: If the name is not part of this module's public surface.
    """

    # Look up which canonical module owns this name.
    module_path = _LAZY_EXPORTS.get(name)

    # Reject unknown names the way a normal module would, so typos and `hasattr`
    # probes behave identically to the pre-lazy eager re-export.
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Import the owning module and pull the requested object out of it.
    from importlib import import_module

    value = getattr(import_module(module_path), name)

    # Cache the resolved object on the module so later lookups skip __getattr__
    # entirely and cost exactly what an eager re-export would have.
    globals()[name] = value

    # Return the resolved object to the caller.
    return value


def __dir__() -> list[str]:
    """
    List this module's public surface for ``dir()`` and tab completion.

    Without this, :pep:`562` lazy modules appear empty to interactive
    introspection until each name happens to be touched.

    Returns:
        Sorted public attribute names.
    """

    # Report the documented public surface regardless of what has been resolved.
    return sorted(__all__)
