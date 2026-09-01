"""Reusable statistical primitives that sit on top of engine stage outputs.

These are study-agnostic building blocks — no lineage or study biology is
hardcoded. They take plain numpy/pandas inputs (a per-cell score matrix plus a
metadata frame) rather than an :class:`~anndata.AnnData`, so they are trivially
unit-testable and importable from any hypothesis repo. A caller extracts the
score matrix (e.g. ``obsm["X_state_aucell"]`` written by the ``state_scoring``
stage) and the design columns, then calls these functions.

The module-remodeling suite (donor-aware LMM effect sizes, PERMANOVA-by-group,
signature-argmax subtyping, the signed program-contrast index, leading-edge
concordance, and program correlations) makes rigorous — and reusable — the
by-hand statistics in the LEC "Module Remodeling" analysis. The study-specific
pieces (which gene modules, which subtype signatures, the EndoMT index
definition, figure titles) live in the hypothesis repo, not here.
"""

from __future__ import annotations

from cellquorum.stats.module_remodeling import (
    bh_fdr,
    leading_edge_jaccard,
    lmm_effect_sizes,
    module_gene_overlap,
    permanova_by_group,
    program_correlation_matrix,
    signature_argmax_labels,
    signed_program_contrast_index,
    upset_membership,
)

__all__ = [
    "bh_fdr",
    "leading_edge_jaccard",
    "lmm_effect_sizes",
    "module_gene_overlap",
    "permanova_by_group",
    "program_correlation_matrix",
    "signature_argmax_labels",
    "signed_program_contrast_index",
    "upset_membership",
]
