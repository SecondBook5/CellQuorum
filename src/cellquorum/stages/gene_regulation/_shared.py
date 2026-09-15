"""Small helpers shared across the gene-regulation stage category.

Kept local to gene_regulation rather than a generic core location: the
"cell_type -> leiden -> <default>" precedence is a convention specific to how
hdWGCNA and pySCENIC pick their grouping variable, not a general engine
concept.
"""

from __future__ import annotations

import anndata as ad


def resolve_group_by(adata: ad.AnnData, configured: str | None, *, default: str) -> str:
    """Resolve a stage's grouping column.

    Precedence: an explicit ``configured`` value is always trusted as-is (it
    is not validated against ``adata.obs`` here -- each caller's own
    downstream code decides how to handle a configured-but-absent column).
    Otherwise falls back to the first of ``cell_type`` / ``leiden`` present in
    ``adata.obs``, and to ``default`` when neither is.

    Args:
        adata: The object being analyzed.
        configured: The stage's configured group_by, or None.
        default: Value to use when nothing else resolves. Callers differ here
            deliberately (e.g. hdWGCNA's R script does its own "all" fallback
            when the named column is absent, so it defaults to "cell_type";
            pySCENIC checks column presence in Python, so it defaults to
            "all") -- this function does not decide that policy.

    Returns:
        The resolved grouping column name.
    """
    if configured:
        return configured
    if "cell_type" in adata.obs.columns:
        return "cell_type"
    if "leiden" in adata.obs.columns:
        return "leiden"
    return default


__all__ = ["resolve_group_by"]
