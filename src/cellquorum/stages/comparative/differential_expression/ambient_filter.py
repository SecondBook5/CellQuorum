"""AnnData/file glue for the optional cross-compartment ambient-gene filter.

The statistical definition (a gene is ambient for a target group when it peaks in a
foreign group at low specificity) lives in
:func:`cellquorum.stats.ambient_specificity.ambient_gene_mask`, which takes a plain
genes-x-groups profile table, not an AnnData. This module is the boilerplate around
it: loading a precomputed reference profile from disk and subsetting an AnnData's
genes to the non-ambient set for one target group.

The reference profile must come from a WIDER cohort than the one being filtered (a
single-lineage run has no other lineage's expression to compare against), so building
it is deliberately out of scope here — the caller supplies a path to an already-built
table (e.g. produced once from a multi-lineage atlas and cached).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.stats.ambient_specificity import ambient_gene_mask


def load_expression_profile(path: str | Path) -> pd.DataFrame:
    """
    Load a genes (index) x groups (columns) mean-expression reference table.

    Args:
        path: CSV or parquet file. The first CSV column is used as the gene index.

    Returns:
        The loaded profile DataFrame.
    """

    resolved = Path(path)
    if resolved.suffix == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved, index_col=0)


def filter_ambient_genes(
    adata: ad.AnnData,
    *,
    profile: pd.DataFrame,
    target_group: str,
    spec_threshold: float = 0.25,
    group_map: dict[str, str] | None = None,
) -> tuple[ad.AnnData, dict]:
    """
    Subset ``adata`` to genes not flagged ambient for ``target_group``.

    Args:
        adata: The AnnData being filtered (already scoped to the target group's cells).
        profile: Genes x groups reference profile (see ``ambient_gene_mask``).
        target_group: The group ``adata`` represents in ``profile``'s columns.
        spec_threshold: Passed through to ``ambient_gene_mask``.
        group_map: Passed through to ``ambient_gene_mask``.

    Returns:
        ``(filtered_adata, info)`` where ``info`` records how many genes were dropped.
        A gene present in ``adata`` but absent from ``profile`` cannot be judged and is
        kept (the reference simply never covered it, which is not evidence of ambient
        contamination).
    """

    mask = ambient_gene_mask(
        profile,
        target_group=target_group,
        spec_threshold=spec_threshold,
        group_map=group_map,
    )
    keep = mask.reindex(adata.var_names, fill_value=True)
    dropped = [str(g) for g in adata.var_names[~keep.to_numpy()]]
    filtered = adata[:, keep.to_numpy()].copy()
    info = {
        "n_genes_before": int(adata.n_vars),
        "n_genes_after": int(filtered.n_vars),
        "n_dropped": len(dropped),
        "dropped_genes": dropped,
    }
    return filtered, info


__all__ = ["load_expression_profile", "filter_ambient_genes"]
