"""Convert a differential-expression table to a preranked GSEA contrast vector."""

from __future__ import annotations

import numpy as np
import pandas as pd


def de_table_to_ranking(
    df: pd.DataFrame,
    gene_col: str = "gene",
    lfc_col: str = "logFC",
    p_col: str = "PValue",
) -> pd.DataFrame:
    """Build a 1-row preranked contrast vector from a DE table.

    Metric = ``sign(logFC) * -log10(PValue)`` (the signed -log10 p-value AJ used
    for single-cell preranked GSEA). Duplicate genes are collapsed to the row of
    largest |metric|; genes with non-finite metric (NaN inputs, or p==0 → inf)
    are dropped. Returns a DataFrame indexed ``["contrast"]`` with genes as
    columns — the shape decoupler's ``gsea`` consumes for a single contrast.

    Args:
        df: DE result table (e.g. edgeR output).
        gene_col: Column holding gene symbols.
        lfc_col: Column holding log fold-change.
        p_col: Column holding the raw p-value.

    Returns:
        A one-row DataFrame (index ``["contrast"]``) of gene → signed metric.
    """

    work = df[[gene_col, lfc_col, p_col]].copy()
    work.columns = ["gene", "logFC", "PValue"]

    # Signed -log10 p metric.
    metric = np.sign(work["logFC"]) * -np.log10(work["PValue"])
    work = work.assign(metric=metric)

    # Drop non-finite metric (NaN logFC/PValue, or p==0 → inf).
    work = work[np.isfinite(work["metric"])]

    # Collapse duplicate genes to the row of largest |metric|.
    work = work.assign(abs_metric=work["metric"].abs())
    work = work.sort_values("abs_metric", ascending=False).drop_duplicates("gene")

    # Sort descending by metric (preranked convention).
    work = work.sort_values("metric", ascending=False)

    return pd.DataFrame([work.set_index("gene")["metric"]], index=["contrast"])


__all__ = ["de_table_to_ranking"]
