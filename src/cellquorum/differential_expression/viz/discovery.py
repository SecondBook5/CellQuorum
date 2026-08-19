"""IO-only discovery + column normalization for the DE volcano."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_CSV_NAME = "de_pseudobulk_edger.csv"
_LOGFC_ALIASES = ("logFC", "log2FoldChange", "log2fc", "avg_log2FC")
_FDR_ALIASES = ("FDR", "padj", "adj_pvalue", "qvalue")
_GENE_ALIASES = ("gene", "gene_name", "symbol", "names")


def _first_present(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def load_de_table(results_dir: Path) -> pd.DataFrame | None:
    """Return a normalized (gene, logFC, FDR, ...) frame, or None if unusable."""
    results_dir = Path(results_dir)
    path = results_dir / _CSV_NAME
    if not path.exists():
        return None
    try:
        raw = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    if raw.empty:
        return None
    gene_col = _first_present(raw.columns, _GENE_ALIASES)
    fc_col = _first_present(raw.columns, _LOGFC_ALIASES)
    fdr_col = _first_present(raw.columns, _FDR_ALIASES)
    if gene_col is None or fc_col is None or fdr_col is None:
        return None
    out = raw.rename(columns={gene_col: "gene", fc_col: "logFC", fdr_col: "FDR"})
    # Reorder so gene/logFC/FDR lead; keep any extra columns (e.g. likely_ambient).
    lead = ["gene", "logFC", "FDR"]
    rest = [c for c in out.columns if c not in lead]
    out = out[lead + rest]
    if out.empty:
        return None
    return out


__all__ = ["load_de_table"]
