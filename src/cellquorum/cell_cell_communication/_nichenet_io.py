"""Biology-free IO + adapter helpers for the NicheNet/MultiNicheNet methods."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def export_sce_inputs(adata: ad.AnnData, obs_cols: list[str], scratch: Path) -> dict[str, Path]:
    """Export counts + metadata for building a SingleCellExperiment in R.

    Writes genes x cells matrix-market counts plus gene/barcode/obs CSVs. The
    R side reconstructs an SCE from these (deliberately avoiding SeuratDisk).
    """
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    # genes x cells (R/Bioconductor orientation).
    mat = adata.X
    mat = sp.csr_matrix(mat) if not sp.issparse(mat) else mat
    counts = scratch / "counts.mtx"
    scipy.io.mmwrite(counts, mat.T.tocoo())

    genes = scratch / "genes.csv"
    pd.DataFrame({"gene": list(adata.var_names)}).to_csv(genes, index=False)

    barcodes = scratch / "barcodes.csv"
    pd.DataFrame({"barcode": list(adata.obs_names)}).to_csv(barcodes, index=False)

    obs = scratch / "obs.csv"
    meta = adata.obs[list(obs_cols)].copy()
    meta.insert(0, "barcode", list(adata.obs_names))
    meta.to_csv(obs, index=False)

    return {"counts": counts, "genes": genes, "barcodes": barcodes, "obs": obs}


def de_to_geneset(de_df: pd.DataFrame, fdr: float, top_n: int) -> tuple[list[str], list[str]]:
    """Build (receiver geneset, background) from a pseudobulk DE table.

    Background = all tested genes (sorted, deduped). Geneset = genes passing
    FDR < ``fdr``, capped to top-``top_n`` by absolute logFC. Returned sorted
    for deterministic downstream ordering.
    """
    background = sorted(pd.unique(de_df["gene"].astype(str)).tolist())

    sig = de_df[de_df["FDR"] < fdr].copy()
    if sig.empty:
        return [], background

    sig["_abs"] = sig["logFC"].abs()
    sig = sig.sort_values("_abs", ascending=False, kind="mergesort")
    top = sig.head(int(top_n))
    return sorted(top["gene"].astype(str).tolist()), background


# Canonical LR schema (contract with spec #3).
CANONICAL_COLUMNS = ["source", "target", "ligand", "receptor", "weight", "sample", "condition"]


def _empty_canonical() -> pd.DataFrame:
    """Return an empty DataFrame matching the canonical LR schema."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_COLUMNS})


def mnn_prioritization_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Map a MultiNicheNet group_prioritization_tbl to the canonical LR schema.

    Requires columns: sender, receiver, ligand, receptor, prioritization_score, group.
    Maps to: source, target, ligand, receptor, weight, sample, condition.
    Drops rows missing any required column.
    """
    required = ["sender", "receiver", "ligand", "receptor", "prioritization_score", "group"]
    if df is None or any(c not in df.columns for c in required):
        return _empty_canonical()
    sub = df.dropna(subset=required)
    if sub.empty:
        return _empty_canonical()
    out = pd.DataFrame(
        {
            "source": sub["sender"].astype(str).to_numpy(),
            "target": sub["receiver"].astype(str).to_numpy(),
            "ligand": sub["ligand"].astype(str).to_numpy(),
            "receptor": sub["receptor"].astype(str).to_numpy(),
            "weight": np.clip(sub["prioritization_score"].astype(float).to_numpy(), 0.0, None),
            "sample": "",
            "condition": sub["group"].astype(str).to_numpy(),
        }
    )
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def ligand_activity_to_canonical(
    df: pd.DataFrame, sender: str, receiver: str, condition: str | None
) -> pd.DataFrame:
    """Map a NicheNet ligand-activity table to the canonical LR schema.

    Requires columns: ligand, receptor, aupr_corrected.
    Maps to: source, target, ligand, receptor, weight, sample, condition.
    Clamps weight (aupr_corrected) to [0.0, inf). Drops rows with NaN in required columns.
    """
    required = ["ligand", "receptor", "aupr_corrected"]
    if df is None or any(c not in df.columns for c in required):
        return _empty_canonical()
    sub = df.dropna(subset=required)
    if sub.empty:
        return _empty_canonical()
    out = pd.DataFrame(
        {
            "source": str(sender),
            "target": str(receiver),
            "ligand": sub["ligand"].astype(str).to_numpy(),
            "receptor": sub["receptor"].astype(str).to_numpy(),
            "weight": np.clip(sub["aupr_corrected"].astype(float).to_numpy(), 0.0, None),
            "sample": "",
            "condition": condition or "",
        }
    )
    return out[CANONICAL_COLUMNS].reset_index(drop=True)
