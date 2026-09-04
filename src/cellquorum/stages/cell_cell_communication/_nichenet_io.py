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


def de_to_geneset(
    de_df: pd.DataFrame, fdr: float, top_n: int, direction: str = "up"
) -> tuple[list[str], list[str]]:
    """Build (receiver geneset, background) from a pseudobulk DE table.

    Background = all tested genes (sorted, deduped). Geneset = genes passing FDR < ``fdr``
    in the requested ``direction``, capped to top-``top_n`` by absolute logFC. Returned
    sorted for deterministic downstream ordering.

    ``direction`` defaults to ``"up"`` because NicheNet's ligand-target matrix holds
    *positive* regulatory potential: it scores how well a ligand's predicted targets explain
    the gene set. A set that mixes induced and repressed genes asks the model a question it
    cannot answer, and the resulting AUPR is not interpretable in either direction. ``"both"``
    remains available for callers whose model is direction-agnostic.

    Parameters
    ----------
    de_df
        Table with ``gene``, ``logFC`` and ``FDR`` columns.
    fdr
        Significance threshold, applied to ``FDR``.
    top_n
        Cap on the geneset size, taken by ``|logFC|``.
    direction
        ``"up"``, ``"down"`` or ``"both"``.
    """
    if direction not in {"up", "down", "both"}:
        raise ValueError(f"direction must be 'up', 'down' or 'both', got {direction!r}")

    background = sorted(pd.unique(de_df["gene"].astype(str)).tolist())

    sig = de_df[de_df["FDR"] < fdr].copy()
    if direction == "up":
        sig = sig[sig["logFC"] > 0]
    elif direction == "down":
        sig = sig[sig["logFC"] < 0]
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


def attribute_senders(
    canonical: pd.DataFrame,
    expression: pd.DataFrame,
    *,
    sender_label: str,
    min_fraction: float | None = None,
) -> pd.DataFrame:
    """Replace a pooled sender label with one row per sender that expresses the ligand.

    A multi-sender NicheNet run ranks ligands once against the union of what all senders
    express, so the ranking has no single source. Writing the joined label ("Fibro, Mac, T")
    into a ``source`` column would be a value no downstream consumer can detect as a group,
    and ``ccc_network`` would draw it as a cell type that does not exist. Attribution is an
    expression question, answered here from the per-sender expression table.

    Rows whose ``source`` is not ``sender_label`` pass through untouched. Ligands no sender
    expresses are dropped: a network edge needs a source, and there is none. Callers that care
    how many were dropped should compare lengths.

    Parameters
    ----------
    canonical
        Canonical LR table, as returned by :func:`ligand_activity_to_canonical`.
    expression
        Per-sender expression table with ``sender``, ``ligand`` and either ``expressed`` or
        ``fraction_expressing``.
    sender_label
        The pooled label to expand.
    min_fraction
        Threshold on ``fraction_expressing``, used when ``expressed`` is absent.
    """
    if canonical is None or canonical.empty:
        return _empty_canonical()
    if expression is None or not {"sender", "ligand"}.issubset(expression.columns):
        return canonical.reset_index(drop=True)

    expressing = expression
    if "expressed" in expression.columns:
        flag = expression["expressed"]
        # R writes booleans as the strings "TRUE"/"FALSE", and bool("FALSE") is True.
        if flag.dtype == object:
            flag = flag.astype(str).str.upper().isin({"TRUE", "T", "1"})
        expressing = expression[flag.astype(bool)]
    elif min_fraction is not None and "fraction_expressing" in expression.columns:
        expressing = expression[expression["fraction_expressing"].astype(float) >= min_fraction]

    per_ligand: dict[str, list[str]] = {}
    for ligand, sender in zip(
        expressing["ligand"].astype(str), expressing["sender"].astype(str), strict=False
    ):
        per_ligand.setdefault(ligand, []).append(sender)

    pooled = canonical["source"].astype(str) == str(sender_label)
    kept = [canonical.loc[~pooled]]
    for _, row in canonical.loc[pooled].iterrows():
        senders = per_ligand.get(str(row["ligand"]), [])
        if not senders:
            continue
        block = pd.DataFrame([row] * len(senders)).reset_index(drop=True)
        block["source"] = senders
        kept.append(block)

    out = pd.concat(kept, ignore_index=True) if kept else _empty_canonical()
    if out.empty:
        return _empty_canonical()
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
