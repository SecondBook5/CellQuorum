"""Biology-free IO helpers for the DIALOGUE method."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def export_dialogue_inputs(
    adata: ad.AnnData,
    *,
    cell_type_col: str,
    sample_col: str,
    use_rep: str,
    n_pcs: int,
    layer: str | None,
    quality_col: str | None,
    condition_col: str | None,
    confounders: list[str],
    min_cells_per_type: int,
    scratch: Path,
) -> dict:
    """Export per-cell-type files for the DIALOGUE R script.

    Writes:
      - celltypes.json: {stripped_name: {"label": original, "dir": stripped_name}}
      - For each eligible cell type (subdir named by stripped name):
          expr.mtx   genes x cells sparse matrix
          genes.txt  one gene per line
          cells.txt  one cell per line
          X.csv      cells x features (first col 'cell', then n_pcs PCs)
          meta.csv   first col 'cell', then 'sample', 'cellQ', optional pheno + confounders

    Args:
        adata: AnnData object.
        cell_type_col: obs column for cell type labels.
        sample_col: obs column for sample IDs.
        use_rep: obsm key for dimensionality reduction (e.g., "X_pca").
        n_pcs: number of PCs to export from use_rep.
        layer: layer to use for expression (None = .X).
        quality_col: obs column for pre-computed cell quality (None = auto-compute).
        condition_col: obs column for phenotype (None = omit from meta.csv).
        confounders: list of obs columns to include as confounders.
        min_cells_per_type: minimum cells per type to include.
        scratch: directory to write files.

    Returns:
        {"scratch": Path, "cell_types": {stripped: original}, "n_samples": int}
    """
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    # Compute n_samples from the full adata (before subsetting)
    n_samples = adata.obs[sample_col].nunique()

    # Strip underscores from cell type names (DIALOGUE R script does this)
    cell_types = adata.obs[cell_type_col].unique()
    cell_type_map = {}  # stripped -> original
    celltypes_json = {}  # stripped -> {"label": original, "dir": stripped}

    for ct in cell_types:
        stripped = ct.replace("_", "")
        # Subset to this cell type
        mask = adata.obs[cell_type_col] == ct
        n_cells = mask.sum()

        if n_cells < min_cells_per_type:
            continue

        cell_type_map[stripped] = ct
        celltypes_json[stripped] = {"label": ct, "dir": stripped}

        # Create subdir
        ct_dir = scratch / stripped
        ct_dir.mkdir(parents=True, exist_ok=True)

        # Subset adata
        adata_sub = adata[mask].copy()

        # Get expression matrix (layer or X)
        if layer is not None:
            mat = adata_sub.layers[layer]
        else:
            mat = adata_sub.X

        # Convert to sparse if needed
        if not sp.issparse(mat):
            mat = sp.csr_matrix(mat)

        # Write expr.mtx (genes × cells, so transpose)
        scipy.io.mmwrite(ct_dir / "expr.mtx", mat.T.tocoo())

        # Write genes.txt
        with open(ct_dir / "genes.txt", "w") as f:
            for gene in adata_sub.var_names:
                f.write(f"{gene}\n")

        # Write cells.txt
        cell_ids = list(adata_sub.obs_names)
        with open(ct_dir / "cells.txt", "w") as f:
            for cell_id in cell_ids:
                f.write(f"{cell_id}\n")

        # Write X.csv (cells × features, first col 'cell')
        X_mat = adata_sub.obsm[use_rep][:, :n_pcs]
        X_df = pd.DataFrame(X_mat, columns=[f"PC{i+1}" for i in range(n_pcs)])
        X_df.insert(0, "cell", cell_ids)
        X_df.to_csv(ct_dir / "X.csv", index=False)

        # Compute cellQ
        if quality_col is not None:
            cellQ = adata_sub.obs[quality_col].to_numpy()
        else:
            # Count detected genes per cell (non-zero entries)
            if sp.issparse(mat):
                cellQ = np.array((mat > 0).sum(axis=1)).flatten()
            else:
                cellQ = (mat > 0).sum(axis=1)

        # Build meta.csv
        meta_df = pd.DataFrame({"cell": cell_ids})
        meta_df["sample"] = adata_sub.obs[sample_col].to_numpy()
        meta_df["cellQ"] = cellQ

        # Add optional phenotype column
        if condition_col is not None:
            meta_df["pheno"] = adata_sub.obs[condition_col].to_numpy()

        # Add confounders
        for conf in confounders:
            meta_df[conf] = adata_sub.obs[conf].to_numpy()

        meta_df.to_csv(ct_dir / "meta.csv", index=False)

    # Write celltypes.json
    with open(scratch / "celltypes.json", "w") as f:
        json.dump(celltypes_json, f, indent=2)

    return {
        "scratch": scratch,
        "cell_types": cell_type_map,
        "n_samples": int(n_samples),
    }


def read_dialogue_outputs(out_dir: Path) -> dict[str, pd.DataFrame]:
    """Read DIALOGUE output CSVs (empty-but-headed if missing).

    Args:
        out_dir: directory containing mcp_*.csv files.

    Returns:
        {"programs": df, "scores": df, "associations": df}
    """
    out_dir = Path(out_dir)

    # Canonical schemas
    prog_cols = ["program", "cell_type", "gene", "loading", "direction"]
    score_cols = ["cell_id", "sample", "cell_type", "program", "score"]
    assoc_cols = ["program", "statistic", "pvalue", "padj", "direction"]

    def _read_or_empty(fname: str, cols: list[str]) -> pd.DataFrame:
        path = out_dir / fname
        if path.exists():
            return pd.read_csv(path)
        else:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    return {
        "programs": _read_or_empty("mcp_gene_programs.csv", prog_cols),
        "scores": _read_or_empty("mcp_scores.csv", score_cols),
        "associations": _read_or_empty("mcp_associations.csv", assoc_cols),
    }
