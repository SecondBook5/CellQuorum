"""Biology-free IO + adapter helpers for the NicheNet/MultiNicheNet methods."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
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
