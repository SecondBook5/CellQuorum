# src/cellquorum/trajectory/_loom_io.py
"""Biology-free loom ingestion: corrupt-tolerant read + barcode reconciliation.

velocyto loom stores genes×cells with layers ``spliced``/``unspliced`` and a
``CellID`` column attribute formatted ``"<stem>:<BARCODE>x"``. The working atlas
names cells ``"<sample_id>_<BARCODE>-1"``. Reconciliation is per-sample so
barcodes never collide across samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.sparse import csr_matrix


def loom_cellid_to_bare(cell_id: str) -> str:
    """``"<stem>:<BARCODE>x"`` → ``"<BARCODE>-1"`` (tool-format constant)."""
    barcode = cell_id.split(":", 1)[1]
    return barcode.rstrip("x") + "-1"


def read_loom_layers(
    path: str | Path,
) -> tuple[csr_matrix, csr_matrix, NDArray[np.str_], NDArray[np.str_]] | None:
    """Read a velocyto loom, returning cells×genes spliced/unspliced or None.

    Deliberately tolerant: any exception (missing file, truncated/corrupt layer)
    returns None so the caller can skip this sample and continue.

    Returns:
        ``(spliced_csr, unspliced_csr, gene_names, cell_ids)`` with matrices in
        cells×genes orientation, or None.
    """
    try:
        import loompy
        from scipy import sparse

        with loompy.connect(str(path), validate=False) as ds:
            gene_names = np.asarray(ds.ra["Gene"]).astype(str)
            cell_ids = np.asarray(ds.ca["CellID"]).astype(str)
            # loompy is genes×cells; transpose to cells×genes.
            spliced = sparse.csr_matrix(ds.layers["spliced"][:, :]).T.tocsr()
            unspliced = sparse.csr_matrix(ds.layers["unspliced"][:, :]).T.tocsr()
        return spliced, unspliced, gene_names, cell_ids
    except Exception:
        return None


def _reindex_columns(
    mat: csr_matrix,
    source_genes: pd.Index | NDArray[np.str_],
    target_genes: pd.Index | NDArray[np.str_],
) -> csr_matrix:
    """Reindex a cells×genes CSR from source_genes onto target_genes (fill 0)."""
    from scipy import sparse

    source_index = pd.Index(np.asarray(source_genes).astype(str))
    col_for_target = source_index.get_indexer(pd.Index(np.asarray(target_genes).astype(str)))
    n_cells = mat.shape[0]
    n_target = len(target_genes)
    out = sparse.lil_matrix((n_cells, n_target), dtype=mat.dtype)
    csc = mat.tocsc()
    for target_col, source_col in enumerate(col_for_target):
        if source_col >= 0:
            out[:, target_col] = csc[:, source_col]
    return out.tocsr()


def reconcile_looms(
    adata: ad.AnnData,
    manifest: pd.DataFrame,
    *,
    sample_col: str,
    loom_path_col: str,
) -> tuple[ad.AnnData | None, list[str]]:
    """Attach reconciled spliced/unspliced layers to a subset of ``adata``.

    Iterates samples in sorted order, reads each loom, maps loom barcodes to the
    atlas obs_names for that sample only (no cross-sample collision), concatenates,
    intersects with the atlas, and reindexes layers onto ``adata.var_names``.

    Returns:
        ``(out_adata, notes)`` where ``out_adata`` is the atlas subset carrying
        ``layers['spliced']``/``layers['unspliced']``, or ``(None, notes)`` when
        no sample produced overlapping cells.
    """
    from scipy import sparse

    notes: list[str] = []
    if loom_path_col not in manifest.columns or sample_col not in manifest.columns:
        notes.append(f"manifest missing '{sample_col}' or '{loom_path_col}' column")
        return None, notes

    atlas_samples = adata.obs[sample_col].astype(str)
    parts: list[ad.AnnData] = []

    for sample_id in sorted(manifest[sample_col].astype(str).unique()):
        rows = manifest[manifest[sample_col].astype(str) == sample_id]
        loom_path = rows[loom_path_col].iloc[0] if len(rows) else None
        if loom_path is None or (isinstance(loom_path, float) and np.isnan(loom_path)):
            notes.append(f"{sample_id}: no loom_path")
            continue
        if not Path(str(loom_path)).exists():
            notes.append(f"{sample_id}: loom_path does not exist")
            continue
        read = read_loom_layers(loom_path)
        if read is None:
            notes.append(f"{sample_id}: loom unreadable/corrupt")
            continue
        spliced, unspliced, gene_names, cell_ids = read

        obj_names = adata.obs_names[(atlas_samples == sample_id).to_numpy()]
        prefix = f"{sample_id}_"
        bare_to_obj: dict[str, str] = {}
        for name in obj_names:
            bare = name[len(prefix) :] if name.startswith(prefix) else str(name)
            bare_to_obj[bare] = str(name)

        loom_bare = np.array([loom_cellid_to_bare(c) for c in cell_ids])
        keep = np.array([b in bare_to_obj for b in loom_bare])
        if not keep.any():
            notes.append(f"{sample_id}: no barcode overlap with atlas")
            continue

        new_names = [bare_to_obj[b] for b in loom_bare[keep]]
        part = ad.AnnData(
            X=spliced[keep],
            obs=pd.DataFrame(index=new_names),
            var=pd.DataFrame(index=np.asarray(gene_names).astype(str)),
        )
        part.layers["spliced"] = spliced[keep]
        part.layers["unspliced"] = unspliced[keep]
        parts.append(part)

    if not parts:
        notes.append("no sample produced spliced/unspliced layers")
        return None, notes

    combined = ad.concat(parts, join="outer", index_unique=None)
    combined_set = set(combined.obs_names)
    shared = [n for n in adata.obs_names if n in combined_set]
    if not shared:
        notes.append("no atlas cells overlap the reconciled looms")
        return None, notes

    combined = combined[shared]
    spliced_re = _reindex_columns(
        sparse.csr_matrix(combined.layers["spliced"]), combined.var_names, adata.var_names
    )
    unspliced_re = _reindex_columns(
        sparse.csr_matrix(combined.layers["unspliced"]), combined.var_names, adata.var_names
    )

    out = adata[shared].copy()
    out.layers["spliced"] = spliced_re
    out.layers["unspliced"] = unspliced_re
    notes.append(f"reconciled {len(shared)} cells across {len(parts)} sample(s)")
    return out, notes


__all__ = ["read_loom_layers", "loom_cellid_to_bare", "reconcile_looms"]
