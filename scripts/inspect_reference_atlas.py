"""Inspect the SCP2613 reference atlas for the keys and gene overlap the run needs.

Usage:
    python scripts/inspect_reference_atlas.py <atlas.h5ad> <one_query_filtered_h5>
"""

from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import scanpy as sc


def main(atlas_path: str, query_h5: str) -> int:
    atlas = ad.read_h5ad(atlas_path, backed="r")
    obs = atlas.obs
    print("atlas shape:", atlas.shape)

    for key in ("disease_lesional", "Cell_type", "Cell_type_granular"):
        if key in obs.columns:
            vals = obs[key].astype(str).unique()[:20]
            print(f"[{key}] n={obs[key].nunique()} sample={list(vals)}")
        else:
            print(f"[{key}] MISSING")

    for cand in ("sample_name", "biosample_id", "donor_id", "batch"):
        if cand in obs.columns:
            print(f"batch-candidate [{cand}] n={obs[cand].nunique()}")

    has_counts = "counts" in atlas.layers
    print("has counts layer:", has_counts)
    if has_counts:
        c = atlas.layers["counts"][:50]
        try:
            c = c.toarray()
        except Exception:
            pass
        print("counts integer?", bool(np.allclose(c, np.round(c))))

    atlas_genes = set(map(str, atlas.var_names))
    q = sc.read_10x_h5(query_h5)
    q.var_names_make_unique()
    query_genes = set(map(str, q.var_names))
    overlap = atlas_genes & query_genes
    print(f"atlas genes={len(atlas_genes)} query genes={len(query_genes)} overlap={len(overlap)}")
    if len(overlap) < 10000:
        print("WARNING: gene overlap < 10000 — check symbol vs Ensembl id mismatch.")
    atlas.file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
