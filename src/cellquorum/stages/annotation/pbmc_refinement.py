"""RNA-only refinement of three named human PBMC marker-vote panels on a cell graph."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from cellquorum.core.anndata_access import dense_matrix

CD4, CD8, NK = "CD4 T cells", "CD8 T cells", "NK cells"
GROUPS = {
    "T_lineage": ["CD3D", "CD3E", "CD3G", "TRAC"],
    "innate": ["TYROBP", "FCER1G", "KLRF1"],
    "cytotoxic": ["KLRD1", "GNLY"],
    "CD8": ["CD8A", "CD8B"],
    "helper": ["IL7R"],
}


def refine_pbmc(adata: ad.AnnData, layer: str, key: str, cluster_key: str) -> pd.DataFrame:
    """Keep the cluster vote and expose every decision and its local RNA evidence."""
    if "connectivities" not in adata.obsp:
        raise ValueError("human_pbmc_v1 requires the RNA neighbors graph in obsp['connectivities']")
    genes = [gene for members in GROUPS.values() for gene in members if gene in adata.var_names]
    columns = {
        name: [genes.index(g) for g in members if g in genes] for name, members in GROUPS.items()
    }
    missing = [name for name, indices in columns.items() if not indices]
    if missing:
        raise ValueError(f"human_pbmc_v1 lacks all RNA markers for {missing}")
    values = dense_matrix(adata[:, genes].layers[layer])
    graph = sparse.csr_matrix(adata.obsp["connectivities"])
    mass = np.asarray(graph.sum(axis=1)).ravel()
    graph = graph.multiply((1 / np.maximum(mass, 1e-12))[:, None]).tocsr()
    # Average one- and two-step neighbor means to reduce single-cell dropout.
    first = graph @ values
    local = (first + graph @ first) / 2
    evidence = pd.DataFrame(
        {name: local[:, idx].mean(axis=1) for name, idx in columns.items()}, index=adata.obs_names
    )

    baseline = adata.obs[key].astype(str)
    eligible = baseline.isin([CD4, CD8, NK])
    final = baseline.copy()
    reason = pd.Series("cluster vote", index=baseline.index)
    reason[baseline == "Unassigned"] = "no significant cluster panel"
    final[eligible] = "Unassigned"
    reason[eligible] = "weak or conflicting T and innate RNA"
    t_lineage = evidence["T_lineage"]
    innate = evidence["innate"]
    # Fixed Kang development thresholds. A 1.25-fold margin on either side
    # of the weighted innate score leaves mixed lineage evidence unresolved.
    t_cell = eligible & (t_lineage > 0.15) & (t_lineage > 1.25 * 2 * innate)
    nk_cell = (
        eligible
        & (innate > 0.15)
        & (2 * innate > 1.25 * t_lineage)
        & (evidence["cytotoxic"] > 0.15)
    )
    cd8 = evidence["CD8"]
    helper = evidence["helper"]
    # The stricter override in CD4-voted clusters trades sensitivity for precision.
    # IL7R is shared with naive and central-memory CD8 T cells (see the Hao benchmark).
    cd8_cell = t_cell & (cd8 > 0.18) & (cd8 > 1.15 * helper) & ((baseline != CD4) | (cd8 > 0.65))
    cd4_cell = t_cell & (helper > 0.15) & (helper > 1.15 * cd8)
    reason[t_cell] = "T lineage with unresolved CD4/CD8 evidence"
    for mask, label in [(cd4_cell, CD4), (cd8_cell, CD8), (nk_cell, NK)]:
        final[mask] = label
        reason[mask] = "local RNA supports " + label
    reason[eligible & (mass == 0)] = "no connected RNA neighbors"

    adata.obs[f"{key}_cluster_vote"] = pd.Categorical(baseline)
    adata.obs[f"{key}_reason"] = pd.Categorical(reason)
    adata.obs[key] = pd.Categorical(final)
    return (
        pd.concat(
            [
                adata.obs[cluster_key].rename("cluster"),
                baseline.rename("cluster_vote"),
                final.rename("call"),
                reason.rename("reason"),
                evidence,
            ],
            axis=1,
        )
        .rename_axis("barcode")
        .reset_index()
    )
