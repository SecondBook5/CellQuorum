"""Biology-free network construction + LR adapters for ccc_network.

No anndata, no optional deps, no I/O. The canonical LR edge schema
(source, target, ligand, receptor, weight, sample[, condition]) is the contract
between the adapter, the network builders, and the two methods.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

# Canonical LR edge columns in stable order.
CANONICAL_COLUMNS = ["source", "target", "ligand", "receptor", "weight", "sample"]


def liana_to_canonical(liana_res: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map spec #1's ``uns['liana_res']`` frame to the canonical LR schema.

    ``magnitude_rank`` is a rank in [0, 1] where LOWER is stronger, so the
    canonical weight is ``1 - magnitude_rank`` (higher = stronger), matching the
    ``inverse_fun`` convention used by the Tensor method.

    Rows missing any required source column produce an empty frame and a note.

    Returns
    -------
    (canonical_df, notes)
    """
    notes: list[str] = []
    required = [
        "sample",
        "source",
        "target",
        "ligand_complex",
        "receptor_complex",
        "magnitude_rank",
    ]
    missing = [c for c in required if c not in liana_res.columns]
    if missing:
        notes.append(f"liana_to_canonical: source frame missing columns {missing}; no edges built.")
        return pd.DataFrame(columns=CANONICAL_COLUMNS), notes

    df = liana_res.loc[:, required].copy()
    # Drop rows with any null in a required field (skip-not-crash on dirty input).
    n_before = len(df)
    df = df.dropna(subset=required)
    if len(df) < n_before:
        notes.append(f"liana_to_canonical: dropped {n_before - len(df)} rows with missing values.")

    canon = pd.DataFrame(
        {
            "source": df["source"].astype(str).to_numpy(),
            "target": df["target"].astype(str).to_numpy(),
            "ligand": df["ligand_complex"].astype(str).to_numpy(),
            "receptor": df["receptor_complex"].astype(str).to_numpy(),
            "weight": 1.0 - df["magnitude_rank"].astype(float).to_numpy(),
            "sample": df["sample"].astype(str).to_numpy(),
        }
    )
    return canon.loc[:, CANONICAL_COLUMNS], notes


def build_cci_network(lr: pd.DataFrame, name: str = "cci") -> nx.DiGraph:
    """Cell-cell interaction network: nodes = cell types, edge weight = sum(weight)."""
    G = nx.DiGraph()
    G.graph["name"] = name
    if lr.empty:
        return G
    # Deterministic: aggregate then iterate in a stable, sorted order.
    agg = lr.groupby(["source", "target"], sort=True)["weight"].sum().reset_index()
    agg = agg.sort_values(["source", "target"], kind="mergesort")
    for _, row in agg.iterrows():
        G.add_edge(
            row["source"], row["target"], weight=float(row["weight"]), LRScore=float(row["weight"])
        )
    return G


def build_gci_network(lr: pd.DataFrame, name: str = "gci") -> nx.DiGraph:
    """Gene-channel network: nodes = '{celltype}/{gene}|L' / '|R', one edge per LR row."""
    G = nx.DiGraph()
    G.graph["name"] = name
    if lr.empty:
        return G
    ordered = lr.sort_values(["source", "target", "ligand", "receptor"], kind="mergesort")
    for _, row in ordered.iterrows():
        src = f"{row['source']}/{row['ligand']}|L"
        tgt = f"{row['target']}/{row['receptor']}|R"
        w = float(row["weight"])
        # Multiple samples can repeat a channel; accumulate onto the edge.
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += w
            G[src][tgt]["LRScore"] += w
        else:
            G.add_edge(
                src, tgt, weight=w, LRScore=w, ligand=row["ligand"], receptor=row["receptor"]
            )
    return G


def _safe_pagerank(G: nx.DiGraph) -> dict[str, float]:
    """PageRank that never aborts: retry looser tolerances, then weighted-in-degree fallback."""
    if G.number_of_edges() == 0:
        return {}
    for max_iter, tol in ((500, 1e-6), (1000, 1e-4)):
        try:
            return nx.pagerank(G, weight="weight", max_iter=max_iter, tol=tol)
        except nx.PowerIterationFailedConvergence:
            continue
    indeg = dict(G.in_degree(weight="weight"))
    total = sum(abs(v) for v in indeg.values()) or 1.0
    return {n: abs(v) / total for n, v in indeg.items()}


__all__ = [
    "CANONICAL_COLUMNS",
    "liana_to_canonical",
    "build_cci_network",
    "build_gci_network",
    "_safe_pagerank",
]
