"""Biology-free network construction + LR adapters for ccc_network.

No anndata, no optional deps, no I/O. The canonical LR edge schema
(source, target, ligand, receptor, weight, sample[, condition]) is the contract
between the adapter, the network builders, and the two methods.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
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


def _with_allpair(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 'allpair' merge key and aggregate weight per channel (deterministic)."""
    out = df.copy()
    out["allpair"] = (
        out["source"] + "/" + out["ligand"] + "@" + out["target"] + "/" + out["receptor"]
    )
    # Collapse multi-sample rows to one weight per channel.
    agg = (
        out.groupby(["allpair", "source", "target", "ligand", "receptor"], sort=True)["weight"]
        .sum()
        .reset_index()
    )
    return agg


def build_differential_network(
    lr_control: pd.DataFrame,
    lr_case: pd.DataFrame,
    control_name: str,
    case_name: str,
) -> tuple[pd.DataFrame, nx.DiGraph, nx.DiGraph]:
    """Signed differential network: weight = case - control (positive = gained in case)."""
    ctrl = _with_allpair(lr_control)
    case = _with_allpair(lr_case)

    merged = pd.merge(case, ctrl, on="allpair", how="outer", suffixes=("_case", "_ctrl"))
    merged["weight_case"] = merged["weight_case"].fillna(0.0)
    merged["weight_ctrl"] = merged["weight_ctrl"].fillna(0.0)
    merged["weight"] = merged["weight_case"] - merged["weight_ctrl"]

    # Coalesce identity columns ONCE (case value, else control value).
    for col in ("source", "target", "ligand", "receptor"):
        merged[col] = merged[f"{col}_case"].combine_first(merged[f"{col}_ctrl"])

    diff_table = merged.loc[
        merged["weight"] != 0, ["source", "target", "ligand", "receptor", "weight", "allpair"]
    ].copy()
    diff_table = diff_table.sort_values("allpair", kind="mergesort").reset_index(drop=True)

    name = f"{case_name}_vs_{control_name}"
    diff_cci = build_cci_network(diff_table, name=f"{name}_cci")
    diff_gci = build_gci_network(diff_table, name=f"{name}_gci")
    return diff_table, diff_cci, diff_gci


def _comparative_pagerank(
    pr_case: dict[str, float],
    pr_ctrl: dict[str, float],
    nodes: list[str],
    alpha: float = 0.01,
) -> dict[str, float]:
    """Bayesian comparative PageRank: log(P_case / P_control) per node."""
    result: dict[str, float] = {}
    for n in nodes:
        p = pr_ctrl.get(n, 0.0) + alpha
        q = pr_case.get(n, 0.0) + alpha
        p_norm = p / (p + q)
        q_norm = q / (p + q)
        result[n] = float(np.log(q_norm / p_norm)) if p_norm > 0 and q_norm > 0 else 0.0
    return result


def resolve_condition_arms(
    adata: object,
    canon: pd.DataFrame,
    sample_col: str,
    condition_col: str | None,
    case: object,
    control: object,
    notes: list[str],
    warnings: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split the canonical table into case/control arms via an obs sample->condition map.

    Returns an empty dict if design is absent or obs lacks required columns;
    otherwise returns {"case": case_df, "control": control_df}.

    A design that is configured but cannot be resolved goes to ``warnings``, not
    ``notes``. On the LEC arm it was a note, and so the run reported success while
    the comparative Lymphedema-vs-Normal topology and curvature — the reason the
    stage is in the pipeline — had silently not been computed. ``notes`` is still
    accepted (and used when no warnings list is passed) so callers need not change
    at once.
    """
    arms: dict[str, pd.DataFrame] = {}
    if not (condition_col and case and control):
        return arms
    missing = [c for c in (sample_col, condition_col) if c not in adata.obs.columns]
    if missing:
        (warnings if warnings is not None else notes).append(
            f"comparative CCC skipped: design declares {case} vs {control} but obs "
            f"lacks {missing} (have sample_col='{sample_col}', "
            f"condition_col='{condition_col}')"
        )
        return arms
    mapping = (
        adata.obs[[sample_col, condition_col]]
        .astype(str)
        .drop_duplicates()
        .set_index(sample_col)[condition_col]
        .to_dict()
    )
    cond = canon["sample"].map(mapping)
    arms["case"] = canon.loc[cond == str(case)].copy()
    arms["control"] = canon.loc[cond == str(control)].copy()
    return arms


def compute_topology_ranking(
    G: nx.DiGraph,
    comparative: bool = False,
    G_other: nx.DiGraph | None = None,
    pagerank_alpha: float = 0.01,
    G_diff: nx.DiGraph | None = None,
) -> pd.DataFrame:
    """Topology ranking. Single: Listener/Influencer/Mediator/Pagerank.

    Comparative (G=case, G_other=control, G_diff=signed differential network):
    - Listener/Influencer from G_diff (signed weighted degree, case-control)
    - Mediator from betweenness(case) - betweenness(control)
    - Pagerank from log-ratio of the two arms
    """
    cols = ["node", "Listener", "Influencer", "Mediator", "Pagerank"]
    if G.number_of_nodes() == 0 and (G_other is None or G_other.number_of_nodes() == 0):
        return pd.DataFrame(columns=cols)

    if not comparative:
        nodes = sorted(G.nodes())
        in_deg = dict(G.in_degree(weight="weight"))
        out_deg = dict(G.out_degree(weight="weight"))
        betw = nx.betweenness_centrality(G, weight="weight", normalized=False)
        pr = _safe_pagerank(G)
        return pd.DataFrame(
            {
                "node": nodes,
                "Listener": [in_deg.get(n, 0.0) for n in nodes],
                "Influencer": [out_deg.get(n, 0.0) for n in nodes],
                "Mediator": [betw.get(n, 0.0) for n in nodes],
                "Pagerank": [pr.get(n, 0.0) for n in nodes],
            }
        )

    if G_other is None:
        raise ValueError("G_other (control) required for comparative ranking")
    if G_diff is None:
        raise ValueError("G_diff (signed differential network) required for comparative ranking")

    # Sorted union -> deterministic.
    all_nodes = sorted(set(G.nodes()) | set(G_other.nodes()) | set(G_diff.nodes()))

    # Listener/Influencer from signed weighted degree of the differential network.
    in_diff = dict(G_diff.in_degree(weight="weight")) if G_diff.number_of_edges() else {}
    out_diff = dict(G_diff.out_degree(weight="weight")) if G_diff.number_of_edges() else {}
    listener = {n: in_diff.get(n, 0.0) for n in all_nodes}
    influencer = {n: out_diff.get(n, 0.0) for n in all_nodes}

    # Mediator from betweenness difference (case - control).
    med_case = (
        nx.betweenness_centrality(G, weight="weight", normalized=False)
        if G.number_of_edges()
        else {}
    )
    med_ctrl = (
        nx.betweenness_centrality(G_other, weight="weight", normalized=False)
        if G_other.number_of_edges()
        else {}
    )
    mediator = {n: med_case.get(n, 0.0) - med_ctrl.get(n, 0.0) for n in all_nodes}

    # Pagerank from log-ratio.
    pr = _comparative_pagerank(
        _safe_pagerank(G), _safe_pagerank(G_other), all_nodes, alpha=pagerank_alpha
    )

    return pd.DataFrame(
        {
            "node": all_nodes,
            "Listener": [listener[n] for n in all_nodes],
            "Influencer": [influencer[n] for n in all_nodes],
            "Mediator": [mediator[n] for n in all_nodes],
            "Pagerank": [pr[n] for n in all_nodes],
        }
    )


__all__ = [
    "CANONICAL_COLUMNS",
    "liana_to_canonical",
    "build_cci_network",
    "build_gci_network",
    "_safe_pagerank",
    "build_differential_network",
    "compute_topology_ranking",
    "resolve_condition_arms",
]
