"""Build a co-expression graph from hdWGCNA module membership, for Ricci
curvature analysis.

hdWGCNA's real output (modules.csv) does not carry the full TOM (topological
overlap matrix) -- only per-gene module assignment and one kME_<module>
column per module (eigengene-based membership strength). There is no
between-module edge weight available at all, and no within-module edge
weight beyond kME, so the graph built here connects each module's own
top-kME "hub" genes to each other, weighted by min(kME_i, kME_j) -- the
standard proxy for co-expression strength when the TOM itself is
unavailable. Capped per module: a real module can have hundreds of genes,
and Ollivier-Ricci curvature (optimal transport per edge) is not tractable
on a fully-connected few-hundred-node graph.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from cellquorum.stages.cell_cell_communication.network.ricci_method import (
    compute_ricci_curvature,
)
from cellquorum.stages.cell_cell_communication.viz._plots import curvature_network
from cellquorum.visualization import figstyle

_GREY = "grey"


def build_module_coexpression_graph(
    modules: pd.DataFrame,
    *,
    gene_col: str = "gene",
    module_col: str = "module",
    top_k_per_module: int = 15,
) -> nx.Graph:
    """Build an undirected, weighted co-expression graph from module membership.

    Args:
        modules: hdWGCNA's modules.csv shape -- gene_col, module_col, and one
            kME_<module value> column per real module.
        gene_col: Column naming each gene.
        module_col: Column naming each gene's assigned module.
        top_k_per_module: Cap on hub genes kept per module, ranked by that
            module's own kME column, descending.

    Returns:
        An undirected nx.Graph. Nodes are gene names; edges exist only
        between genes in the SAME real (non-"grey") module, weighted by
        min(kME_i, kME_j) for that module. Empty when no kME_<module>
        columns are present at all.
    """
    G = nx.Graph()
    if modules is None or modules.empty:
        return G

    for module_id, group in modules.groupby(module_col, observed=True):
        if str(module_id) == _GREY:
            continue
        kme_col = f"kME_{module_id}"
        if kme_col not in group.columns:
            continue
        top = group.nlargest(top_k_per_module, kme_col)
        genes = list(top[gene_col])
        kmes = dict(zip(top[gene_col], top[kme_col], strict=False))
        for gene in genes:
            G.add_node(gene)
        for i, gi in enumerate(genes):
            for gj in genes[i + 1 :]:
                G.add_edge(gi, gj, weight=min(kmes[gi], kmes[gj]))

    return G


def plot_module_curvature(
    modules: pd.DataFrame,
    out_dir: str | Path,
    *,
    gene_col: str = "gene",
    module_col: str = "module",
    top_k_per_module: int = 15,
    ricci_alpha: float = 0.5,
) -> list[Path]:
    """Render an Ollivier-Ricci curvature network of hub-gene co-expression.

    Bridge genes between hub cliques show up as negative-curvature edges;
    tightly redundant hub cores show up as positive curvature -- structure a
    plain co-expression diagram does not surface. Reuses the same curvature
    algorithm and plotting function as the CCC network curvature panel
    (cell_cell_communication.network.ricci_method /
    cell_cell_communication.viz._plots.curvature_network) rather than a
    second implementation.

    Returns:
        List of written figure paths (PNG+PDF), one pair per real module, or
        [] when no module has kME_<module> edges to plot.

    One file per module, not one combined figure: modules never share an
    edge (build_module_coexpression_graph only connects genes within the
    same module), so a single canvas across all of them is N unrelated
    cliques, not one meaningful plot -- on real data (10 modules x 15 hub
    genes) that rendered as an unreadable label smear.
    """
    if modules is None or modules.empty:
        return []

    figstyle.set_style()
    out_dir = Path(out_dir)
    paths: list[Path] = []
    for module_id in sorted(modules[module_col].astype(str).unique()):
        if module_id == _GREY:
            continue
        sub = modules[modules[module_col].astype(str) == module_id]
        G = build_module_coexpression_graph(
            sub, gene_col=gene_col, module_col=module_col, top_k_per_module=top_k_per_module
        )
        if G.number_of_edges() == 0:
            continue
        edge_df, node_df = compute_ricci_curvature(nx.DiGraph(G), ricci_alpha)
        if edge_df.empty:
            continue
        fig = curvature_network(edge_df, node_df)
        fig.suptitle(
            f"Module {module_id} curvature (Ollivier-Ricci)", fontsize=12, fontweight="bold"
        )
        paths += figstyle.save_figure(fig, out_dir, f"module_curvature_{module_id}")
    return paths


__all__ = ["build_module_coexpression_graph", "plot_module_curvature"]
