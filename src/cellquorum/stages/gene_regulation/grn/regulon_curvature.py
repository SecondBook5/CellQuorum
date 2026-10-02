"""Build a TF -> target regulatory graph from pySCENIC regulons, for Ricci
curvature analysis.

The pruned, motif-validated regulon table (not the raw ~2.5M-row GRNBoost2
adjacency output, which carries every candidate TF-target pair before any
biological filtering) is the correct source for a curvature graph: it is
what pySCENIC itself considers a real regulatory relationship. A TF can
appear on more than one row (one per supporting motif); its real regulon is
the union of all of them, each target's edge weight the strongest importance
seen across those rows.
"""

from __future__ import annotations

import ast
from pathlib import Path

import networkx as nx
import pandas as pd

from cellquorum.stages.cell_cell_communication.network.ricci_method import (
    compute_ricci_curvature,
)
from cellquorum.stages.cell_cell_communication.viz._plots import curvature_network
from cellquorum.visualization import figstyle


def load_regulons_csv(path: str | Path) -> pd.DataFrame:
    """Load pySCENIC's real regulons.csv shape into flat, named columns.

    pySCENIC writes a 2-row column MultiIndex (('Enrichment', 'AUC'), ...)
    over a 2-level row index (TF, MotifID). A bare ``pd.read_csv`` reads TF
    and MotifID as unnamed data rows rather than real columns -- every
    downstream ``TF``/``TargetGenes`` lookup on that frame would silently
    return the wrong thing (confirmed: it inflated n_tfs on the fibroblast
    cohort from the true 161 to 162, one column-0 value came from the
    embedded header text "TF" itself).
    """
    df = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    df.columns = df.columns.get_level_values(1)
    return df.reset_index()


def _parse_target_genes(cell: object) -> list[tuple[str, float]]:
    """Parse one TargetGenes cell into [(gene, importance), ...].

    TargetGenes is free text written by an external tool (pySCENIC); a
    malformed cell returns an empty list rather than raising, so one bad row
    never takes down the whole graph build.
    """
    try:
        parsed = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list | tuple):
        return []
    out = []
    for item in parsed:
        if isinstance(item, list | tuple) and len(item) == 2:
            try:
                out.append((str(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return out


def build_regulon_graph(
    regulons: pd.DataFrame,
    *,
    tf_col: str = "TF",
    target_genes_col: str = "TargetGenes",
    top_k_targets: int = 15,
) -> nx.DiGraph:
    """Build a directed TF -> target graph from a pySCENIC regulons table.

    Args:
        regulons: One row per (TF, motif); must carry tf_col and
            target_genes_col (a string-literal list of (gene, importance)
            tuples, pySCENIC's own regulons.csv shape).
        tf_col: Column naming the transcription factor.
        target_genes_col: Column holding the stringified target-gene list.
        top_k_targets: Cap on targets kept per TF, ranked by the strongest
            importance seen for that target across all of the TF's motif
            rows, descending.

    Returns:
        A directed nx.DiGraph, edges TF -> target weighted by importance
        (max across motif rows for that TF-target pair). Empty when the
        input is empty or every row is unparseable.
    """
    if regulons is None or regulons.empty:
        return nx.DiGraph()

    # TF -> {target: max importance seen}, unioned across motif rows.
    per_tf: dict[str, dict[str, float]] = {}
    for _, row in regulons.iterrows():
        tf = str(row[tf_col])
        for gene, importance in _parse_target_genes(row[target_genes_col]):
            targets = per_tf.setdefault(tf, {})
            if gene not in targets or importance > targets[gene]:
                targets[gene] = importance

    G = nx.DiGraph()
    for tf, targets in per_tf.items():
        if not targets:
            continue
        ranked = sorted(targets.items(), key=lambda kv: kv[1], reverse=True)[:top_k_targets]
        for gene, importance in ranked:
            G.add_edge(tf, gene, weight=importance)

    return G


def compute_regulon_curvature_tables(
    regulons: pd.DataFrame,
    *,
    tf_col: str = "TF",
    target_genes_col: str = "TargetGenes",
    top_k_targets: int = 15,
    ricci_alpha: float = 0.5,
) -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    """Build the regulon graph and its Ollivier-Ricci curvature once.

    The single source of truth for both the force-directed curvature
    hairball (``plot_regulon_curvature``) and the radial-layout curvature
    figure (``radial_figures.radial_regulon_curvature``) -- Ricci curvature
    is the expensive part, so a caller rendering both figures should call
    this once and pass the results to each, not recompute per figure.

    Returns:
        ``(G, edge_df, node_df)`` -- the directed TF->target graph (needed
        by the radial figure to re-derive real edge orientation, since
        ``compute_ricci_curvature`` runs on an undirected copy) plus its
        edge- and node-level curvature tables. Empty graph/frames when the
        regulon table has no parseable edges.
    """
    G = build_regulon_graph(
        regulons, tf_col=tf_col, target_genes_col=target_genes_col, top_k_targets=top_k_targets
    )
    if G.number_of_edges() == 0:
        return G, pd.DataFrame(), pd.DataFrame()
    edge_df, node_df = compute_ricci_curvature(G, ricci_alpha)
    return G, edge_df, node_df


def plot_regulon_curvature(
    regulons: pd.DataFrame,
    out_dir: str | Path,
    *,
    tf_col: str = "TF",
    target_genes_col: str = "TargetGenes",
    top_k_targets: int = 15,
    ricci_alpha: float = 0.5,
    plot_top_k: int = 30,
) -> list[Path]:
    """Render an Ollivier-Ricci curvature network of the TF->target regulon graph.

    Negative-curvature edges mark TF-target relationships that bridge
    otherwise disjoint regulatory neighborhoods; positive curvature marks
    redundant, tightly co-regulated cores. Reuses the same curvature
    algorithm and plotting function as the CCC network curvature panel
    (cell_cell_communication.network.ricci_method /
    cell_cell_communication.viz._plots.curvature_network).

    A real regulon graph is unlike co-expression modules: TFs legitimately
    share targets, so the graph stays one connected network rather than
    splitting into disjoint per-TF figures. But the full graph (161 TFs x up
    to top_k_targets each on the fibroblast cohort, ~2000 edges) is an
    unreadable label-smeared hairball if drawn whole -- ``plot_top_k`` caps
    how many of the strongest edges are actually drawn, matching the same
    cap curvature_network's own CCC network caller already uses.

    Returns:
        List of written figure paths (PNG+PDF), or [] when the regulon graph
        has no edges.
    """
    _G, edge_df, node_df = compute_regulon_curvature_tables(
        regulons,
        tf_col=tf_col,
        target_genes_col=target_genes_col,
        top_k_targets=top_k_targets,
        ricci_alpha=ricci_alpha,
    )
    return plot_regulon_curvature_from_tables(_G, edge_df, node_df, out_dir, plot_top_k=plot_top_k)


def plot_regulon_curvature_from_tables(
    G: nx.DiGraph,
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    plot_top_k: int = 30,
) -> list[Path]:
    """Render the force-directed curvature hairball from already-computed tables.

    Split out of :func:`plot_regulon_curvature` so a caller rendering BOTH
    this figure and the radial-layout curvature figure
    (``radial_figures.radial_regulon_curvature``) can call
    :func:`compute_regulon_curvature_tables` once and pass the same tables
    to each, instead of paying for Ollivier-Ricci twice.

    Returns:
        List of written figure paths (PNG+PDF), or [] when ``edge_df`` is
        empty.
    """
    if edge_df is None or edge_df.empty:
        return []

    # curvature_network's own top_k relies on _guarantee_node_seats, whose
    # "give every node at least one edge" step is correct for its original
    # use case (a handful of CCC clusters -- dropping one is a real loss) but
    # breaks down here: a regulon graph has one distinct node per TARGET
    # GENE, not per cluster, so "every node gets a seat" can mean hundreds of
    # nodes even after the guarantee logic runs, defeating plot_top_k
    # entirely (confirmed directly: a 500-target-gene graph produced a
    # 550-node hairball instead of the requested ~30). Pre-trimming to the
    # globally strongest plot_top_k edges here, before curvature_network
    # ever sees the graph, keeps that shared helper's tested CCC contract
    # untouched while still bounding what a regulon run can produce.
    edge_df = edge_df.nlargest(plot_top_k, "weight")

    figstyle.apply_theme("compact")
    out_dir = Path(out_dir)
    fig = curvature_network(edge_df, node_df, top_k=plot_top_k)
    fig.suptitle("Regulon curvature (Ollivier-Ricci)", fontsize=12, fontweight="bold")
    return figstyle.save_figure(fig, out_dir, "regulon_curvature_network")


__all__ = [
    "build_regulon_graph",
    "compute_regulon_curvature_tables",
    "load_regulons_csv",
    "plot_regulon_curvature",
    "plot_regulon_curvature_from_tables",
]
