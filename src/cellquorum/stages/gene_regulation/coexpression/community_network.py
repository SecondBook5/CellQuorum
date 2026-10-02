"""The community-colored coexpression network figure ("the gorgeous one").

Ports the algorithm behind Adams et al., Sci Adv 2020's gene-correlation
network figure (bigSCale2 -> igraph cluster_fast_greedy -> per-community
cell-type relevance scores) onto CellQuorum's own hdWGCNA output, reusing
hdWGCNA's module assignments as the "community" (no bigSCale2/igraph
dependency needed) and the real topological-overlap-matrix edges
(``tom_edges.csv``, exported by ``hdwgcna.R``) as the coexpression graph --
modules.csv alone carries only per-gene module labels + kME, no gene-gene
edges, so a network built from it alone could only fake within-module
cliques (``module_curvature.py``'s existing proxy graph).

The published figure's per-community bar charts are NOT decoration: each
is a real statistic (PageRank centrality within that community, weighted
by a marker gene's log diagnostic-odds-ratio for a cell-type/condition
group, summed and z-scored across groups). ``diagnostic_odds_ratio`` and
``pagerank_relevance_scores`` reproduce that statistic; see
``docs`` / the coexpression-community-network-spec memory note for the
full derivation and page references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import sparse

from cellquorum.visualization import figstyle

if TYPE_CHECKING:
    import anndata as ad
    from matplotlib.axes import Axes

_GREY = "grey"


# ---------------------------------------------------------------------------
# Diagnostic odds ratio (the paper's marker-gene score, p.11)
# ---------------------------------------------------------------------------


def diagnostic_odds_ratio(
    adata: ad.AnnData,
    *,
    group_col: str,
    layer: str | None = None,
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Per (gene, group) log diagnostic odds ratio.

    DOR = ((TP + 0.5) / (FP + 0.5)) / ((FN + 0.5) / (TN + 0.5))

    where, for gene *i* and group *k*: TP = cells in *k* with detected
    expression (value > 0), FP = cells outside *k* with detected
    expression, FN = cells in *k* with none detected, TN = cells outside
    *k* with none detected. The 0.5 pseudocount avoids undefined values
    for a gene with zero detections in an arm (Adams et al. Sci Adv 2020,
    p.11, "Generation of cell type markers").

    Args:
        adata: AnnData with ``group_col`` in ``.obs``.
        group_col: obs column defining cell-type/condition groups.
        layer: Expression layer to binarize (``.X`` when ``None``).
        genes: Optional gene subset (restricts both the computation and
            the returned rows) -- a community-network figure only needs
            DOR for the genes actually drawn, not the full transcriptome.

    Returns:
        Long-format DataFrame: ``gene``, ``group``, ``log_dor``. Empty
        when ``adata`` has no cells.
    """
    cols = ["gene", "group", "log_dor"]
    if adata is None or adata.n_obs == 0:
        return pd.DataFrame(columns=cols)

    work = adata[:, genes] if genes is not None else adata
    X = work.layers[layer] if layer else work.X
    X = sparse.csr_matrix(X) if not sparse.issparse(X) else X.tocsr()
    detected = X.copy()
    detected.data = np.ones_like(detected.data)

    groups = work.obs[group_col].astype(str)
    levels = sorted(groups.unique())
    indicator = pd.get_dummies(groups)[levels].to_numpy(dtype=float)  # cells x groups
    indicator_sp = sparse.csr_matrix(indicator)

    tp = np.asarray((detected.T @ indicator_sp).todense())  # genes x groups
    group_sizes = indicator.sum(axis=0)  # groups
    # detected is CELLS x genes -- sum over cells (axis=0) for the
    # per-gene total detected count, not axis=1 (which would sum over
    # genes for each cell, the wrong quantity and the wrong shape).
    total_detected = np.asarray(detected.sum(axis=0)).ravel()  # genes
    n_cells = work.n_obs

    fn = group_sizes[np.newaxis, :] - tp
    fp = total_detected[:, np.newaxis] - tp
    tn = (n_cells - group_sizes[np.newaxis, :]) - fp

    dor = ((tp + 0.5) / (fp + 0.5)) / ((fn + 0.5) / (tn + 0.5))
    log_dor = np.log(dor)

    gene_names = np.asarray(work.var_names)
    out = pd.DataFrame(log_dor, index=gene_names, columns=levels)
    out.index.name = "gene"
    return out.reset_index().melt(id_vars="gene", var_name="group", value_name="log_dor")


# ---------------------------------------------------------------------------
# Per-community relevance scores (p.13-14)
# ---------------------------------------------------------------------------


def pagerank_relevance_scores(
    G: nx.Graph,
    communities: dict[str, str],
    marker_log_dor: pd.DataFrame,
) -> pd.DataFrame:
    """Per-community, per-group relevance score.

    For each community, PageRank centrality is computed within that
    community's own subgraph (normalized -- networkx's pagerank already
    sums to 1 over the subgraph), then for each group: sum(pagerank[gene]
    * log_dor[gene, group]) over genes the community and the marker table
    share. The cumulative score is z-scored ACROSS groups WITHIN each
    community (not globally) -- a community's bar chart asks "which of
    these groups does THIS community best explain", not "which community
    is biggest overall".

    Args:
        G: The coexpression graph (nodes = genes).
        communities: gene -> community/module id, restricted to genes
            that should be scored (e.g. real modules, "grey" excluded).
        marker_log_dor: Long-format ``gene``, ``group``, ``log_dor``
            (:func:`diagnostic_odds_ratio`'s output).

    Returns:
        Long-format DataFrame: ``community``, ``group``, ``z_score``.
        Empty when the graph has no edges or no community has any scored
        gene.
    """
    cols = ["community", "group", "z_score"]
    if G is None or G.number_of_nodes() == 0 or marker_log_dor is None or marker_log_dor.empty:
        return pd.DataFrame(columns=cols)

    dor_lookup = marker_log_dor.set_index(["gene", "group"])["log_dor"]
    groups = sorted(marker_log_dor["group"].unique())

    by_community: dict[str, list[str]] = {}
    for gene, community in communities.items():
        if gene in G:
            by_community.setdefault(community, []).append(gene)

    rows: list[dict[str, object]] = []
    for community, genes in by_community.items():
        sub = G.subgraph(genes)
        if sub.number_of_edges() == 0:
            pr = dict.fromkeys(genes, 1.0 / len(genes)) if genes else {}
        else:
            pr = nx.pagerank(sub, weight="weight")
        cumulative = {}
        for group in groups:
            score = 0.0
            for gene in genes:
                score += pr.get(gene, 0.0) * dor_lookup.get((gene, group), 0.0)
            cumulative[group] = score
        vals = np.array(list(cumulative.values()))
        mean, std = vals.mean(), vals.std()
        for group, score in cumulative.items():
            z = (score - mean) / std if std > 0 else 0.0
            rows.append({"community": community, "group": group, "z_score": z})

    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def _select_edges(edges: pd.DataFrame, module_of: dict[str, str], max_edges: int) -> pd.DataFrame:
    """Cap edges to ``max_edges`` without letting one huge module starve the rest.

    A naive ``edges.nlargest(max_edges, "weight")`` silently drops entire
    modules: real coexpression weight scales with module size (more genes
    -> more strong pairs), so on a real hdWGCNA run one module with
    thousands of strong within-module edges filled the ENTIRE budget and
    4 of 8 real modules, plus every genuine cross-module bridge,
    disappeared from the figure -- confirmed directly against the real
    fibroblast run. The same "global top-K starves smaller groups"
    failure mode this codebase already fixed once for a different figure
    (``curvature_network``'s ``_guarantee_node_seats``).

    Every module with at least one edge keeps its own top
    ``max_edges // (3 * n_modules)`` SAME-module edges first (so no module
    can vanish entirely), and cross-module edges get their OWN reserved
    quarter of the budget, ranked by weight WITHIN that reservation --
    not mixed into one global ranking against same-module edges. A real
    cross-module bridge is structurally always weaker than a big module's
    own internal edges (a bridge connects two neighborhoods; a core edge
    reinforces one), so "whatever's left over by raw weight" always loses
    to same-module edges and every bridge vanishes -- confirmed directly:
    fixing only the per-module floor above left every real cross-module
    edge (e.g. M1-M7) out of a real fibroblast render. Remaining budget
    after both reservations goes to the globally strongest leftover
    edges of either kind.
    """
    mod1 = edges["gene1"].astype(str).map(module_of)
    mod2 = edges["gene2"].astype(str).map(module_of)
    same_module = mod1 == mod2
    modules = sorted(set(mod1) | set(mod2))
    n_modules = max(len(modules), 1)
    per_module_floor = max(1, max_edges // (3 * n_modules))
    cross_module_budget = max(1, max_edges // 4)

    guaranteed_parts = []
    for module_id in modules:
        in_module = same_module & (mod1 == module_id)
        guaranteed_parts.append(edges[in_module].nlargest(per_module_floor, "weight"))
    cross = edges[~same_module].nlargest(cross_module_budget, "weight")
    guaranteed_parts.append(cross)
    guaranteed = (
        pd.concat(guaranteed_parts).drop_duplicates() if guaranteed_parts else edges.iloc[0:0]
    )

    remaining_budget = max_edges - len(guaranteed)
    if remaining_budget <= 0:
        return guaranteed.nlargest(max_edges, "weight")
    leftover = edges.drop(guaranteed.index, errors="ignore")
    filler = leftover.nlargest(remaining_budget, "weight")
    return pd.concat([guaranteed, filler])


def plot_community_network(
    edge_df: pd.DataFrame,
    modules_df: pd.DataFrame,
    marker_log_dor: pd.DataFrame | None = None,
    *,
    gene_col: str = "gene",
    module_col: str = "module",
    max_edges: int = 1500,
    seed: int = 0,
    title: str = "Coexpression network",
    module_labels: dict[str, str] | None = None,
) -> Figure:
    """Community-colored, force-directed coexpression network.

    Nodes are genes (colored by hdWGCNA module = community), edges are
    real TOM-thresholded coexpression (``tom_edges.csv``). When
    ``marker_log_dor`` is given, each community gets a small relevance-
    score bar chart positioned at its own centroid on the canvas (not a
    separate legend) -- the real per-community, per-group z-score from
    :func:`pagerank_relevance_scores`, matching the reference figure's
    "predominant community driver" panels.

    Returns:
        The figure, or an explained empty figure when there are no edges
        or no gene survives both the edge list and a real (non-"grey")
        module assignment.
    """
    if edge_df is None or edge_df.empty or modules_df is None or modules_df.empty:
        return figstyle.empty_figure("no coexpression edges")

    modules = modules_df[modules_df[module_col].astype(str) != _GREY]
    if modules.empty:
        return figstyle.empty_figure("no real (non-grey) modules")
    module_of = dict(
        zip(modules[gene_col].astype(str), modules[module_col].astype(str), strict=False)
    )

    edges = edge_df[
        edge_df["gene1"].astype(str).isin(module_of) & edge_df["gene2"].astype(str).isin(module_of)
    ].copy()
    if edges.empty:
        return figstyle.empty_figure("no edges between real-module genes")
    edges = _select_edges(edges, module_of, max_edges)

    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["gene1"]), str(r["gene2"]), weight=float(r["weight"]))
    if G.number_of_edges() == 0:
        return figstyle.empty_figure("no edges to draw")

    communities = {n: module_of[n] for n in G.nodes() if n in module_of}
    community_ids = sorted(set(communities.values()))
    palette = figstyle.get_group_palette(community_ids)

    n_nodes = max(G.number_of_nodes(), 1)
    pos = nx.spring_layout(G, seed=seed, weight="weight", k=2.0 / n_nodes**0.5, iterations=100)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_axis_off()

    # Edges first (under nodes): plain thin lines -- at this scale (up to
    # max_edges, far more than the curvature figures' ~30-edge cap) curved
    # bezier/arrow patches per edge would be both unreadable and slow.
    xs0, ys0, xs1, ys1 = [], [], [], []
    for u, v in G.edges():
        xs0.append(pos[u][0])
        ys0.append(pos[u][1])
        xs1.append(pos[v][0])
        ys1.append(pos[v][1])
    from matplotlib.collections import LineCollection

    segments = [[(x0, y0), (x1, y1)] for x0, y0, x1, y1 in zip(xs0, ys0, xs1, ys1, strict=False)]
    ax.add_collection(LineCollection(segments, colors="0.75", linewidths=0.4, alpha=0.6, zorder=1))

    degree = dict(G.degree())
    max_degree = max(degree.values()) if degree else 1
    for community_id in community_ids:
        nodes = [n for n in G.nodes() if communities.get(n) == community_id]
        if not nodes:
            continue
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        sizes = [12.0 + 28.0 * (degree.get(n, 1) / max_degree) for n in nodes]
        label = (module_labels or {}).get(community_id, community_id)
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=[palette[community_id]],
            edgecolor="white",
            linewidths=0.3,
            zorder=3,
            label=label,
        )

    if marker_log_dor is not None and not marker_log_dor.empty:
        _draw_community_relevance_bars(ax, G, pos, communities, marker_log_dor)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8.5,
        title="module",
        title_fontsize=9,
        markerscale=1.3,
    )
    ax.margins(0.15)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.subplots_adjust(right=0.82)
    return fig


_MAX_BARS = 5


def _draw_community_relevance_bars(
    ax: Axes,
    G: nx.Graph,
    pos: dict[str, tuple[float, float]],
    communities: dict[str, str],
    marker_log_dor: pd.DataFrame,
) -> None:
    """One small bar chart per community, placed at that community's own
    centroid -- real per-group relevance z-scores, not a shared legend.

    Bars are numbered (1, 2, 3, ...) rather than labeled with the group
    name directly on the x-axis: a real group name (a cell type, a
    condition) is often too long to rotate into a legible tick label at
    this scale (confirmed on a real hdWGCNA rerun -- 90-degree rotated
    single-letter synthetic labels already looked cramped). The numbered
    rank is spelled out in a text list beside the bars instead, matching
    the reference figure's own layout; only the top _MAX_BARS groups are
    shown, since the reference typically annotates 3-5 per community and
    a real run can have far more groups than fit legibly.
    """
    scores = pagerank_relevance_scores(G, communities, marker_log_dor)
    if scores.empty:
        return

    for community_id, sub in scores.groupby("community"):
        nodes = [n for n in G.nodes() if communities.get(n) == community_id]
        if not nodes:
            continue
        cx = float(np.mean([pos[n][0] for n in nodes]))
        cy = float(np.mean([pos[n][1] for n in nodes]))
        ranked = sub.sort_values("z_score", ascending=False).head(_MAX_BARS)
        groups = ranked["group"].tolist()
        zvals = ranked["z_score"].to_numpy()
        n_bars = len(groups)

        bar_w, bar_h = 0.14, 0.09
        inset = ax.inset_axes((cx - bar_w / 2, cy + 0.09, bar_w, bar_h), transform=ax.transData)
        inset.bar(range(n_bars), zvals, color="0.4", width=0.7)
        inset.axhline(1.0, color="#C41E3A", linewidth=0.6, linestyle="--")
        inset.set_xticks(range(n_bars))
        inset.set_xticklabels([str(i + 1) for i in range(n_bars)], fontsize=5.5)
        inset.set_yticks([])
        for spine in ("top", "right", "left"):
            inset.spines[spine].set_visible(False)
        inset.patch.set_alpha(0.0)

        # Rank -> group-name text list, drivers (z >= 1, the paper's own
        # threshold) bolded and colored, the rest plain -- above the bars,
        # left-aligned to the inset's own left edge (axes-fraction x=0,
        # data y so it tracks the bar chart's own position).
        label_lines = [
            f"{i + 1}. {g}" + (" *" if z >= 1.0 else "")
            for i, (g, z) in enumerate(zip(groups, zvals, strict=False))
        ]
        inset.text(
            0.0,
            1.05,
            "\n".join(label_lines),
            transform=inset.transAxes,
            fontsize=4.8,
            va="bottom",
            ha="left",
            linespacing=1.3,
        )


__all__ = ["diagnostic_odds_ratio", "pagerank_relevance_scores", "plot_community_network"]
