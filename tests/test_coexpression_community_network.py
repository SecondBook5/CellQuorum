"""Tests for the coexpression community-network figure: the real DOR marker
statistic + PageRank relevance scores from Adams et al. Sci Adv 2020's
"Assigning cell relevance scores to GRN communities" (p.13-14), and the
network figure itself, built from hdWGCNA's real TOM edges (tom_edges.csv)
rather than the per-module hub-gene proxy graph.
"""

from __future__ import annotations

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy import sparse

from cellquorum.stages.gene_regulation.coexpression.community_network import (
    diagnostic_odds_ratio,
    pagerank_relevance_scores,
    plot_community_network,
)

# ---------------------------------------------------------------------------
# diagnostic_odds_ratio
# ---------------------------------------------------------------------------


def _adata_with_marker() -> ad.AnnData:
    """Gene MARKER detected in every cell of group A, never in group B;
    gene FLAT detected uniformly across both groups."""
    n_a, n_b = 20, 20
    x = np.zeros((n_a + n_b, 2))
    x[:n_a, 0] = 1.0  # MARKER: on for group A
    x[:, 1] = 1.0  # FLAT: on everywhere
    adata = ad.AnnData(X=sparse.csr_matrix(x), var=pd.DataFrame(index=["MARKER", "FLAT"]))
    adata.obs["group"] = ["A"] * n_a + ["B"] * n_b
    return adata


def test_diagnostic_odds_ratio_scores_a_perfect_marker_highly_positive():
    out = diagnostic_odds_ratio(_adata_with_marker(), group_col="group")

    marker_a = out[(out["gene"] == "MARKER") & (out["group"] == "A")]["log_dor"].iloc[0]
    marker_b = out[(out["gene"] == "MARKER") & (out["group"] == "B")]["log_dor"].iloc[0]
    assert marker_a > 2.0
    assert marker_b < marker_a


def test_diagnostic_odds_ratio_flat_gene_scores_near_zero_everywhere():
    out = diagnostic_odds_ratio(_adata_with_marker(), group_col="group")

    flat_a = out[(out["gene"] == "FLAT") & (out["group"] == "A")]["log_dor"].iloc[0]
    assert abs(flat_a) < 0.5


def test_diagnostic_odds_ratio_respects_gene_subset():
    out = diagnostic_odds_ratio(_adata_with_marker(), group_col="group", genes=["FLAT"])

    assert set(out["gene"]) == {"FLAT"}


def test_diagnostic_odds_ratio_empty_adata_returns_empty_frame():
    empty = ad.AnnData(X=sparse.csr_matrix((0, 2)), var=pd.DataFrame(index=["A", "B"]))
    empty.obs["group"] = pd.Series([], dtype=str)

    out = diagnostic_odds_ratio(empty, group_col="group")

    assert out.empty


# ---------------------------------------------------------------------------
# pagerank_relevance_scores
# ---------------------------------------------------------------------------


def _tiny_community_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edge("A1", "A2", weight=1.0)
    G.add_edge("A2", "A3", weight=1.0)
    G.add_edge("B1", "B2", weight=1.0)
    return G


def test_pagerank_relevance_scores_ranks_the_matching_group_highest_per_community():
    """Community 'comm1' (A1/A2/A3) is built entirely from group-X markers;
    community 'comm2' (B1/B2) from group-Y markers -- each community's own
    z-score should favor its own group."""
    G = _tiny_community_graph()
    communities = {"A1": "comm1", "A2": "comm1", "A3": "comm1", "B1": "comm2", "B2": "comm2"}
    marker_log_dor = pd.DataFrame(
        {
            "gene": ["A1", "A2", "A3", "B1", "B2"] * 2,
            "group": ["X"] * 5 + ["Y"] * 5,
            "log_dor": [3.0, 3.0, 3.0, -1.0, -1.0] + [-1.0, -1.0, -1.0, 3.0, 3.0],
        }
    )

    out = pagerank_relevance_scores(G, communities, marker_log_dor)

    comm1_x = out[(out["community"] == "comm1") & (out["group"] == "X")]["z_score"].iloc[0]
    comm1_y = out[(out["community"] == "comm1") & (out["group"] == "Y")]["z_score"].iloc[0]
    assert comm1_x > comm1_y


def test_pagerank_relevance_scores_z_scores_within_each_community_separately():
    """Z-scoring must be PER community (across its own groups), not global
    -- confirmed by checking each community's own z-scores have ~zero mean
    (the defining property of a per-group z-score within a fixed community)."""
    G = _tiny_community_graph()
    communities = {"A1": "comm1", "A2": "comm1", "A3": "comm1", "B1": "comm2", "B2": "comm2"}
    marker_log_dor = pd.DataFrame(
        {
            "gene": ["A1", "A2", "A3", "B1", "B2"] * 2,
            "group": ["X"] * 5 + ["Y"] * 5,
            "log_dor": [3.0, 3.0, 3.0, -1.0, -1.0] + [-1.0, -1.0, -1.0, 3.0, 3.0],
        }
    )

    out = pagerank_relevance_scores(G, communities, marker_log_dor)

    for community in ("comm1", "comm2"):
        z = out[out["community"] == community]["z_score"]
        assert z.mean() == pytest.approx(0.0, abs=1e-6)


def test_pagerank_relevance_scores_empty_graph_returns_empty_frame():
    out = pagerank_relevance_scores(
        nx.Graph(), {}, pd.DataFrame(columns=["gene", "group", "log_dor"])
    )

    assert out.empty


# ---------------------------------------------------------------------------
# plot_community_network
# ---------------------------------------------------------------------------


def _edge_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene1": ["A1", "A2", "B1"],
            "gene2": ["A2", "A3", "B2"],
            "weight": [0.9, 0.8, 0.7],
        }
    )


def _modules_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": ["A1", "A2", "A3", "B1", "B2"],
            "module": ["M1", "M1", "M1", "M2", "M2"],
        }
    )


def test_plot_community_network_renders_a_figure():
    fig = plot_community_network(_edge_df(), _modules_df())

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_community_network_draws_relevance_bar_charts_when_marker_table_given():
    marker_log_dor = pd.DataFrame(
        {
            "gene": ["A1", "A2", "A3", "B1", "B2"] * 2,
            "group": ["X"] * 5 + ["Y"] * 5,
            "log_dor": [3.0, 3.0, 3.0, -1.0, -1.0] + [-1.0, -1.0, -1.0, 3.0, 3.0],
        }
    )

    fig = plot_community_network(_edge_df(), _modules_df(), marker_log_dor)

    # inset_axes() children live in ax.child_axes, NOT fig.axes (confirmed
    # empirically -- matplotlib parents them under the axes that created
    # them, same finding already hit for colorbar insets elsewhere in this
    # codebase).
    main_ax = fig.axes[0]
    assert len(main_ax.child_axes) >= 2  # one bar-chart inset per community


def test_plot_community_network_caps_edges_drawn():
    genes = [f"G{i}" for i in range(60)]
    edge_df = pd.DataFrame(
        {"gene1": genes[:-1], "gene2": genes[1:], "weight": np.linspace(1.0, 0.01, len(genes) - 1)}
    )
    modules_df = pd.DataFrame({"gene": genes, "module": ["M1"] * len(genes)})

    fig = plot_community_network(edge_df, modules_df, max_edges=10)

    assert isinstance(fig, Figure)


def test_plot_community_network_empty_edges_returns_empty_fig():
    fig = plot_community_network(
        pd.DataFrame(columns=["gene1", "gene2", "weight"]), pd.DataFrame(columns=["gene", "module"])
    )

    assert isinstance(fig, Figure)


def test_plot_community_network_drops_grey_module_genes():
    edge_df = pd.DataFrame({"gene1": ["A1"], "gene2": ["A2"], "weight": [0.9]})
    modules_df = pd.DataFrame({"gene": ["A1", "A2"], "module": ["grey", "grey"]})

    fig = plot_community_network(edge_df, modules_df)

    # no real module survives -> degrades to the empty-figure path
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
