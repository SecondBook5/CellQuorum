from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from cellquorum.stages.cell_cell_communication.network._networks import (
    _comparative_pagerank,
    _safe_pagerank,
    build_cci_network,
    build_differential_network,
    build_gci_network,
    compute_topology_ranking,
    liana_to_canonical,
    sparsify_top_k_per_node,
)
from cellquorum.stages.cell_cell_communication.network.config import CCCNetworkConfig


def _liana_res() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": ["s1", "s1", "s2"],
            "source": ["A", "B", "A"],
            "target": ["B", "A", "B"],
            "ligand_complex": ["L1", "L2", "L1"],
            "receptor_complex": ["R1", "R2", "R1"],
            "magnitude_rank": [0.1, 0.4, 0.2],
        }
    )


def test_config_defaults():
    cfg = CCCNetworkConfig()
    assert cfg.enabled is True
    assert cfg.source_key == "liana_res"
    assert cfg.gci_max_edges == 200_000
    assert cfg.ricci_alpha == 0.5
    assert cfg.pagerank_alpha == 0.01
    assert cfg.min_edges == 1
    assert cfg.curvature_sparsify_k == 3


def test_liana_to_canonical_maps_and_inverts_weight():
    canon, notes = liana_to_canonical(_liana_res())
    assert list(canon.columns) == ["source", "target", "ligand", "receptor", "weight", "sample"]
    assert len(canon) == 3
    # weight = 1 - magnitude_rank (higher = stronger)
    row = canon[(canon["source"] == "A") & (canon["ligand"] == "L1") & (canon["sample"] == "s1")]
    assert np.isclose(row["weight"].iloc[0], 0.9)


def test_liana_to_canonical_drops_missing_required_columns():
    bad = _liana_res().drop(columns=["magnitude_rank"])
    canon, notes = liana_to_canonical(bad)
    assert canon.empty
    assert any("magnitude_rank" in n for n in notes)


# Task 2: CCI/GCI network builders + safe pagerank


def _canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A", "A", "B"],
            "target": ["B", "B", "A"],
            "ligand": ["L1", "L2", "L3"],
            "receptor": ["R1", "R2", "R3"],
            "weight": [0.9, 0.1, 0.6],
            "sample": ["s1", "s2", "s1"],
        }
    )


def test_build_cci_aggregates_weight_per_pair():
    G = build_cci_network(_canonical())
    assert set(G.nodes()) == {"A", "B"}
    # A->B aggregates 0.9 + 0.1
    assert np.isclose(G["A"]["B"]["weight"], 1.0)
    assert np.isclose(G["B"]["A"]["weight"], 0.6)


def test_build_cci_empty_returns_empty_graph():
    G = build_cci_network(
        pd.DataFrame(columns=["source", "target", "ligand", "receptor", "weight", "sample"])
    )
    assert G.number_of_nodes() == 0


def test_build_gci_nodes_and_edges():
    G = build_gci_network(_canonical())
    assert "A/L1|L" in G.nodes()
    assert "B/R1|R" in G.nodes()
    assert G["A/L1|L"]["B/R1|R"]["ligand"] == "L1"
    assert G.number_of_edges() == 3


def test_safe_pagerank_returns_distribution():
    G = build_cci_network(_canonical())
    pr = _safe_pagerank(G)
    assert set(pr.keys()) == {"A", "B"}
    assert abs(sum(pr.values()) - 1.0) < 1e-6


def test_safe_pagerank_empty_graph():
    assert _safe_pagerank(nx.DiGraph()) == {}


# Task 3: Differential network builder


def test_differential_network_signed_and_nonzero():
    control = pd.DataFrame(
        {
            "source": ["A", "A"],
            "target": ["B", "B"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "weight": [0.5, 0.3],
            "sample": ["c1", "c1"],
        }
    )
    case = pd.DataFrame(
        {
            "source": ["A", "A"],
            "target": ["B", "B"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "weight": [0.9, 0.3],
            "sample": ["d1", "d1"],
        }
    )
    diff_table, diff_cci, diff_gci = build_differential_network(control, case, "ctrl", "case")
    # L1@R1 gained (0.9-0.5=+0.4); L2@R2 unchanged (0.0) -> dropped.
    assert len(diff_table) == 1
    row = diff_table.iloc[0]
    assert row["ligand"] == "L1"
    assert np.isclose(row["weight"], 0.4)
    assert np.isclose(diff_cci["A"]["B"]["weight"], 0.4)


def test_differential_network_handles_arm_only_channels():
    control = pd.DataFrame(
        {
            "source": ["A"],
            "target": ["B"],
            "ligand": ["L1"],
            "receptor": ["R1"],
            "weight": [0.5],
            "sample": ["c1"],
        }
    )
    case = pd.DataFrame(
        {
            "source": ["A"],
            "target": ["B"],
            "ligand": ["L9"],
            "receptor": ["R9"],
            "weight": [0.7],
            "sample": ["d1"],
        }
    )
    diff_table, _, _ = build_differential_network(control, case, "ctrl", "case")
    # L1 lost (-0.5), L9 gained (+0.7)
    weights = dict(zip(diff_table["ligand"], diff_table["weight"], strict=False))
    assert np.isclose(weights["L1"], -0.5)
    assert np.isclose(weights["L9"], 0.7)


# Task 4: Topology ranking + comparative pagerank


def test_topology_single_condition_degrees():
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=2.0)
    G.add_edge("A", "C", weight=1.0)
    df = compute_topology_ranking(G)
    assert list(df.columns) == ["node", "Listener", "Influencer", "Mediator", "Pagerank"]
    a = df[df["node"] == "A"].iloc[0]
    assert np.isclose(a["Influencer"], 3.0)  # out-degree 2+1
    assert np.isclose(a["Listener"], 0.0)
    b = df[df["node"] == "B"].iloc[0]
    assert np.isclose(b["Listener"], 2.0)
    # Deterministic node order (sorted).
    assert list(df["node"]) == ["A", "B", "C"]


def test_topology_empty_graph_returns_empty_frame():
    df = compute_topology_ranking(nx.DiGraph())
    assert list(df.columns) == ["node", "Listener", "Influencer", "Mediator", "Pagerank"]
    assert df.empty


def test_comparative_pagerank_sign():
    # Node enriched in case -> positive log-ratio.
    pr_case = {"A": 0.8, "B": 0.2}
    pr_ctrl = {"A": 0.2, "B": 0.8}
    out = _comparative_pagerank(pr_case, pr_ctrl, ["A", "B"], alpha=0.01)
    assert out["A"] > 0
    assert out["B"] < 0


def test_topology_comparative_runs():
    G_case = nx.DiGraph()
    G_case.add_edge("A", "B", weight=1.0)
    G_ctrl = nx.DiGraph()
    G_ctrl.add_edge("B", "A", weight=1.0)
    # Build a minimal differential graph for the test.
    G_diff = nx.DiGraph()
    G_diff.add_edge("A", "B", weight=0.5)
    G_diff.add_edge("B", "A", weight=-0.5)
    df = compute_topology_ranking(G_case, comparative=True, G_other=G_ctrl, G_diff=G_diff)
    assert list(df["node"]) == ["A", "B"]  # sorted union, deterministic


def test_comparative_degrees_are_signed_difference():
    """FIX 1 regression guard: prove Listener/Influencer are genuine case-control deltas."""
    # Case: A->B with weight=2.0
    # Control: A->B with weight=5.0 (STRONGER in control)
    # Differential: A->B weight = 2.0 - 5.0 = -3.0 (LOST in case)
    G_case = nx.DiGraph()
    G_case.add_edge("A", "B", weight=2.0)
    G_case.add_edge("C", "D", weight=4.0)

    G_ctrl = nx.DiGraph()
    G_ctrl.add_edge("A", "B", weight=5.0)
    G_ctrl.add_edge("C", "D", weight=1.0)

    # Build the signed differential network.
    G_diff = nx.DiGraph()
    G_diff.add_edge("A", "B", weight=-3.0)  # LOST in case
    G_diff.add_edge("C", "D", weight=3.0)  # GAINED in case

    df = compute_topology_ranking(G_case, comparative=True, G_other=G_ctrl, G_diff=G_diff)

    # B is a receiver of A->B; lost in case -> negative Listener.
    b = df[df["node"] == "B"].iloc[0]
    assert b["Listener"] < 0, "Listener for B (receives LOST edge) should be negative"

    # D is a receiver of C->D; gained in case -> positive Listener.
    d = df[df["node"] == "D"].iloc[0]
    assert d["Listener"] > 0, "Listener for D (receives GAINED edge) should be positive"

    # A is a sender of A->B; lost in case -> negative Influencer.
    a = df[df["node"] == "A"].iloc[0]
    assert a["Influencer"] < 0, "Influencer for A (sends LOST edge) should be negative"

    # C is a sender of C->D; gained in case -> positive Influencer.
    c = df[df["node"] == "C"].iloc[0]
    assert c["Influencer"] > 0, "Influencer for C (sends GAINED edge) should be positive"


def _complete_digraph(n: int, weight_fn) -> nx.DiGraph:
    """A complete directed graph on n nodes, weight_fn(u, v) -> float."""
    G = nx.DiGraph()
    nodes = list(range(n))
    G.add_nodes_from(nodes)
    for u in nodes:
        for v in nodes:
            if u != v:
                G.add_edge(u, v, weight=weight_fn(u, v))
    return G


def test_sparsify_keeps_only_top_k_edges_per_node():
    """Ollivier-Ricci curvature is mathematically degenerate on a complete
    graph (every edge gets the identical value, confirmed directly on the real
    9-cluster fibroblast CCI network: 36/36 possible edges, weights ranging
    2061-4347, yet every edge curvature was exactly 0.5625). Real interaction
    networks are near-complete, so curvature needs a sparsified backbone with
    actual topology (bridges, non-neighbors) to have anything to measure."""
    G = _complete_digraph(9, lambda u, v: float(abs(u - v)) + 1.0)

    H = sparsify_top_k_per_node(G, k=3)

    assert set(H.nodes()) == set(G.nodes())
    assert H.number_of_edges() < G.number_of_edges()
    # Every node kept at least one of its own top-k edges (in or out).
    for n in H.nodes():
        assert H.in_degree(n) + H.out_degree(n) > 0


def test_sparsify_keeps_each_nodes_own_strongest_edges():
    """A node's top-k must survive sparsification even if another node's
    top-k doesn't include it -- the guarantee is per-node, not global."""
    G = nx.DiGraph()
    G.add_edge("weak_hub", "strong_partner", weight=100.0)
    for i in range(10):
        G.add_edge("weak_hub", f"other{i}", weight=1.0)
        G.add_edge(f"other{i}", "weak_hub", weight=1.0)
    # strong_partner's only edge is its strongest; must survive at k=1.
    G.add_edge("strong_partner", "weak_hub", weight=100.0)

    H = sparsify_top_k_per_node(G, k=1)

    assert H.has_edge("weak_hub", "strong_partner")


def test_sparsify_produces_meaningfully_varied_curvature_on_a_near_complete_graph():
    """The actual point of sparsifying: curvature computed on the sparsified
    graph must show real variation, unlike the degenerate complete-graph case."""
    from cellquorum.stages.cell_cell_communication.network.ricci_method import (
        compute_ricci_curvature,
    )

    G = _complete_digraph(9, lambda u, v: float(abs(u - v)) * 500.0 + 1000.0)
    complete_edges, _ = compute_ricci_curvature(G, alpha=0.5)
    assert complete_edges["ricci_curvature"].std() < 1e-9  # degenerate, as expected

    H = sparsify_top_k_per_node(G, k=3)
    sparse_edges, _ = compute_ricci_curvature(H, alpha=0.5)
    assert sparse_edges["ricci_curvature"].std() > 1e-6


def test_sparsify_returns_empty_graph_for_empty_input():
    G = nx.DiGraph()
    H = sparsify_top_k_per_node(G, k=3)
    assert H.number_of_edges() == 0
