from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from cellquorum.ccc_network._networks import (
    _safe_pagerank,
    build_cci_network,
    build_differential_network,
    build_gci_network,
    liana_to_canonical,
)
from cellquorum.ccc_network.config import CCCNetworkConfig


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
    assert cfg.seed == 42


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
