"""Tests for pySCENIC regulon network construction (Ricci curvature input)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.gene_regulation.grn.regulon_curvature import (
    build_regulon_graph,
    compute_regulon_curvature_tables,
    load_regulons_csv,
    plot_regulon_curvature,
    plot_regulon_curvature_from_tables,
)

_REAL_SHAPE_CSV = (
    ",,Enrichment,Enrichment,Enrichment,Enrichment,Enrichment,Enrichment,Enrichment,Enrichment\n"
    ",,AUC,NES,MotifSimilarityQvalue,OrthologousIdentity,Annotation,Context,TargetGenes,RankAtMax\n"
    "TF,MotifID,,,,,,,,\n"
    "ALX1,metacluster_9.19,0.13,3.2,5e-07,1.0,anno,ctx,\"[('PEX5', 0.8), ('MACROD1', 0.5)]\",10\n"
    "ALX1,metacluster_9.20,0.11,3.0,6e-07,1.0,anno,ctx,\"[('PEX5', 0.6)]\",12\n"
    "ATF4,metacluster_2.1,0.09,2.8,1e-06,0.9,anno,ctx,\"[('ZNF574', 0.95)]\",8\n"
)


def test_load_regulons_csv_flattens_pyscenics_real_multiindex_shape(tmp_path: Path) -> None:
    """pySCENIC's regulons.csv has a 2-row MultiIndex header (Enrichment/<metric>)
    and a 2-column MultiIndex row index (TF, MotifID) -- a bare pd.read_csv
    leaves TF and MotifID as unnamed data rows instead of real columns, which
    corrupted the n_tfs metric (162 instead of the real 161 on the fibroblast
    cohort: the literal header row "TF" counted as an extra unique value)."""
    csv_path = tmp_path / "regulons.csv"
    csv_path.write_text(_REAL_SHAPE_CSV)

    df = load_regulons_csv(csv_path)

    assert list(df.columns[:3]) == ["TF", "MotifID", "AUC"]
    assert "TargetGenes" in df.columns
    assert set(df["TF"]) == {"ALX1", "ATF4"}
    assert df["TF"].nunique() == 2


def test_load_regulons_csv_output_feeds_build_regulon_graph(tmp_path: Path) -> None:
    """The loader's output must be exactly what build_regulon_graph expects --
    no separate second parse of the same file with different column names."""
    csv_path = tmp_path / "regulons.csv"
    csv_path.write_text(_REAL_SHAPE_CSV)

    df = load_regulons_csv(csv_path)
    G = build_regulon_graph(df, top_k_targets=10)

    assert set(G.successors("ALX1")) == {"PEX5", "MACROD1"}
    assert G["ALX1"]["PEX5"]["weight"] == 0.8


def _regulons_df():
    """Mirrors pySCENIC's real regulons.csv shape after loading: one row per
    (TF, MotifID) with a TargetGenes column holding a string-literal list of
    (gene, importance) tuples. A TF often has multiple motif rows -- its full
    regulon is the union of all of them."""
    return pd.DataFrame(
        {
            "TF": ["TF1", "TF1", "TF2"],
            "MotifID": ["m1", "m2", "m1"],
            "TargetGenes": [
                "[('G1', 0.9), ('G2', 0.5)]",
                "[('G2', 0.7), ('G3', 0.2)]",
                "[('G4', 0.6)]",
            ],
        }
    )


def test_builds_a_directed_tf_to_target_graph():
    """Real, biological directionality: a regulatory edge flows TF -> target,
    never the reverse -- this is the whole point of a GRN over an undirected
    co-expression graph."""
    G = build_regulon_graph(_regulons_df(), top_k_targets=10)

    assert G.has_edge("TF1", "G1")
    assert not G.has_edge("G1", "TF1")


def test_a_tfs_targets_are_unioned_across_its_motif_rows():
    """TF1 has two motif rows (m1, m2); its real regulon is their union, not
    just the first row encountered."""
    G = build_regulon_graph(_regulons_df(), top_k_targets=10)

    assert set(G.successors("TF1")) == {"G1", "G2", "G3"}


def test_a_target_seen_in_multiple_motif_rows_keeps_its_strongest_importance():
    """G2 appears under TF1 with importance 0.5 (motif m1) and 0.7 (motif
    m2) -- the edge weight should be the max seen, not the first or the sum
    (which would double-count the same regulatory relationship)."""
    G = build_regulon_graph(_regulons_df(), top_k_targets=10)

    assert G["TF1"]["G2"]["weight"] == 0.7


def test_caps_to_the_top_k_targets_per_tf_by_importance():
    """A real regulon can have hundreds of targets; Ollivier-Ricci curvature
    is not tractable on that many edges per TF, so only the top_k_targets
    strongest (by importance) targets survive per TF."""
    regulons = pd.DataFrame(
        {
            "TF": ["TF1"],
            "MotifID": ["m1"],
            "TargetGenes": [str([(f"G{i}", 1.0 - i * 0.05) for i in range(10)])],
        }
    )
    G = build_regulon_graph(regulons, top_k_targets=3)

    assert set(G.successors("TF1")) == {"G0", "G1", "G2"}


def test_a_malformed_target_genes_cell_is_skipped_not_fatal():
    """TargetGenes is a free-text string column from an external tool
    (pySCENIC) -- a single malformed row must not crash the whole build."""
    regulons = pd.DataFrame(
        {
            "TF": ["TF1", "TF2"],
            "MotifID": ["m1", "m1"],
            "TargetGenes": ["not a list", "[('G1', 0.5)]"],
        }
    )
    G = build_regulon_graph(regulons, top_k_targets=10)

    assert list(G.successors("TF2")) == ["G1"]
    assert "TF1" not in G.nodes() or list(G.successors("TF1")) == []


def test_returns_empty_graph_for_empty_input():
    G = build_regulon_graph(
        pd.DataFrame(columns=["TF", "MotifID", "TargetGenes"]), top_k_targets=10
    )

    assert G.number_of_nodes() == 0


def test_compute_regulon_curvature_tables_returns_graph_and_both_curvature_frames():
    """Both the existing force-directed figure and the new radial-curvature
    figure need the SAME expensive Ricci computation -- this is the single
    source of truth both callers should share instead of recomputing."""
    G, edge_df, node_df = compute_regulon_curvature_tables(_regulons_df(), top_k_targets=10)

    assert G.number_of_edges() > 0
    assert not edge_df.empty
    assert not node_df.empty
    assert {"source", "target", "ricci_curvature", "weight"}.issubset(edge_df.columns)


def test_compute_regulon_curvature_tables_empty_input_returns_empty_graph_and_frames():
    G, edge_df, node_df = compute_regulon_curvature_tables(
        pd.DataFrame(columns=["TF", "MotifID", "TargetGenes"]), top_k_targets=10
    )

    assert G.number_of_edges() == 0
    assert edge_df.empty
    assert node_df.empty


def test_plot_regulon_curvature_from_tables_renders_without_recomputing(
    tmp_path: Path,
) -> None:
    """A caller (pyscenic_method) computes curvature tables ONCE, then hands
    them to both the hairball renderer and the radial one -- this is the
    hairball half of that contract, taking already-computed tables instead
    of a raw regulons DataFrame."""
    G, edge_df, node_df = compute_regulon_curvature_tables(_regulons_df(), top_k_targets=10)

    figs = plot_regulon_curvature_from_tables(G, edge_df, node_df, tmp_path, plot_top_k=30)

    assert len(figs) > 0
    assert all(p.exists() for p in figs)


def test_plot_regulon_curvature_from_tables_empty_edges_writes_nothing(tmp_path: Path) -> None:
    import networkx as nx

    figs = plot_regulon_curvature_from_tables(
        nx.DiGraph(), pd.DataFrame(), pd.DataFrame(), tmp_path
    )

    assert figs == []
    assert list(tmp_path.iterdir()) == []


def test_plot_regulon_curvature_writes_no_figure_when_graph_is_empty(tmp_path: Path) -> None:
    """No parseable TargetGenes -> empty graph -> nothing to curve; must not raise."""
    regulons = pd.DataFrame(columns=["TF", "MotifID", "TargetGenes"])

    figs = plot_regulon_curvature(regulons, tmp_path)

    assert figs == []
    assert list(tmp_path.iterdir()) == []


def test_plot_regulon_curvature_renders_real_ollivier_ricci_curvature(tmp_path: Path) -> None:
    """A real TF->target regulon graph must produce a saved figure whose edges
    carry genuine Ollivier-Ricci curvature (computed, not stubbed)."""
    figs = plot_regulon_curvature(_regulons_df(), tmp_path, top_k_targets=10)

    assert len(figs) > 0
    assert all(p.exists() for p in figs)


def test_plot_regulon_curvature_caps_edges_drawn_to_stay_legible(
    tmp_path: Path, monkeypatch
) -> None:
    """A real regulon graph can have thousands of TF->target edges (161 TFs x
    up to 15 targets on the fibroblast cohort); drawing all of them produced
    an unreadable label-smeared hairball. The figure must cap how many edges
    it actually draws -- not just how many the underlying graph has."""
    import cellquorum.stages.gene_regulation.grn.regulon_curvature as mod

    seen_top_k = {}
    real_curvature_network = mod.curvature_network

    def spy(edges, nodes, **kwargs):
        seen_top_k["top_k"] = kwargs.get("top_k")
        return real_curvature_network(edges, nodes, **kwargs)

    monkeypatch.setattr(mod, "curvature_network", spy)

    # 50 TFs x 10 targets each = 500 edges -- far more than a legible plot.
    regulons = pd.DataFrame(
        {
            "TF": [f"TF{i}" for i in range(50)],
            "MotifID": ["m1"] * 50,
            "TargetGenes": [
                str([(f"G{i}_{j}", 1.0 - j * 0.05) for j in range(10)]) for i in range(50)
            ],
        }
    )

    plot_regulon_curvature(regulons, tmp_path, top_k_targets=10)

    assert seen_top_k["top_k"] is not None
    assert seen_top_k["top_k"] <= 40


def test_plot_regulon_curvature_actually_caps_labeled_nodes_not_just_the_kwarg(
    tmp_path: Path, monkeypatch
) -> None:
    """The test above only proved top_k was PASSED to curvature_network --

    not that curvature_network's OWN node-seat guarantee actually honors it.
    On this exact 50-TF x 10-unique-targets-each shape (500 distinct target
    genes), curvature_network's shared _guarantee_node_seats helper's "give
    every node at least one edge" step produces a guaranteed set of ~550
    rows before it ever compares against top_k, hits
    `len(guaranteed) >= top_k` immediately, and returns the WHOLE guaranteed
    set -- confirmed directly by calling curvature_network on this shape:
    550 labels, not ~30. That helper's unconditional-guarantee behavior is
    correct and separately tested for its original use case (a handful of
    CCC clusters, where dropping one is a real loss), so the fix belongs in
    the regulon-specific caller: pre-trim to the globally strongest
    plot_top_k edges before curvature_network ever sees the graph.
    """
    import cellquorum.stages.gene_regulation.grn.regulon_curvature as mod

    regulons = pd.DataFrame(
        {
            "TF": [f"TF{i}" for i in range(50)],
            "MotifID": ["m1"] * 50,
            "TargetGenes": [
                str([(f"G{i}_{j}", 1.0 - j * 0.05) for j in range(10)]) for i in range(50)
            ],
        }
    )
    captured = {}
    real_curvature_network = mod.curvature_network

    def spy(edges, nodes, **kwargs):
        captured["n_edges"] = len(edges)
        return real_curvature_network(edges, nodes, **kwargs)

    monkeypatch.setattr(mod, "curvature_network", spy)

    figs = plot_regulon_curvature(regulons, tmp_path, top_k_targets=10, plot_top_k=30)
    assert figs
    assert captured["n_edges"] <= 30, (
        f"curvature_network received {captured['n_edges']} edges, "
        "not the pre-trimmed ~30 the hairball fix relies on"
    )
