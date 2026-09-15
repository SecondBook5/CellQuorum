"""Tests for pySCENIC regulon network construction (Ricci curvature input)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.gene_regulation.grn.regulon_curvature import (
    build_regulon_graph,
    load_regulons_csv,
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
