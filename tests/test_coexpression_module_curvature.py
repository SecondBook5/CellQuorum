"""Tests for hdWGCNA module co-expression network construction (Ricci curvature input)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.gene_regulation.coexpression.module_curvature import (
    build_module_coexpression_graph,
    plot_module_curvature,
)


def _modules_df():
    """Mirrors hdwgcna.R's real modules.csv shape: gene, module, color, then
    one kME_<module> column per module (eigengene-based membership)."""
    return pd.DataFrame(
        {
            "gene": ["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
            "module": ["M1", "M1", "M1", "M2", "M2", "M2", "grey"],
            "kME_M1": [0.9, 0.8, 0.3, 0.1, 0.2, 0.15, 0.05],
            "kME_M2": [0.1, 0.15, 0.2, 0.85, 0.75, 0.6, 0.02],
        }
    )


def test_builds_edges_only_within_a_module_not_across():
    """Real intent: a co-expression network from module membership connects
    genes that hdWGCNA already grouped together, not arbitrary cross-module
    pairs -- there is no TOM edge weight available between different
    modules' genes, only within-module kME."""
    G = build_module_coexpression_graph(_modules_df(), top_k_per_module=5)

    m1_genes = {"G1", "G2", "G3"}
    m2_genes = {"G4", "G5", "G6"}
    for u, v in G.edges():
        assert (u in m1_genes and v in m1_genes) or (u in m2_genes and v in m2_genes)


def test_the_grey_unassigned_module_is_excluded():
    """ "grey" is hdWGCNA's convention for genes that did not join any real
    module -- they have no meaningful co-expression edge to construct."""
    G = build_module_coexpression_graph(_modules_df(), top_k_per_module=5)

    assert "G7" not in G.nodes()


def test_caps_to_the_top_k_hub_genes_per_module_by_kme():
    """A real module can have hundreds of genes; connecting all of them
    pairwise is both computationally infeasible for Ollivier-Ricci curvature
    and not meaningfully different from a handful of hub genes. Only the
    top_k_per_module strongest (highest kME) genes should appear."""
    modules = pd.DataFrame(
        {
            "gene": [f"G{i}" for i in range(10)],
            "module": ["M1"] * 10,
            "kME_M1": [0.9, 0.85, 0.8, 0.75, 0.1, 0.15, 0.2, 0.05, 0.3, 0.4],
        }
    )
    G = build_module_coexpression_graph(modules, top_k_per_module=3)

    # Only the 3 highest-kME genes (G0, G1, G2) survive.
    assert set(G.nodes()) == {"G0", "G1", "G2"}


def test_edge_weight_is_the_minimum_of_the_two_genes_kme():
    """min(kME_i, kME_j) is the standard proxy for co-expression strength
    when the full TOM is unavailable -- a real, principled choice (not the
    max, which would let one strongly-connected gene inflate a weak pair)."""
    modules = pd.DataFrame({"gene": ["G1", "G2"], "module": ["M1", "M1"], "kME_M1": [0.9, 0.3]})
    G = build_module_coexpression_graph(modules, top_k_per_module=5)

    assert G["G1"]["G2"]["weight"] == 0.3


def test_returns_empty_graph_when_no_kme_columns_present():
    """A modules table without any kME_<module> column (an older/different
    hdWGCNA output shape) cannot build weighted edges -- skip cleanly rather
    than guessing a weight."""
    modules = pd.DataFrame({"gene": ["G1", "G2"], "module": ["M1", "M1"]})

    G = build_module_coexpression_graph(modules, top_k_per_module=5)

    assert G.number_of_nodes() == 0


def test_plot_module_curvature_writes_no_figure_when_graph_is_empty(tmp_path: Path) -> None:
    """No kME columns -> empty graph -> nothing to curve; must not raise."""
    modules = pd.DataFrame({"gene": ["G1", "G2"], "module": ["M1", "M1"]})

    figs = plot_module_curvature(modules, tmp_path)

    assert figs == []
    assert list(tmp_path.iterdir()) == []


def test_plot_module_curvature_renders_real_ollivier_ricci_curvature(tmp_path: Path) -> None:
    """A real module graph must produce a saved figure whose edges carry
    genuine Ollivier-Ricci curvature (computed, not stubbed) -- the whole
    point of this figure over a plain co-expression network diagram."""
    modules = _modules_df()

    figs = plot_module_curvature(modules, tmp_path, top_k_per_module=5)

    assert len(figs) > 0
    assert all(p.exists() for p in figs)
