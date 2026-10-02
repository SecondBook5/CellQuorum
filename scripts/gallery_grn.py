"""Reference-gallery fixtures for the gene regulatory network figures."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from _gallery_io import via_files
from matplotlib.figure import Figure

from cellquorum.stages.gene_regulation.grn import radial_figures, regulon_figures
from cellquorum.visualization.figstyle import apply_theme

TFS = ["SOX9", "FOXC2", "PROX1", "STAT3", "TWIST1", "KLF4"]


def _de_and_net() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Differential expression for 60 genes plus a TF-target network over them."""
    rng = np.random.default_rng(0)
    genes = [f"G{i:02d}" for i in range(60)]
    de = pd.DataFrame(
        {
            "gene": genes,
            "logFC": np.linspace(2.5, -2.5, 60) + rng.normal(0, 0.2, 60),
            "PValue": 0.001,
            "FDR": np.where(np.arange(60) % 4 == 3, 0.4, 0.01),
        }
    )
    rows = []
    for tf in TFS:
        for gene in rng.choice(genes, size=14, replace=False):
            rows.append({"source": tf, "target": gene, "weight": float(rng.uniform(0.3, 1.0))})
    return de, pd.DataFrame(rows)


def radial_grn() -> Figure:
    de, net = _de_and_net()
    tf = radial_figures.tf_activity_from_de(de, net, tmin=3)
    return radial_figures.radial_grn(
        tf,
        de.set_index("gene"),
        net,
        title="Synthetic regulatory network",
        n_tf=6,
        per_tf=7,
        n_sectors=3,
        case="Lymphedema",
        control="Normal",
    )


def radial_regulon_curvature() -> Figure:
    rng = np.random.default_rng(1)
    graph = nx.DiGraph()
    edges = []
    for tf in TFS[:4]:
        for target in rng.choice([f"G{i:02d}" for i in range(30)], size=8, replace=False):
            weight = float(rng.uniform(0.2, 1.0))
            graph.add_edge(tf, target, weight=weight)
            edges.append((tf, target, float(rng.uniform(-0.5, 0.4)), weight))
    edge_df = pd.DataFrame(edges, columns=["source", "target", "ricci_curvature", "weight"])
    node_df = pd.DataFrame(
        {
            "node": list(graph.nodes),
            "ricci_curvature": rng.uniform(-0.4, 0.3, graph.number_of_nodes()),
        }
    )
    return radial_figures.radial_regulon_curvature(
        edge_df, node_df, graph, n_tf=4, per_tf=6, n_sectors=2
    )


def _regulon_activity() -> tuple[pd.DataFrame, pd.Series]:
    """AUCell-like matrix where each of three groups lights up its own regulons."""
    rng = np.random.default_rng(2)
    groups = pd.Series(np.repeat(["0", "1", "2"], 40), index=[f"cell_{i}" for i in range(120)])
    auc = pd.DataFrame(
        rng.random((120, 9)) * 0.3,
        index=groups.index,
        columns=[f"{tf}_(+)" for tf in TFS + ["ETS1", "RUNX1", "MYC"]],
    )
    for g, code in enumerate(["0", "1", "2"]):
        auc.loc[groups == code, auc.columns[g * 3 : g * 3 + 3]] += 0.6
    return auc, groups


def rss_panels() -> Path:
    auc, groups = _regulon_activity()
    return via_files(lambda out: regulon_figures.plot_rss_panels(auc, groups, out, top_n=3))()


def regulon_clustermap() -> Path:
    auc, groups = _regulon_activity()
    return via_files(
        lambda out: regulon_figures.plot_regulon_clustermap(auc, groups, out, top_n=3)
    )()


def regulon_cell_clustermap() -> Path:
    auc, groups = _regulon_activity()
    return via_files(
        lambda out: regulon_figures.plot_regulon_cell_clustermap(
            auc, groups, out, rss_groups=groups, top_n=3
        )
    )()


def regulon_umap() -> Path:
    auc, groups = _regulon_activity()
    rng = np.random.default_rng(3)
    centers = {"0": (-2, 0), "1": (2, 1), "2": (0, -2)}
    umap = pd.DataFrame(
        [rng.normal(centers[g], 0.5) for g in groups], index=auc.index, columns=["UMAP1", "UMAP2"]
    )
    return via_files(
        lambda out: regulon_figures.plot_regulon_umap(auc, umap, out, groups=groups, top_n=4)
    )()


THEME = partial(apply_theme, "compact")

FIGURES: dict[str, Callable[[], Figure | Path]] = {
    "grn_radial": radial_grn,
    "grn_radial_curvature": radial_regulon_curvature,
    "grn_rss_panels": rss_panels,
    "grn_regulon_clustermap": regulon_clustermap,
    "grn_regulon_cell_clustermap": regulon_cell_clustermap,
    "grn_regulon_umap": regulon_umap,
}
