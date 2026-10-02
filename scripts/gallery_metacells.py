"""Reference-gallery fixtures for the metacell figures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from _gallery_io import via_files
from matplotlib.figure import Figure

from cellquorum.stages.metacells.viz import (
    composition,
    heatmap,
    marker_traces,
    network,
    qc_scatter,
)
from cellquorum.visualization.figstyle import set_style

GROUPS = ["Fibroblast", "Endothelial", "Pericyte", "Keratinocyte"]


def _nodes() -> pd.DataFrame:
    """Twelve metacells, three per dominant cell type."""
    return pd.DataFrame(
        {
            "metacell_id": [f"MC{i:02d}" for i in range(12)],
            "dominant_group": np.repeat(GROUPS, 3),
            "n_cells": [80, 95, 60, 70, 88, 64, 75, 90, 55, 82, 68, 77],
        }
    )


def _assignments() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for _, node in _nodes().iterrows():
        for i in range(node["n_cells"]):
            impure = rng.uniform() < 0.08
            rows.append(
                {
                    "cell_id": f"{node['metacell_id']}_{i}",
                    "metacell_id": node["metacell_id"],
                    "group": rng.choice(GROUPS) if impure else node["dominant_group"],
                    "n_umis": int(rng.normal(1000, 150)),
                }
            )
    for i in range(14):
        rows.append(
            {
                "cell_id": f"out_{i}",
                "metacell_id": "Outliers",
                "group": GROUPS[i % 4],
                "n_umis": int(rng.normal(250, 40)),
            }
        )
    return pd.DataFrame(rows)


def _expression() -> pd.DataFrame:
    """60 genes x 12 metacells; each gene prefers one cell type."""
    rng = np.random.default_rng(1)
    nodes = _nodes()
    genes = [f"Gene{i:02d}" for i in range(58)] + ["PECAM1", "PDGFRB"]
    preferred = [GROUPS[i % 4] for i in range(60)]
    data = rng.normal(0.3, 0.15, (60, 12))
    for g, group in enumerate(preferred):
        data[g, (nodes["dominant_group"] == group).to_numpy()] += 1.2
    return pd.DataFrame(data, index=genes, columns=nodes["metacell_id"])


def composition_bar() -> Path:
    return via_files(
        lambda out: composition.plot_metacell_composition_bar(_assignments(), _nodes(), out)
    )()


def gene_heatmap() -> Path:
    return via_files(
        lambda out: heatmap.plot_metacell_gene_heatmap(_expression(), _nodes(), out, n_genes=40)
    )()


def marker_trace() -> Path:
    return via_files(
        lambda out: marker_traces.plot_metacell_marker_traces(
            _expression(), _nodes(), out, ["PECAM1"]
        )
    )()


def similarity_network() -> Path:
    rng = np.random.default_rng(2)
    ids = _nodes()["metacell_id"].tolist()
    edges = pd.DataFrame(
        [
            {"source": a, "target": b, "weight": float(rng.uniform(0.3, 1.0))}
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
            if a[2:] != b[2:] and (int(a[2:]) // 3 == int(b[2:]) // 3 or rng.uniform() < 0.08)
        ]
    )
    return via_files(lambda out: network.plot_metacell_network(edges, _nodes(), out))()


def qc_unclassified_vs_size() -> Path:
    return via_files(
        lambda out: qc_scatter.plot_metacell_qc_unclassified_vs_size(_assignments(), _nodes(), out)
    )()


def qc_size_comparison() -> Path:
    return via_files(
        lambda out: qc_scatter.plot_metacell_qc_size_comparison(_assignments(), _nodes(), out)
    )()


THEME = set_style

FIGURES: dict[str, Callable[[], Figure | Path]] = {
    "metacells_composition_bar": composition_bar,
    "metacells_gene_heatmap": gene_heatmap,
    "metacells_marker_trace": marker_trace,
    "metacells_similarity_network": similarity_network,
    "metacells_qc_unclassified_vs_size": qc_unclassified_vs_size,
    "metacells_qc_size_comparison": qc_size_comparison,
}
