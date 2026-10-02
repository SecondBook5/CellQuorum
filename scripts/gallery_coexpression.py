"""Reference-gallery fixtures for the coexpression (hdWGCNA) figures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from _gallery_io import via_files
from matplotlib.figure import Figure

from cellquorum.stages.gene_regulation.coexpression import (
    community_network,
    module_curvature,
    module_umap_plot,
)
from cellquorum.visualization.figstyle import set_style

MODULES = ["M1", "M2", "M3"]


def _modules() -> pd.DataFrame:
    """hdWGCNA modules.csv shape: gene, module, then one kME_<module> column each."""
    rng = np.random.default_rng(0)
    rows = []
    for module in MODULES:
        for i in range(12):
            row = {"gene": f"{module}_G{i:02d}", "module": module}
            for other in MODULES:
                row[f"kME_{other}"] = float(
                    rng.uniform(0.6, 0.95) if other == module else rng.uniform(0, 0.3)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _tom_edges(modules: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for _, members in modules.groupby("module"):
        genes = members["gene"].tolist()
        for i, a in enumerate(genes):
            for b in genes[i + 1 :]:
                if rng.uniform() < 0.45:
                    rows.append({"gene1": a, "gene2": b, "weight": float(rng.uniform(0.2, 1.0))})
    # a few weak cross-module edges
    rows.append({"gene1": "M1_G00", "gene2": "M2_G00", "weight": 0.25})
    rows.append({"gene1": "M2_G01", "gene2": "M3_G01", "weight": 0.2})
    return pd.DataFrame(rows)


def community_network_figure() -> Figure:
    modules = _modules()
    marker = pd.DataFrame(
        [
            {
                "gene": gene,
                "group": group,
                "log_dor": 3.0 if module == target else -1.0,
            }
            for gene, module in zip(modules["gene"], modules["module"], strict=True)
            for group, target in zip(["Fibroblast", "LEC", "Pericyte"], MODULES, strict=True)
        ]
    )
    return community_network.plot_community_network(_tom_edges(modules), modules, marker)


def module_curvature_figure() -> Path:
    modules = _modules()
    return via_files(
        lambda out: module_curvature.plot_module_curvature(modules, out, top_k_per_module=8),
        pick="_M1",
    )()


def module_umap() -> Path:
    modules = _modules()
    rng = np.random.default_rng(2)
    centers = {"M1": (-2, 0), "M2": (2, 1), "M3": (0, -2)}
    colors = {"M1": "#4C72B0", "M2": "#DD8452", "M3": "#55A868"}
    frame = pd.DataFrame(
        {
            "gene": modules["gene"],
            "UMAP1": [rng.normal(centers[m][0], 0.6) for m in modules["module"]],
            "UMAP2": [rng.normal(centers[m][1], 0.6) for m in modules["module"]],
            "module": modules["module"],
            "color": [colors[m] for m in modules["module"]],
            "hub": np.where(np.arange(len(modules)) % 6 == 0, "hub", "other"),
            "kME": rng.uniform(0.3, 0.95, len(modules)),
        }
    )

    def write(out: Path) -> list[Path]:
        csv = out / "module_umap.csv"
        frame.to_csv(csv, index=False)
        return module_umap_plot.plot_module_umap(csv, out, tag="synthetic")

    return via_files(write)()


THEME = set_style

FIGURES: dict[str, Callable[[], Figure | Path]] = {
    "coexpression_community_network": community_network_figure,
    "coexpression_module_curvature": module_curvature_figure,
    "coexpression_module_umap": module_umap,
}
