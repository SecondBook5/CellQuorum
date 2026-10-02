"""Reference-gallery fixtures for the embedding figures (UMAP overlays and score maps)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import sparse

from cellquorum.stages.integration.embeddings import plots
from cellquorum.visualization.figstyle import apply_theme

AXES = ("UMAP1", "UMAP2")
CLUSTERS = {"0": "Papillary", "1": "Reticular", "2": "Perivascular", "3": "Mechanosensory"}


def _adata() -> ad.AnnData:
    """Four clusters on a UMAP, two conditions, and a PAGA graph over the clusters."""
    rng = np.random.default_rng(0)
    centers = np.array([[-3.0, 0.0], [3.0, 1.0], [0.0, 3.5], [0.5, -3.0]])
    labels = np.repeat(list(CLUSTERS), 150)
    coords = np.vstack([rng.normal(centers[int(k)], 0.7, (150, 2)) for k in CLUSTERS])
    adata = ad.AnnData(
        X=np.zeros((len(labels), 1)),
        obs=pd.DataFrame(
            {
                "cluster": pd.Categorical(labels, categories=list(CLUSTERS)),
                "condition": np.where(np.arange(len(labels)) % 2 == 0, "Normal", "Lymphedema"),
            },
            index=[f"cell_{i}" for i in range(len(labels))],
        ),
    )
    adata.obsm["X_umap"] = coords
    connectivities = np.array(
        [[0, 0.6, 0.1, 0.0], [0.6, 0, 0.5, 0.1], [0.1, 0.5, 0, 0.3], [0.0, 0.1, 0.3, 0]]
    )
    adata.uns["paga"] = {"connectivities": sparse.csr_matrix(connectivities)}
    return adata


def categorical_umap() -> Figure:
    return plots.categorical_embedding(
        _adata(),
        "cluster",
        basis="X_umap",
        axis_labels=AXES,
        title="Synthetic clusters",
        cluster_labels=CLUSTERS,
    )


def continuous_umap() -> Figure:
    adata = _adata()
    coords = adata.obsm["X_umap"]
    values = np.exp(-np.linalg.norm(coords - np.array([3.0, 1.0]), axis=1) / 2)
    return plots.continuous_overlay(
        coords, values, title="Synthetic module score", axis_labels=AXES, clip_pct=2.0
    )


def score_by_condition() -> Figure:
    adata = _adata()
    coords = adata.obsm["X_umap"]
    score = np.exp(-np.linalg.norm(coords - np.array([0.0, 3.5]), axis=1) / 2)
    score = score + 0.3 * (adata.obs["condition"] == "Lymphedema").to_numpy()
    return plots.plot_score_by_condition(
        adata,
        score=score,
        score_label="fibrosis_ecm (ucell)",
        cluster_key="cluster",
        basis="X_umap",
        axis_labels=AXES,
        condition_col="condition",
        case="Lymphedema",
        control="Normal",
        cluster_labels=CLUSTERS,
    )


THEME = partial(apply_theme, "notebook")

FIGURES: dict[str, Callable[[], Figure]] = {
    "embeddings_categorical_umap": categorical_umap,
    "embeddings_continuous_overlay": continuous_umap,
    "embeddings_score_by_condition": score_by_condition,
}
