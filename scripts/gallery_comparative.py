"""Reference-gallery fixtures for the comparative-stage figures.

Covers differential expression, enrichment, differential abundance and composition,
and state scoring, each drawn from small synthetic tables with a fixed seed.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.stages.comparative.differential_abundance import composition_figure, da_figures
from cellquorum.stages.comparative.differential_expression.viz import plots as de_plots
from cellquorum.stages.comparative.enrichment.viz import plots as enrich
from cellquorum.stages.state_scoring.viz.plots import state_score_violin
from cellquorum.visualization.figstyle import LE_RED, NORMAL_BLUE, apply_cellquorum_theme

CASE, CONTROL = "Lymphedema", "Normal"
TYPES = {"0": "Papillary", "1": "Reticular", "2": "Perivascular", "3": "Mechanosensory"}


def volcano() -> Figure:
    rng = np.random.default_rng(31)
    table = pd.DataFrame(
        {
            "gene": [f"Gene{i:03d}" for i in range(200)],
            "logFC": rng.normal(0, 1.4, 200),
            "FDR": np.power(10.0, -rng.uniform(0.1, 6.0, 200)),
            "likely_ambient": np.arange(200) % 23 == 0,
        }
    )
    return de_plots.volcano(
        table,
        fc_cut=1,
        fdr_cut=0.05,
        case_color=LE_RED,
        control_color=NORMAL_BLUE,
        x_label="Log2 fold change (Lymphedema / Normal)",
        top_n_labels=12,
        highlight_genes={"Gene007", "Gene042"},
    )


def _activity_table() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "source": [f"HALLMARK_PATHWAY_{i}" for i in range(16)],
            "score": np.concatenate([rng.uniform(1, 5, 8), -rng.uniform(1, 5, 8)]),
            "padj": np.power(10.0, -rng.uniform(0.5, 6, 16)),
        }
    )


def diverging_bar() -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    enrich.diverging_bar(
        _activity_table(), value_col="score", label_col="source", pvalue_col="padj", top_k=6, ax=ax
    )
    return fig


def activity_dotplot() -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    enrich.activity_dotplot(
        _activity_table(), value_col="score", label_col="source", pvalue_col="padj", top_k=6, ax=ax
    )
    return fig


def running_es() -> Figure:
    n = 120
    hit = np.zeros(n, dtype=int)
    hit[[3, 8, 12, 20, 25, 31, 44, 52, 60, 77]] = 1
    weights = np.abs(np.linspace(4, -4, n)) * hit
    running = np.cumsum(weights / weights.sum() - (1 - hit) / (1 - hit).sum())
    df = pd.DataFrame(
        {
            "rank": np.arange(1, n + 1),
            "running_es": running,
            "hit": hit,
            "metric": np.linspace(4, -4, n),
            "gene": [f"GENE{i}" for i in range(n)],
        }
    )
    return enrich.running_es_curve(df, title="Synthetic pathway")


def _ora_table() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for direction in ("up", "down"):
        for i in range(8):
            rows.append(
                {
                    "source": f"HALLMARK_{direction.upper()}_SET_{i}",
                    "direction": direction,
                    "count": int(rng.integers(3, 30)),
                    "ratio": float(rng.uniform(0.02, 0.3)),
                    "padj": float(np.power(10.0, -rng.uniform(1, 6))),
                }
            )
    return pd.DataFrame(rows)


def ora_barplot() -> Figure:
    return enrich.ora_barplot(
        _ora_table(),
        count_col="count",
        label_col="source",
        padj_col="padj",
        facet_col="direction",
        top_k=6,
    )


def ora_dotplot() -> Figure:
    return enrich.ora_dotplot(
        _ora_table(),
        ratio_col="ratio",
        count_col="count",
        padj_col="padj",
        label_col="source",
        facet_col="direction",
        top_k=6,
    )


def clustermap() -> Figure:
    rng = np.random.default_rng(3)
    matrix = pd.DataFrame(
        rng.normal(size=(10, 8)),
        index=[f"PATHWAY_{i}" for i in range(10)],
        columns=[f"sample{j}" for j in range(8)],
    )
    matrix.iloc[:5, 4:] += 2.0
    colors = pd.Series([NORMAL_BLUE] * 4 + [LE_RED] * 4, index=matrix.columns)
    return enrich.annotated_clustermap(matrix, col_colors=colors).fig


def _cross_group_table() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        [
            {
                "source": f"TF{i}",
                "cell_type": ct,
                "condition": cond,
                "score": float(rng.normal(2.0 if cond == CASE else 0.0, 1.0) + i * 0.1),
            }
            for i in range(8)
            for ct in TYPES
            for cond in (CONTROL, CASE)
        ]
    )


def cross_group_dotplot() -> Figure:
    return enrich.cross_group_dotplot(
        _cross_group_table().query("condition == @CASE"),
        row_col="source",
        col_col="cell_type",
        value_col="score",
        top_k=8,
        col_labels=TYPES,
    )


def cross_group_dotplot_by_condition() -> Figure:
    return enrich.cross_group_dotplot_by_condition(
        _cross_group_table(),
        row_col="source",
        col_col="cell_type",
        value_col="score",
        condition_col="condition",
        control=CONTROL,
        case=CASE,
        top_k=8,
        col_labels=TYPES,
    )


def _proportions() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    base = {"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.1}
    shifted = {"0": 0.2, "1": 0.45, "2": 0.2, "3": 0.15}
    rows = []
    for donor in ("d1", "d2", "d3", "d4"):
        for cond, props in ((CONTROL, base), (CASE, shifted)):
            noisy = np.array([props[k] for k in TYPES]) * rng.uniform(0.85, 1.15, 4)
            noisy /= noisy.sum()
            for ct, p in zip(TYPES, noisy, strict=True):
                rows.append(
                    {
                        "sample": f"{donor}_{cond}",
                        "donor": donor,
                        "condition": cond,
                        "cell_type": ct,
                        "count": int(p * 1000),
                        "proportion": float(p),
                    }
                )
    return pd.DataFrame(rows)


def condition_composition() -> Figure:
    return composition_figure.plot_condition_composition(
        _proportions(), case=CASE, control=CONTROL, cluster_labels=TYPES
    )


def per_patient_composition() -> Figure:
    return composition_figure.plot_per_patient_composition(
        _proportions(), case=CASE, control=CONTROL, cluster_labels=TYPES
    )


def milo_beeswarm() -> Figure:
    rng = np.random.default_rng(6)
    n = 120
    da = pd.DataFrame(
        {
            "nhood": np.arange(n),
            "logFC": rng.normal(0, 1.5, n),
            "PValue": rng.uniform(0, 1, n),
            "SpatialFDR": rng.uniform(0, 0.4, n),
            "nhood_size": rng.integers(20, 80, n),
            "majority_celltype": rng.choice(list(TYPES), n),
            "celltype_fraction": rng.uniform(0.6, 1.0, n),
        }
    )
    return da_figures.plot_milo_beeswarm(da, case=CASE, control=CONTROL, cluster_labels=TYPES)


def sccoda_composition() -> Figure:
    effects = pd.DataFrame(
        {
            "cell_type": list(TYPES),
            "log2_fold_change": [-1.2, 0.9, 0.05, 0.6],
            "inclusion_probability": [0.85, 0.92, 0.2, 0.7],
            "credible_effect": [True, True, False, True],
            "reference": "auto",
        }
    )
    return da_figures.plot_sccoda_composition(
        effects, _proportions(), case=CASE, control=CONTROL, cluster_labels=TYPES
    )


def state_score() -> Figure:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        [
            {"cluster": cl, "condition": cond, "score": float(s)}
            for cl in TYPES
            for cond, shift in ((CONTROL, 0.0), (CASE, 0.3))
            for s in rng.normal(0.4 + shift, 0.08, 50)
        ]
    )
    return state_score_violin(
        df,
        cluster_col="cluster",
        condition_col="condition",
        score_col="score",
        case=CASE,
        control=CONTROL,
        case_color=LE_RED,
        control_color=NORMAL_BLUE,
        score_label="fibrosis_ecm (ucell)",
        cluster_labels=TYPES,
    )


THEME = apply_cellquorum_theme

FIGURES: dict[str, Callable[[], Figure]] = {
    "de_volcano": volcano,
    "enrichment_diverging_bar": diverging_bar,
    "enrichment_activity_dotplot": activity_dotplot,
    "enrichment_running_es": running_es,
    "enrichment_ora_barplot": ora_barplot,
    "enrichment_ora_dotplot": ora_dotplot,
    "enrichment_clustermap": clustermap,
    "enrichment_cross_group_dotplot": cross_group_dotplot,
    "enrichment_cross_group_by_condition": cross_group_dotplot_by_condition,
    "da_condition_composition": condition_composition,
    "da_per_patient_composition": per_patient_composition,
    "da_milo_beeswarm": milo_beeswarm,
    "da_sccoda_composition": sccoda_composition,
    "state_scoring_violin": state_score,
}
