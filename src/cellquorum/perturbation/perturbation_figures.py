"""Publication figures for the in-silico KO (CellOracle) stage.

House-styled on cellquorum.visualization.figstyle. Every plot returns the written
PNG+PDF paths and returns [] on empty/degenerate input — never raises to the caller,
so a single failed figure never sinks the stage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellquorum.visualization import figstyle


def plot_target_ranking(
    ranking_df: pd.DataFrame,
    out_dir: Path | str,
    *,
    n_top: int = 20,
    name: str = "perturbation_target_ranking",
) -> list[Path]:
    """Horizontal lollipop of the top-N knockout targets by shift score."""
    if ranking_df is None or ranking_df.shape[0] == 0 or "score" not in ranking_df:
        return []
    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = ranking_df.sort_values("score", ascending=False).head(n_top).iloc[::-1]
    color = figstyle.CATEGORICAL_PALETTE[0]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.32 * len(df))))
    y = np.arange(len(df))
    ax.hlines(y, 0, df["score"].to_numpy(), color=color, linewidth=2)
    ax.plot(df["score"].to_numpy(), y, "o", color=color, markersize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["tf"].astype(str).tolist(), fontsize=8)
    ax.set_xlabel("KO shift score")
    ax.set_title(f"Top {min(n_top, len(df))} in-silico knockout targets", fontweight="bold")
    return figstyle.save_figure(fig, out_dir, name)


def plot_ko_shift_field(
    shift_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    out_dir: Path | str,
    *,
    tf: str,
    groups: pd.Series | None = None,
    name: str | None = None,
) -> list[Path]:
    """Quiver of per-cell shift vectors on the 2-D embedding, aligned by index intersection.

    Args:
        shift_df: per-cell shift vectors; the first two columns are the 2-D shift
            components (e.g. d0, d1); index = cell.
        embedding_df: 2-D embedding; columns like `DIM1`, `DIM2`; index = cell.
        out_dir: output directory.
        tf: TF name for the title.
        groups: optional per-cell grouping for background coloring.
        name: output basename (default: f"perturbation_ko_shift_{tf}").

    Returns:
        List of written figure paths (PNG+PDF), or [] on empty/no-overlap.
    """
    if shift_df is None or shift_df.shape[0] == 0:
        return []
    if embedding_df is None or embedding_df.shape[0] == 0:
        return []
    if shift_df.shape[1] < 2:
        return []

    # Align by index intersection
    common = shift_df.index.intersection(embedding_df.index)
    if len(common) == 0:
        return []

    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shift = shift_df.loc[common]
    emb = embedding_df.loc[common]

    # Extract embedding coordinates (first two columns)
    x = emb.iloc[:, 0].to_numpy()
    y = emb.iloc[:, 1].to_numpy()
    dx = shift.iloc[:, 0].to_numpy()
    dy = shift.iloc[:, 1].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 7))

    # Optional background coloring by groups
    if groups is not None:
        g = groups.reindex(common).astype(str)
        cats = sorted(g.unique())
        for i, cat in enumerate(cats):
            mask = g == cat
            ax.scatter(
                x[mask],
                y[mask],
                c=[figstyle.CATEGORICAL_PALETTE[i % len(figstyle.CATEGORICAL_PALETTE)]],
                s=8,
                alpha=0.3,
                edgecolors="none",
                label=cat,
                rasterized=True,
            )
        ax.legend(loc="best", markerscale=2)
    else:
        # Plain background scatter
        ax.scatter(x, y, c="#cccccc", s=8, alpha=0.3, edgecolors="none", rasterized=True)

    # Quiver overlay
    # Downsample for legibility on dense embeddings
    n_cells = len(x)
    if n_cells > 500:
        # Show ~500 arrows evenly spaced
        step = max(1, n_cells // 500)
        idx = np.arange(0, n_cells, step)
    else:
        idx = np.arange(n_cells)

    ax.quiver(
        x[idx],
        y[idx],
        dx[idx],
        dy[idx],
        color=figstyle.CATEGORICAL_PALETTE[1],  # orange
        alpha=0.7,
        width=0.003,
        scale=None,
        scale_units="xy",
        angles="xy",
    )

    ax.set_xlabel(emb.columns[0])
    ax.set_ylabel(emb.columns[1])
    ax.set_title(f"{tf} knockout shift field", fontweight="bold")

    if name is None:
        name = f"perturbation_ko_shift_{tf}"

    return figstyle.save_figure(fig, out_dir, name)


def plot_ko_fate_summary(
    fate_df: pd.DataFrame,
    out_dir: Path | str,
    *,
    tf: str,
    name: str | None = None,
) -> list[Path]:
    """Bar of per-cluster net transition-probability change for one KO.

    Args:
        fate_df: per-cluster fate summary; columns `cluster`, `delta`.
        out_dir: output directory.
        tf: TF name for the title.
        name: output basename (default: f"perturbation_ko_fate_{tf}").

    Returns:
        List of written figure paths (PNG+PDF), or [] on empty.
    """
    if fate_df is None or fate_df.shape[0] == 0:
        return []
    if "cluster" not in fate_df.columns or "delta" not in fate_df.columns:
        return []

    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fate_df.sort_values("delta", ascending=True)

    # Color bars by sign: positive (enriched) vs negative (depleted)
    colors = [
        figstyle.CATEGORICAL_PALETTE[1] if d >= 0 else figstyle.CATEGORICAL_PALETTE[7]
        for d in df["delta"]
    ]

    fig, ax = plt.subplots(figsize=(6, max(3, 0.28 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["delta"].to_numpy(), color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["cluster"].astype(str).tolist(), fontsize=8)
    ax.set_xlabel("Net transition probability change")
    ax.axvline(0, color="#555", linewidth=1.0, linestyle="--")
    ax.set_title(f"{tf} knockout fate summary", fontweight="bold")

    if name is None:
        name = f"perturbation_ko_fate_{tf}"

    return figstyle.save_figure(fig, out_dir, name)


def plot_grn_connectivity(
    grn_summary_df: pd.DataFrame,
    out_dir: Path | str,
    *,
    n_top: int = 20,
    name: str = "perturbation_grn_connectivity",
) -> list[Path]:
    """Bar of top regulators by degree/connectivity.

    Accepts either the consumer's `degree` schema (columns `tf`, `degree`) or the
    producer's real per-cluster schema (columns `cluster`, `tf`, `n_targets`); in the
    latter case per-TF degree is derived by summing `n_targets` across clusters.

    Args:
        grn_summary_df: GRN summary; columns `tf` + either `degree` or `n_targets`.
        out_dir: output directory.
        n_top: number of top regulators to display.
        name: output basename.

    Returns:
        List of written figure paths (PNG+PDF), or [] on empty.
    """
    if grn_summary_df is None or grn_summary_df.shape[0] == 0:
        return []
    if "tf" not in grn_summary_df.columns:
        return []
    if "degree" not in grn_summary_df.columns and "n_targets" not in grn_summary_df.columns:
        return []

    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = grn_summary_df
    if "degree" not in df.columns:
        df = (
            df.groupby("tf", as_index=False)["n_targets"]
            .sum()
            .rename(columns={"n_targets": "degree"})
        )
    df = df.sort_values("degree", ascending=False).head(n_top).iloc[::-1]
    color = figstyle.CATEGORICAL_PALETTE[2]  # aqua

    fig, ax = plt.subplots(figsize=(6, max(3, 0.32 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["degree"].to_numpy(), color=color, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["tf"].astype(str).tolist(), fontsize=8)
    ax.set_xlabel("GRN degree (# target genes)")
    ax.set_title(f"Top {min(n_top, len(df))} regulators by connectivity", fontweight="bold")

    return figstyle.save_figure(fig, out_dir, name)
