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


def plot_ko_shift_grid(
    shift_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    out_dir: Path | str,
    *,
    tf: str,
    groups: pd.Series | None = None,
    n_grid: int = 40,
    smooth: float = 0.5,
    n_neighbors: int = 100,
    min_mass_percentile: float = 30.0,
    name: str | None = None,
) -> list[Path]:
    """Gridded KO shift vector field on the 2-D embedding (CellOracle-style).

    Reproduces CellOracle's ``calculate_grid_arrows`` in-process from the per-cell
    shift parquet: a regular grid over the embedding, each gridpoint's arrow a
    Gaussian-distance-weighted mean of its neighboring cells' shift vectors, with
    low-density gridpoints masked out. This is the legible publication figure the
    raw per-cell quiver approximates — cells give context, the grid gives the flow.

    Args:
        shift_df: per-cell shift vectors; first two columns are the 2-D components.
        embedding_df: 2-D embedding; first two columns are coordinates; index = cell.
        out_dir: output directory.
        tf: TF name for the title.
        groups: optional per-cell grouping for faint background coloring.
        n_grid: grid steps per axis.
        smooth: Gaussian kernel width as a multiple of grid step (CellOracle default 0.5).
        n_neighbors: neighbors per gridpoint used in the weighted average.
        min_mass_percentile: gridpoints below this density percentile are dropped.
        name: output basename (default: f"perturbation_ko_shift_grid_{tf}").

    Returns:
        List of written figure paths (PNG+PDF), or [] on empty/no-overlap.
    """
    if shift_df is None or shift_df.shape[0] == 0 or shift_df.shape[1] < 2:
        return []
    if embedding_df is None or embedding_df.shape[0] == 0 or embedding_df.shape[1] < 2:
        return []
    common = shift_df.index.intersection(embedding_df.index)
    if len(common) < 3:
        return []

    from scipy.stats import norm as _norm
    from sklearn.neighbors import NearestNeighbors

    emb = embedding_df.loc[common].iloc[:, :2].to_numpy(dtype=float)
    delta = shift_df.loc[common].iloc[:, :2].to_numpy(dtype=float)

    # Regular grid over the embedding, matching CellOracle's 2.5% margin expansion.
    grs = []
    for dim in range(2):
        m, M = float(emb[:, dim].min()), float(emb[:, dim].max())
        m = m - 0.025 * abs(M - m)
        M = M + 0.025 * abs(M - m)
        grs.append(np.linspace(m, M, n_grid))
    mesh = np.meshgrid(*grs)
    gridpoints = np.vstack([g.flat for g in mesh]).T

    k = min(n_neighbors, len(common))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(emb)
    dists, neighs = nn.kneighbors(gridpoints)

    std = float(np.mean([g[1] - g[0] for g in grs]))
    gaussian_w = _norm.pdf(loc=0, scale=smooth * std, x=dists)
    total_mass = gaussian_w.sum(1)
    uz = (delta[neighs] * gaussian_w[:, :, None]).sum(1) / np.maximum(1, total_mass)[:, None]

    # Mass filter: keep only gridpoints with enough underlying cell density.
    mass_thresh = np.percentile(total_mass, min_mass_percentile)
    keep = total_mass >= mass_thresh
    if not keep.any():
        return []

    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    # Faint cell background for anatomical context.
    if groups is not None:
        g = groups.reindex(common).astype(str)
        for i, cat in enumerate(sorted(g.unique())):
            mask = (g == cat).to_numpy()
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=[figstyle.CATEGORICAL_PALETTE[i % len(figstyle.CATEGORICAL_PALETTE)]],
                s=6,
                alpha=0.25,
                edgecolors="none",
                label=cat,
                rasterized=True,
            )
        ax.legend(loc="best", markerscale=2, fontsize=7, framealpha=0.7)
    else:
        ax.scatter(
            emb[:, 0], emb[:, 1], c="#d9d9d9", s=6, alpha=0.3, edgecolors="none", rasterized=True
        )

    ax.quiver(
        gridpoints[keep, 0],
        gridpoints[keep, 1],
        uz[keep, 0],
        uz[keep, 1],
        color=figstyle.CATEGORICAL_PALETTE[1],
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.004,
        alpha=0.9,
    )
    ax.set_xlabel(str(embedding_df.columns[0]))
    ax.set_ylabel(str(embedding_df.columns[1]))
    ax.set_title(f"{tf} knockout — gridded shift field", fontweight="bold")

    if name is None:
        name = f"perturbation_ko_shift_grid_{tf}"
    return figstyle.save_figure(fig, out_dir, name)


def plot_ko_fate_summary(
    fate_df: pd.DataFrame,
    out_dir: Path | str,
    *,
    tf: str,
    name: str | None = None,
) -> list[Path]:
    """Bar of per-cluster mean KO shift magnitude for one knockout.

    `delta` is a non-negative per-cluster mean shift magnitude (how strongly each
    cluster's cells move under the knockout), so bars are ranked largest-first and
    share one color — there is no sign to encode.

    Args:
        fate_df: per-cluster fate summary; columns `cluster`, `delta` (delta >= 0).
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

    # Rank clusters by how strongly they move under the KO (largest at top).
    df = fate_df.sort_values("delta", ascending=True)
    color = figstyle.CATEGORICAL_PALETTE[1]

    fig, ax = plt.subplots(figsize=(6, max(3, 0.28 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["delta"].to_numpy(), color=color, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["cluster"].astype(str).tolist(), fontsize=8)
    ax.set_xlabel("Mean KO shift magnitude")
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
