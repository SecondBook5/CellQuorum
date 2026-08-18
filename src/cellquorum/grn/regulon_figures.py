"""SCENIC downstream figures from the AUCell regulon-activity matrix.

Reimplements the pySCENIC-protocol downstream plots (RSS panels + Z-scored regulon-activity
clustermap) using the cellquorum house style, reading:
  - auc_mtx: per-cell regulon activity (cells x regulons), from `pyscenic aucell`
  - a per-cell grouping (generic group, cell_type, etc.), aligned by CellID

RSS (regulon specificity score) is the protocol's Jensen-Shannon-based metric: for each grouping
category, how SPECIFIC each regulon's activity is to that category (1 = perfectly specific). This is
independent of R_cond and complements it — it says which regulons *mark* each stage, where R_cond
says which the niche *redirects*.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellquorum.visualization import figstyle

if TYPE_CHECKING:
    from matplotlib.colors import ListedColormap


def regulon_specificity_scores(auc_mtx: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """RSS via Jensen-Shannon distance (the pySCENIC-protocol definition, reimplemented).

    For each regulon, treat its per-cell activity as a distribution; RSS for category c =
    1 - JSD(regulon_activity, indicator(cell in c)). Returns categories x regulons.
    """
    from scipy.spatial.distance import jensenshannon

    cats = sorted(groups.dropna().unique().astype(str))
    aucn = auc_mtx.div(
        auc_mtx.sum(axis=0) + 1e-12, axis=1
    )  # normalize each regulon to a distribution
    rss = np.zeros((len(cats), auc_mtx.shape[1]))
    for i, c in enumerate(cats):
        ind = (groups.astype(str) == c).astype(float).to_numpy()
        ind = ind / (ind.sum() + 1e-12)
        for j, reg in enumerate(auc_mtx.columns):
            rss[i, j] = 1.0 - jensenshannon(aucn[reg].to_numpy(), ind, base=2)
    return pd.DataFrame(rss, index=cats, columns=auc_mtx.columns)


def _align(auc_mtx: pd.DataFrame, groups: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    common = auc_mtx.index.intersection(groups.index)
    return auc_mtx.loc[common], groups.loc[common].astype(str)


def _annotation_palette(col: str, cats: list[str]) -> dict:
    """Map annotation categories -> house categorical colors in fixed sorted order."""
    from matplotlib.colors import to_rgb

    palette = figstyle.CATEGORICAL_PALETTE
    return {c: to_rgb(palette[i % len(palette)]) for i, c in enumerate(sorted(cats))}


def plot_regulon_cell_clustermap(
    auc_mtx: pd.DataFrame,
    annotations: pd.DataFrame | pd.Series,
    out_dir: Path | str,
    *,
    rss_groups: pd.Series | None = None,
    top_n: int = 5,
    max_cells: int = 8000,
    vmin: float = -1.5,
    vmax: float = 6.0,
    name: str = "scenic_regulon_cell_clustermap",
) -> list[Path]:
    """Canonical pySCENIC AUCell clustermap: CELLS x regulons, both hierarchically clustered.

    This is the per-cell, per-patient figure (not the compact category-mean one): every ROW is a
    single cell, columns are regulons, cells are colored by Z-scored regulon activity (YlGnBu,
    vmin/vmax = -1.5/6 as in the protocol). One or more ROW-COLOR bars annotate each cell by the
    columns of `annotations` (e.g. donor/patient, group, cell type) so the patient/group structure
    is visible down the left margin. A companion palplot-style legend is written per annotation.

    Args:
        auc_mtx: per-cell regulon activity (cells x regulons), from AUCell.
        annotations: per-cell labels aligned by index (auc_mtx.index). A DataFrame yields one
            row-color bar per column (order preserved); a Series yields a single bar.
        out_dir: output directory (clustermap PNG+PDF + one *_legend.pdf per annotation column).
        rss_groups: optional grouping used to pick the top-N RSS regulons per category to display
            (defaults to the FIRST annotation column). Keeps the column set legible on 100s of TFs.
        top_n: top RSS regulons per category to include as columns.
        max_cells: cap rows for tractable clustering (subsample, seeded) if exceeded.
        vmin, vmax: Z-score color limits (protocol defaults -1.5 / 6).

    Returns:
        List of written figure paths (PNG+PDF).
    """
    if auc_mtx is None or auc_mtx.shape[0] == 0 or auc_mtx.shape[1] == 0:
        return []

    import seaborn as sns

    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = annotations.to_frame() if isinstance(annotations, pd.Series) else annotations.copy()
    ann = ann.astype(str)

    # align cells present in both AUC and annotations
    common = auc_mtx.index.intersection(ann.index)
    if len(common) == 0:
        return []
    auc = auc_mtx.loc[common]
    ann = ann.loc[common]

    # subsample for tractable hierarchical clustering on very large cohorts (seeded, stratified by
    # the first annotation so every patient/group keeps representation)
    if len(auc) > max_cells:
        rng = np.random.default_rng(0)
        key = ann.iloc[:, 0]
        frac = max_cells / len(auc)
        keep = []
        for _, ix in key.groupby(key).groups.items():
            ix = np.array(ix)
            k = max(1, int(round(len(ix) * frac)))
            keep.extend(rng.choice(ix, size=min(k, len(ix)), replace=False))
        auc = auc.loc[keep]
        ann = ann.loc[keep]

    # column selection: top-N RSS regulons per category (union), so we show marker regulons not all
    grp = rss_groups.loc[auc.index].astype(str) if rss_groups is not None else ann.iloc[:, 0]
    try:
        rss = regulon_specificity_scores(auc, grp)
        topreg: list[str] = []
        for c in rss.index:
            topreg.extend(rss.loc[c].sort_values(ascending=False).head(top_n).index.tolist())
        topreg = sorted(set(topreg))
        if not topreg:
            return []
    except Exception:
        return []

    # Z-score each regulon across cells (the protocol's auc_mtx_Z)
    sub = auc[topreg]
    z = (sub - sub.mean(axis=0)) / (sub.std(axis=0, ddof=0) + 1e-12)

    # build one row-color bar per annotation column + remember palettes for the legends
    row_colors = pd.DataFrame(index=z.index)
    palettes: dict[str, dict] = {}
    for col in ann.columns:
        cats = sorted(ann[col].unique())
        pal = _annotation_palette(col, cats)
        palettes[col] = pal
        row_colors[col] = ann[col].map(pal)

    n_reg = len(topreg)
    g = sns.clustermap(
        z,
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        row_colors=row_colors,
        yticklabels=False,
        xticklabels=True,
        linewidths=0.0,
        linecolor="gray",
        square=False,
        figsize=(max(12, 0.42 * n_reg), 14),
        cbar_kws={"ticks": [vmin, 0, 1.5, 3.0, 4.5, vmax], "orientation": "horizontal"},
        col_cluster=True,
        row_cluster=True,
        dendrogram_ratio=(0.12, 0.06),
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.set_xticklabels(
        [t.get_text().replace("_(+)", "(+)") for t in g.ax_heatmap.get_xticklabels()],
        rotation=90,
        fontsize=8,
    )
    # seaborn parks the colorbar at the top-left, on top of the row dendrogram. Move it to the empty
    # top-right corner (above the heatmap, right side) so nothing overprints the dendrogram.
    g.cax.set_visible(True)
    hm = g.ax_heatmap.get_position()
    g.cax.set_position([hm.x1 - 0.16, hm.y1 + 0.02, 0.14, 0.018])
    g.cax.set_title("Z-scored regulon activity", fontsize=9, fontweight="bold", pad=4)
    g.cax.xaxis.set_ticks_position("bottom")
    g.cax.tick_params(labelsize=8)
    g.fig.suptitle(
        f"Per-cell regulon activity (top {top_n} RSS per {ann.columns[0]})",
        fontsize=14,
        fontweight="bold",
        x=0.5,
        y=1.005,
    )
    paths = figstyle.save_figure(g.fig, out_dir, name)

    # one row-color-key legend per annotation column (adaptive layout: palplot strip for a few short
    # labels, swatch grid with text beside each swatch for many/long labels e.g. cell types)
    for col, pal in palettes.items():
        _write_annotation_legend(pal, col, out_dir, f"{name}_legend_{col}")

    return paths


def mpl_listed(colors: list) -> ListedColormap:
    """ListedColormap from a color list (small local helper to avoid a
    top-level mpl.colors import).
    """
    from matplotlib.colors import ListedColormap

    return ListedColormap(list(colors))


def _dark(rgb: tuple) -> bool:
    """Perceived-luminance test so palplot labels stay legible on their swatch."""
    r, g, b = rgb[:3]
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.55


def _write_annotation_legend(pal: dict, col: str, out_dir: Path | str, fname: str) -> Path:
    """Row-color-key legend for one annotation column, laid out to stay readable.

    Two modes, chosen by category count + label length:
      * palplot strip (a horizontal color bar with the label centered on each swatch) — used only
        when there are few short labels (e.g. group: AAH/AIS). This is the compact protocol look.
      * swatch grid (a small square swatch with the label to its RIGHT, wrapped into columns) —
        used for many or long labels (e.g. 15 cell types), so text never overprints another swatch.
    """
    from matplotlib.patches import Rectangle

    cats = list(pal.keys())
    n = len(cats)
    longest = max((len(str(c)) for c in cats), default=0)
    use_strip = n <= 4 and longest <= 8

    if use_strip:
        f, ax = plt.subplots(1, 1, figsize=(max(n * 0.9, 3), 0.9))
        ax.imshow(
            np.arange(n).reshape(1, n),
            cmap=mpl_listed([pal[c] for c in cats]),
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xticks(np.arange(n) - 0.5)
        ax.set_yticks([-0.5, 0.5])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for idx, c in enumerate(cats):
            ax.text(
                idx,
                0.0,
                c,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if _dark(pal[c]) else "black",
            )
        ax.set_title(col, fontsize=10, fontweight="bold", pad=6)
        paths = figstyle.save_figure(f, out_dir, fname)
        return paths[0]

    # swatch grid: label to the right of each swatch, wrapped into columns so nothing overlaps.
    # Each legend column is `unit` wide in data coords: a 0.5-wide swatch + room for the longest
    # label, so a name never runs into the next column's swatch.
    ncol = 1 if n <= 8 else (2 if n <= 20 else 3)
    nrow = int(np.ceil(n / ncol))
    unit = 0.7 + 0.11 * longest  # data-x width per legend column (swatch + label)
    f, ax = plt.subplots(
        figsize=(max(ncol * (longest * 0.11 + 1.0), 2.4), max(nrow * 0.34 + 0.6, 1.2))
    )
    ax.set_xlim(0, ncol * unit)
    ax.set_ylim(0, nrow)
    ax.invert_yaxis()
    ax.axis("off")
    for i, c in enumerate(cats):
        cc, rr = divmod(i, nrow)
        x = cc * unit
        ax.add_patch(
            Rectangle(
                (x + 0.04, rr + 0.18), 0.5, 0.6, facecolor=pal[c], edgecolor="#333", linewidth=0.5
            )
        )
        ax.text(x + 0.62, rr + 0.48, str(c), ha="left", va="center", fontsize=9)
    ax.set_title(col, fontsize=10, fontweight="bold", loc="left")
    paths = figstyle.save_figure(f, out_dir, fname)
    return paths[0]


def plot_rss_panels(
    auc_mtx: pd.DataFrame,
    groups: pd.Series,
    out_dir: Path | str,
    *,
    group_label: str = "group",
    top_n: int = 5,
    name: str = "scenic_rss_panels",
) -> list[Path]:
    """One panel per category: regulons ranked by RSS, top-N labelled (protocol RSS figure)."""
    if auc_mtx is None or auc_mtx.shape[0] == 0 or auc_mtx.shape[1] == 0:
        return []

    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    auc_mtx, groups = _align(auc_mtx, groups)

    try:
        rss = regulon_specificity_scores(auc_mtx, groups)
        cats = list(rss.index)
        if not cats:
            return []
    except Exception:
        return []

    ncol = min(len(cats), 4)
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False, constrained_layout=True
    )
    try:
        from adjustText import adjust_text

        _have_at = True
    except Exception:
        _have_at = False

    for k, c in enumerate(cats):
        ax = axes[k // ncol][k % ncol]
        s = rss.loc[c].sort_values(ascending=False)
        ranks = np.arange(1, len(s) + 1)
        ax.scatter(ranks, s.values, s=8, c="#4C72B0", alpha=0.6, edgecolors="none")
        top = s.head(top_n)
        texts = []
        for r, (reg, val) in enumerate(top.items(), start=1):
            ax.scatter([r], [val], s=22, c="#C44E52", zorder=5)
            texts.append(ax.text(r, val, reg.replace("_(+)", ""), fontsize=8, fontweight="bold"))
        if _have_at and texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.5))
        ax.set_title(f"{group_label} = {c}", fontsize=11, fontweight="bold")
        ax.set_xlabel("regulon rank", fontsize=9)
        ax.set_ylabel("RSS", fontsize=9)
    # blank any unused axes
    for k in range(len(cats), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(
        f"Regulon specificity scores by {group_label}  (top {top_n} labelled)",
        fontsize=14,
        fontweight="bold",
    )
    return figstyle.save_figure(fig, out_dir, name)


def plot_regulon_clustermap(
    auc_mtx: pd.DataFrame,
    groups: pd.Series,
    out_dir: Path | str,
    *,
    group_label: str = "group",
    top_n: int = 5,
    name: str = "scenic_regulon_clustermap",
) -> list[Path]:
    """Z-scored regulon-activity clustermap over the union of each category's top-N RSS regulons,
    rows annotated by category (protocol clustermap figure)."""
    if auc_mtx is None or auc_mtx.shape[0] == 0 or auc_mtx.shape[1] == 0:
        return []

    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    auc_mtx, groups = _align(auc_mtx, groups)

    try:
        rss = regulon_specificity_scores(auc_mtx, groups)

        # union of top-N specific regulons per category
        topreg = []
        for c in rss.index:
            topreg.extend(rss.loc[c].sort_values(ascending=False).head(top_n).index.tolist())
        topreg = sorted(set(topreg))
        if not topreg:
            return []
    except Exception:
        return []

    # Z-score each regulon across cells, then average per category for a compact, legible map
    sub = auc_mtx[topreg]
    z = (sub - sub.mean(axis=0)) / (sub.std(axis=0, ddof=0) + 1e-12)
    z["_grp"] = groups.values
    cat_mean = z.groupby("_grp").mean()  # categories x regulons (mean Z-activity)

    try:
        import seaborn as sns

        g = sns.clustermap(
            cat_mean,
            cmap=figstyle.SEQUENTIAL_CMAP,
            figsize=(max(8, 0.35 * len(topreg)), 5),
            linewidths=0.3,
            linecolor="#eee",
            cbar_kws={"label": "mean Z-scored regulon activity"},
            xticklabels=[r.replace("_(+)", "") for r in cat_mean.columns],
            yticklabels=True,
            col_cluster=True,
            row_cluster=len(cat_mean) > 2,
        )
        g.ax_heatmap.set_xlabel("regulon")
        g.ax_heatmap.set_ylabel(group_label)
        g.ax_heatmap.tick_params(axis="x", labelsize=7, rotation=90)
        g.fig.suptitle(
            f"Group-specific regulon activity (top {top_n} RSS per {group_label})",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        return figstyle.save_figure(g.fig, out_dir, name)
    except Exception:
        # fallback: plain matshow
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(topreg)), 5))
        im = ax.imshow(cat_mean.values, cmap=figstyle.SEQUENTIAL_CMAP, aspect="auto")
        ax.set_xticks(range(len(cat_mean.columns)))
        ax.set_xticklabels(
            [r.replace("_(+)", "") for r in cat_mean.columns], rotation=90, fontsize=7
        )
        ax.set_yticks(range(len(cat_mean.index)))
        ax.set_yticklabels(cat_mean.index)
        fig.colorbar(im, ax=ax, label="mean Z-scored regulon activity")
        ax.set_title(f"Group-specific regulon activity (top {top_n} RSS)", fontweight="bold")
        return figstyle.save_figure(fig, out_dir, name)


def _tf_symbol(name: str) -> str:
    """Bare TF symbol from any convention: 'STAT1_(+)' / 'tf_STAT1' / 'STAT1(+)' -> 'STAT1'."""
    s = str(name)
    if s.startswith("tf_"):
        s = s[3:]
    return s.replace("_(+)", "").replace("(+)", "").replace("_(-)", "").replace("(-)", "").strip()


def plot_regulon_umap(
    auc_mtx: pd.DataFrame,
    umap: pd.DataFrame | np.ndarray,
    out_dir: Path | str,
    *,
    groups: pd.Series | None = None,
    regulons: list[str] | None = None,
    top_n: int = 12,
    name: str = "scenic_regulon_umap",
) -> list[Path]:
    """Overlay per-cell AUCell activity of key regulons on the cell UMAP (one small panel each).

    The canonical pySCENIC companion to the clustermap: shows *where* on the embedding each regulon
    is active. Regulons default to the union of the top-RSS-per-group set (needs `groups`) so the
    panel shows group/cell-type markers; pass `regulons` to force a specific list.

    Args:
        auc_mtx: per-cell regulon activity (cells x regulons); index = CellID.
        umap: (n_cells, 2) coords aligned to auc_mtx.index — a DataFrame indexed by CellID
            (preferred, robust to ordering) or a bare array in auc_mtx row order.
        out_dir: output directory.
        groups: optional per-cell grouping to pick top-RSS regulons when `regulons` is None.
        regulons: explicit regulon columns to plot (overrides the RSS pick).
        top_n: number of regulon panels when auto-picking.
        name: output basename.

    Returns:
        List of written figure paths (PNG+PDF).
    """
    if auc_mtx is None or auc_mtx.shape[0] == 0 or auc_mtx.shape[1] == 0:
        return []

    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # align UMAP coords to the AUCell cells
    if isinstance(umap, pd.DataFrame):
        common = auc_mtx.index.intersection(umap.index)
        if len(common) == 0:
            return []
        auc = auc_mtx.loc[common]
        xy = umap.loc[common].to_numpy()[:, :2]
    else:
        xy = np.asarray(umap)[:, :2]
        if xy.shape[0] != auc_mtx.shape[0]:
            return []
        auc = auc_mtx

    # choose regulons to display
    if regulons is None:
        if groups is not None:
            try:
                g = groups.reindex(auc.index).astype(str)
                rss = regulon_specificity_scores(auc, g)
                picks: list[str] = []
                per = max(1, top_n // max(len(rss.index), 1))
                for c in rss.index:
                    picks.extend(rss.loc[c].sort_values(ascending=False).head(per).index.tolist())
                regulons = list(dict.fromkeys(picks))[:top_n]
            except Exception:
                regulons = []
        else:  # fall back to the most variable regulons
            regulons = auc.var(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
    regulons = [r for r in regulons if r in auc.columns]
    if not regulons:
        return []

    ncol = min(len(regulons), 4)
    nrow = int(np.ceil(len(regulons) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow), squeeze=False)
    for k, reg in enumerate(regulons):
        ax = axes[k // ncol][k % ncol]
        v = auc[reg].to_numpy()
        # clip to 99th pct so a few high cells don't wash out the gradient (pySCENIC convention)
        vmax = np.quantile(v, 0.99) if np.any(v > 0) else 1.0
        order = np.argsort(v)  # draw high-activity cells last (on top)
        sc = ax.scatter(
            xy[order, 0],
            xy[order, 1],
            c=v[order],
            cmap="YlGnBu",
            s=4,
            vmin=0,
            vmax=max(vmax, 1e-9),
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(_tf_symbol(reg), fontsize=11, fontweight="bold", fontstyle="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    for k in range(len(regulons), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Regulon activity on the cell embedding (AUCell)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return figstyle.save_figure(fig, out_dir, name)
