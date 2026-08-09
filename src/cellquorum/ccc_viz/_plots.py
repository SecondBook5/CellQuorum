"""Biology-agnostic plotting primitives for ccc_viz.

Each function takes a tidy DataFrame plus explicit column-name arguments and
draws. No file I/O, no config objects, no biological literals. Optional plotting
deps (pycirclize, plotly) are import-guarded inside the functions that use them,
with a matplotlib fallback so a missing dep never raises.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

# Colorblind-validated categorical theme (dataviz skill default; validated via
# scripts/validate_palette.js, CVD dE >= 8). Colors map to sorted cell-type
# positions at render time -- never to named biological types.
_CELLTYPE_HEXES = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
]
_OTHER_GRAY = "#9E9E9E"
_SEQ_CMAP = "viridis"


def signed_norm(values: np.ndarray) -> TwoSlopeNorm:
    """TwoSlopeNorm centered at 0, guaranteed vmin < 0 < vmax."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 0.0
    mag = max(abs(lo), abs(hi), 1e-6)
    eps = mag * 1e-3
    return TwoSlopeNorm(vmin=min(lo, -eps), vcenter=0.0, vmax=max(hi, eps))


def celltype_palette(cell_types: list[str]) -> dict[str, str]:
    """Deterministic sorted-celltype -> fixed hex; overflow -> gray. Entity-keyed."""
    uniq = sorted({str(c) for c in cell_types})
    out: dict[str, str] = {}
    for i, ct in enumerate(uniq):
        out[ct] = _CELLTYPE_HEXES[i] if i < len(_CELLTYPE_HEXES) else _OTHER_GRAY
    return out


def _empty_fig(msg: str = "no data") -> Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.set_axis_off()
    return fig


def interaction_dotplot(
    lr: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
    weight_col: str = "weight",
    sample_col: str = "sample",
    top_k: int = 15,
) -> Figure:
    """Top-N LR pairs by summed weight; x=source->target; color=mean, size=#samples."""
    if lr is None or lr.empty:
        return _empty_fig()
    df = lr.copy()
    df["__lr"] = df[ligand_col].astype(str) + "->" + df[receptor_col].astype(str)
    df["__st"] = df[source_col].astype(str) + "->" + df[target_col].astype(str)
    pair_weight = (
        df.groupby("__lr")[weight_col].sum().sort_values(kind="mergesort", ascending=False)
    )
    keep = list(pair_weight.head(top_k).index)
    df = df[df["__lr"].isin(keep)]
    if df.empty:
        return _empty_fig()
    ys = sorted(keep)  # deterministic
    xs = sorted(df["__st"].unique())
    y_pos = {v: i for i, v in enumerate(ys)}
    x_pos = {v: i for i, v in enumerate(xs)}
    agg = (
        df.groupby(["__lr", "__st"])
        .agg(
            mean_w=(weight_col, "mean"),
            n_samp=(sample_col, "nunique"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(xs)), max(4, 0.4 * len(ys))))
    w = agg["mean_w"].to_numpy(dtype=float)
    finite_w = w[np.isfinite(w)]
    if finite_w.size:
        vmin, vmax = float(finite_w.min()), float(finite_w.max())
        if vmax == vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    sizes = 20.0 + 40.0 * agg["n_samp"].to_numpy(dtype=float)
    sc = ax.scatter(
        [x_pos[s] for s in agg["__st"]],
        [y_pos[lr_pair] for lr_pair in agg["__lr"]],
        s=sizes,
        c=w,
        cmap=_SEQ_CMAP,
        norm=norm,
        edgecolor="0.3",
        linewidth=0.4,
    )
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=45, ha="right")
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels(ys)
    fig.colorbar(sc, ax=ax, label=weight_col)
    fig.tight_layout()
    return fig


def cci_heatmap(
    lr: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    diverging: bool = False,
) -> Figure:
    """source x target summed-weight matrix. Sequential, or diverging (center 0)."""
    if lr is None or lr.empty:
        return _empty_fig()
    mat = lr.pivot_table(
        index=source_col, columns=target_col, values=weight_col, aggfunc="sum", fill_value=0.0
    )
    mat = mat.sort_index(kind="mergesort").sort_index(axis=1, kind="mergesort")
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * mat.shape[1]), max(4, 0.5 * mat.shape[0])))
    vals = mat.to_numpy(dtype=float)
    if diverging:
        norm = signed_norm(vals.ravel())
        cmap = "RdBu_r"
    else:
        # weight >= 0 by canonical LR contract; sequential ramp anchored at 0
        norm = Normalize(vmin=0.0, vmax=float(np.nanmax(vals)) or 1.0)
        cmap = _SEQ_CMAP
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(list(mat.columns), rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(list(mat.index))
    ax.set_xlabel(target_col)
    ax.set_ylabel(source_col)
    fig.colorbar(im, ax=ax, label=weight_col)
    fig.tight_layout()
    return fig


def _top_edges(
    lr: pd.DataFrame, source_col: str, target_col: str, weight_col: str, top_k: int
) -> pd.DataFrame:
    agg = (
        lr.groupby([source_col, target_col])[weight_col]
        .sum()
        .reset_index()
        .sort_values([weight_col, source_col, target_col], kind="mergesort", ascending=False)
    )
    return agg.head(top_k)


def chord_diagram(
    lr: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    palette: dict[str, str],
    top_k: int = 15,
) -> Figure:
    """Circos chord (pycirclize) or matplotlib arc fallback. Never raises."""
    if lr is None or lr.empty:
        return _empty_fig()
    edges = _top_edges(lr, source_col, target_col, weight_col, top_k)
    if edges.empty:
        return _empty_fig()
    try:
        from pycirclize import Circos  # noqa: F401

        matrix = edges.pivot_table(
            index=source_col,
            columns=target_col,
            values=weight_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        matrix = matrix.sort_index(kind="mergesort").sort_index(axis=1, kind="mergesort")
        circos = Circos.initialize_from_matrix(
            matrix,
            cmap={
                ct: palette.get(ct, _OTHER_GRAY)
                for ct in sorted(set(matrix.index) | set(matrix.columns))
            },
        )
        return circos.plotfig()
    except Exception:  # noqa: BLE001  (import OR pycirclize runtime -> fallback)
        return _chord_fallback(edges, source_col, target_col, weight_col, palette)


def _chord_fallback(
    edges: pd.DataFrame, source_col: str, target_col: str, weight_col: str, palette: dict[str, str]
) -> Figure:
    """Matplotlib ring-of-arcs fallback (nodes on a circle, chords as lines)."""
    nodes = sorted(set(edges[source_col]) | set(edges[target_col]))
    ang = {n: 2 * np.pi * i / max(1, len(nodes)) for i, n in enumerate(nodes)}
    fig, ax = plt.subplots(figsize=(6, 6))
    for n, a in ang.items():
        ax.add_patch(Wedge((np.cos(a), np.sin(a)), 0.06, 0, 360, color=palette.get(n, _OTHER_GRAY)))
        ax.text(1.12 * np.cos(a), 1.12 * np.sin(a), str(n), ha="center", va="center", fontsize=8)
    wmax = float(edges[weight_col].max()) or 1.0
    for _, r in edges.iterrows():
        a0, a1 = ang[r[source_col]], ang[r[target_col]]
        ax.plot(
            [np.cos(a0), np.cos(a1)],
            [np.sin(a0), np.sin(a1)],
            color=palette.get(r[source_col], _OTHER_GRAY),
            linewidth=0.5 + 3.0 * r[weight_col] / wmax,
            alpha=0.6,
        )
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


def sankey_flow(
    lr: pd.DataFrame,
    *,
    source_col: str = "source",
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
    target_col: str = "target",
    weight_col: str = "weight",
    palette: dict[str, str],
    top_k: int = 15,
) -> Figure:
    """source->ligand->receptor->target flow. plotly+kaleido, else matplotlib fallback."""
    if lr is None or lr.empty:
        return _empty_fig()
    df = lr.copy()
    df["__w"] = df[weight_col].astype(float)
    df = df.sort_values("__w", kind="mergesort", ascending=False).head(top_k)
    if df.empty:
        return _empty_fig()
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        cols = [source_col, ligand_col, receptor_col, target_col]
        labels: list[str] = []
        idx: dict[tuple[int, str], int] = {}
        for ci, col in enumerate(cols):
            for v in sorted(df[col].astype(str).unique()):
                idx[(ci, v)] = len(labels)
                labels.append(v)
        src, tgt, val = [], [], []
        for ci in range(len(cols) - 1):
            g = df.groupby([cols[ci], cols[ci + 1]])["__w"].sum().reset_index()
            for _, r in g.iterrows():
                src.append(idx[(ci, str(r[cols[ci]]))])
                tgt.append(idx[(ci + 1, str(r[cols[ci + 1]]))])
                val.append(float(r["__w"]))
        figp = go.Figure(
            go.Sankey(
                node={"label": labels},
                link={"source": src, "target": tgt, "value": val},
            )
        )
        png = pio.to_image(figp, format="png", width=900, height=600, scale=2)
        import io

        import matplotlib.image as mpimg

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(mpimg.imread(io.BytesIO(png)))
        ax.set_axis_off()
        return fig
    except Exception:  # noqa: BLE001
        return _sankey_fallback(
            df, [source_col, ligand_col, receptor_col, target_col], weight_col, palette
        )


def _sankey_fallback(
    df: pd.DataFrame, cols: list[str], weight_col: str, palette: dict[str, str]
) -> Figure:
    """Matplotlib 4-column node/flow diagram."""
    fig, ax = plt.subplots(figsize=(9, 6))
    positions = {}
    for ci, col in enumerate(cols):
        vals = sorted(df[col].astype(str).unique())
        for vi, v in enumerate(vals):
            y = 1.0 - (vi + 0.5) / max(1, len(vals))
            positions[(ci, v)] = (ci, y)
            ax.scatter([ci], [y], s=200, color=palette.get(v, _OTHER_GRAY), zorder=3)
            ax.text(ci, y + 0.03, str(v), ha="center", va="bottom", fontsize=7)
    wmax = float(df[weight_col].max()) or 1.0
    for ci in range(len(cols) - 1):
        g = df.groupby([cols[ci], cols[ci + 1]])[weight_col].sum().reset_index()
        for _, r in g.iterrows():
            x0, y0 = positions[(ci, str(r[cols[ci]]))]
            x1, y1 = positions[(ci + 1, str(r[cols[ci + 1]]))]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="0.6",
                linewidth=0.5 + 3.0 * r[weight_col] / wmax,
                alpha=0.5,
                zorder=1,
            )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks([])
    ax.set_axis_off()
    return fig


def curvature_network(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    curvature_col: str = "ricci_curvature",
    weight_col: str = "weight",
    node_curv_col: str = "ricci_curvature",
    top_k: int = 30,
) -> Figure:
    """Curvature-colored network; seeded layout; diverging edge color (center 0)."""
    if edges is None or edges.empty:
        return _empty_fig()
    # Guard for missing weight_col (differential curvature frames lack it)
    if weight_col not in edges.columns:
        e = edges.copy()
        e[weight_col] = 1.0
    else:
        e = edges
    e = e.sort_values(weight_col, kind="mergesort", ascending=False).head(top_k)
    G = nx.DiGraph()
    for _, r in e.iterrows():
        G.add_edge(
            str(r[source_col]),
            str(r[target_col]),
            curv=float(r[curvature_col]),
            weight=float(r[weight_col]),
        )
    if G.number_of_edges() == 0:
        return _empty_fig()
    pos = nx.spring_layout(G, seed=0, weight="weight")
    curvs = np.array([d["curv"] for _, _, d in G.edges(data=True)], dtype=float)
    norm = signed_norm(curvs)
    cmap = plt.get_cmap("RdBu_r")
    fig, ax = plt.subplots(figsize=(7, 6))
    for (u, v, d), c in zip(G.edges(data=True), curvs, strict=False):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        wmax = curvs.size and float(e[weight_col].max()) or 1.0
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=cmap(norm(c)),
            linewidth=0.5 + 2.5 * d["weight"] / wmax,
            alpha=0.8,
            zorder=1,
        )
    ncurv = {}
    if nodes is not None and not nodes.empty and node_curv_col in nodes.columns:
        node_strs = nodes["node"].astype(str)
        node_vals = nodes[node_curv_col].astype(float)
        ncurv = dict(zip(node_strs, node_vals, strict=False))
    for n, (x, y) in pos.items():
        ax.scatter(
            [x],
            [y],
            s=120,
            color=cmap(norm(ncurv.get(n, 0.0))),
            edgecolor="0.2",
            linewidth=0.5,
            zorder=2,
        )
        ax.text(x, y, str(n), fontsize=7, ha="center", va="center")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=curvature_col)
    ax.set_axis_off()
    return fig


def topology_facets(
    topo: pd.DataFrame,
    *,
    node_col: str = "node",
    metric_cols: tuple[str, ...] = ("Listener", "Influencer", "Mediator", "Pagerank"),
    top_k: int = 15,
) -> Figure:
    """One small-multiple panel per metric: top-N nodes as horizontal bars."""
    if topo is None or topo.empty:
        return _empty_fig()
    metrics = [m for m in metric_cols if m in topo.columns]
    if not metrics:
        return _empty_fig()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), squeeze=False)
    for ax, m in zip(axes[0], metrics, strict=False):
        sub = topo.reindex(topo[m].abs().sort_values(kind="mergesort", ascending=False).index).head(
            top_k
        )
        sub = sub.iloc[::-1]
        vals = sub[m].to_numpy(dtype=float)
        straddles = float(np.nanmin(vals)) < 0 < float(np.nanmax(vals))
        if straddles:
            norm = signed_norm(vals)
            colors = plt.get_cmap("RdBu_r")(norm(vals))
        else:
            norm = Normalize(vmin=float(np.nanmin(vals)), vmax=float(np.nanmax(vals)) or 1.0)
            colors = plt.get_cmap(_SEQ_CMAP)(norm(vals))
        y = np.arange(len(sub))
        ax.barh(y, vals, color=colors, edgecolor="0.3", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(sub[node_col].astype(str).tolist())
        ax.set_title(m)
    fig.tight_layout()
    return fig


__all__ = [
    "signed_norm",
    "celltype_palette",
    "interaction_dotplot",
    "cci_heatmap",
    "chord_diagram",
    "sankey_flow",
    "curvature_network",
    "topology_facets",
    "_CELLTYPE_HEXES",
    "_OTHER_GRAY",
]
