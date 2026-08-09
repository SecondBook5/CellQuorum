"""Biology-agnostic plotting primitives for ccc_viz.

Each function takes a tidy DataFrame plus explicit column-name arguments and
draws. No file I/O, no config objects, no biological literals. Optional plotting
deps (pycirclize, plotly) are import-guarded inside the functions that use them,
with a matplotlib fallback so a missing dep never raises.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure

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
    norm = Normalize(vmin=float(np.nanmin(w)), vmax=float(np.nanmax(w)) or 1.0)
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


__all__ = [
    "signed_norm",
    "celltype_palette",
    "interaction_dotplot",
    "cci_heatmap",
    "_CELLTYPE_HEXES",
    "_OTHER_GRAY",
]
