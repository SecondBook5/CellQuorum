"""Biology-agnostic volcano primitive for the de_viz stage.

Ported from the house paired-volcano aesthetic: shaded significant quadrants,
case-red / control-blue point tiers, dotted fold-change lines, dashed FDR line,
optional-dependency adjustText labels, and a lower-left stats box. No file I/O,
no config objects, no biological literals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cellquorum.visualization import figstyle

_QUAD_RED = "#FFE4E6"
_QUAD_BLUE = "#E6F0FF"
_AMBIENT_EDGE = "#7A7A7A"
_HIGHLIGHT_COLOR = "#FFD700"  # gold: high-contrast against both case/control reds/blues


def volcano(
    df: pd.DataFrame,
    *,
    fc_cut: float,
    fdr_cut: float,
    case_color: str,
    control_color: str,
    x_label: str,
    top_n_labels: int = 40,
    ambient_col: str = "likely_ambient",
    highlight_genes: set[str] | frozenset[str] | None = None,
) -> Figure:
    """Render a pseudobulk volcano from a (gene, logFC, FDR) frame.

    ``highlight_genes``, when given, marks genes with a distinct outline and
    always labels them -- even when not DE-significant between conditions.
    Meant for a cluster's own identity markers: whether they're also
    condition-responsive is exactly what the highlight makes visible at a
    glance, not something the significance filter should hide.
    """
    if not np.isfinite(fc_cut) or fc_cut < 0:
        raise ValueError(f"fc_cut must be finite and nonnegative, got {fc_cut!r}")
    if not np.isfinite(fdr_cut) or not 0 < fdr_cut < 1:
        raise ValueError(f"fdr_cut must be strictly between zero and one, got {fdr_cut!r}")
    if ((df["FDR"] < 0) | (df["FDR"] > 1)).any():
        raise ValueError("FDR values must be probabilities between zero and one")
    figstyle.apply_theme("compact")
    d = df.copy()
    d = d[np.isfinite(d["logFC"]) & np.isfinite(d["FDR"])].copy()
    d["neg_log10fdr"] = -np.log10(d["FDR"].clip(lower=1e-300))

    sig = (d["FDR"] < fdr_cut) & (d["logFC"].abs() > fc_cut)
    up = d["logFC"] > 0
    if ambient_col in d.columns:
        ambient = d[ambient_col].fillna(False).to_numpy().astype(bool)
    else:
        ambient = np.zeros(len(d), dtype=bool)
    d["ambient"] = ambient

    n_up = int((sig & up & ~d["ambient"]).sum())
    n_down = int((sig & ~up & ~d["ambient"]).sum())

    fig = Figure(figsize=(8, 7))
    ax = fig.add_subplot(111)

    x_lim = float(d["logFC"].abs().max() * 1.15) if len(d) else 1.0
    x_lim = max(x_lim, fc_cut * 1.05, 0.1)
    y_lim = float(d["neg_log10fdr"].max() * 1.08) if len(d) else 1.0
    y_sig = -np.log10(fdr_cut)
    y_lim = max(y_lim, y_sig * 1.08, 0.1)

    ax.fill_between([fc_cut, x_lim], y_sig, y_lim, color=_QUAD_RED, alpha=0.5, zorder=0)
    ax.fill_between([-x_lim, -fc_cut], y_sig, y_lim, color=_QUAD_BLUE, alpha=0.5, zorder=0)

    ns = d[~sig | d["ambient"]]
    ax.scatter(
        ns["logFC"],
        ns["neg_log10fdr"],
        c="grey",
        s=15,
        alpha=0.4,
        edgecolors="none",
        linewidths=0,
        zorder=1,
    )
    up_sig = d[sig & up & ~d["ambient"]]
    ax.scatter(
        up_sig["logFC"],
        up_sig["neg_log10fdr"],
        c=case_color,
        s=28,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        zorder=2,
    )
    down_sig = d[sig & ~up & ~d["ambient"]]
    ax.scatter(
        down_sig["logFC"],
        down_sig["neg_log10fdr"],
        c=control_color,
        s=28,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        zorder=2,
    )
    amb = d[sig & d["ambient"]]
    ax.scatter(
        amb["logFC"],
        amb["neg_log10fdr"],
        facecolors="none",
        edgecolors=_AMBIENT_EDGE,
        s=30,
        linewidths=1.0,
        zorder=2,
    )

    highlight_set = {str(g) for g in (highlight_genes or ())}
    d["highlight"] = d["gene"].astype(str).isin(highlight_set)
    hl = d[d["highlight"]]
    if len(hl):
        # A hollow black-outline star at this size reads as noise against
        # the same-size colored dots; a solid, high-contrast fill is what
        # actually lets a reader pick these out at a glance.
        ax.scatter(
            hl["logFC"],
            hl["neg_log10fdr"],
            facecolors=_HIGHLIGHT_COLOR,
            edgecolors="black",
            marker="*",
            s=140,
            linewidths=1.1,
            zorder=3,
        )

    ax.axvline(fc_cut, color="grey", linestyle=":", linewidth=0.8, zorder=0)
    ax.axvline(-fc_cut, color="grey", linestyle=":", linewidth=0.8, zorder=0)
    ax.axvline(0, color="black", linewidth=0.5, zorder=0)
    ax.axhline(y_sig, color="grey", linestyle="--", linewidth=0.8, zorder=0)

    # Label the most-significant genes (cap to keep dense volcanoes legible),
    # plus every highlighted gene regardless of significance -- a cluster's
    # own identity markers are worth showing even when condition-stable.
    labelled = pd.concat([d[sig].sort_values("FDR").head(top_n_labels), hl]).drop_duplicates(
        subset="gene"
    )
    texts = []
    for _, r in labelled.iterrows():
        if r["ambient"]:
            color = _AMBIENT_EDGE
            style = "italic"
        elif r["highlight"]:
            # Always the highlight color, even when the gene is also
            # DE-significant -- falling back to the ordinary case/control
            # color here (the old behavior) defeated the point of
            # highlighting it: a reader couldn't tell it apart from the
            # other ~40 labeled genes.
            color = "black"
            style = "normal"
        else:
            color = case_color if r["logFC"] > 0 else control_color
            style = "normal"
        texts.append(
            ax.text(
                r["logFC"],
                r["neg_log10fdr"],
                str(r["gene"]),
                fontsize=6,
                fontweight="bold",
                style=style,
                color=color,
            )
        )
    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            ax=ax,
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.3, "alpha": 0.5},
            expand_points=(1.5, 1.5),
            force_text=(0.5, 0.5),
        )
    except Exception:  # noqa: BLE001 — adjustText optional; plain labels are fine
        pass

    stats_text = f"FDR<{fdr_cut} & |log2FC|>{fc_cut}:\n  {n_up} up, {n_down} down"
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("-log10(FDR)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


__all__ = ["volcano"]
