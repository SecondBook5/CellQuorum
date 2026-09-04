"""Differential-abundance figures for the Milo and scCODA methods.

Study-agnostic engine figures driven by the per-method result tables:

* a Milo neighborhood beeswarm (log-fold-change swarmed by majority cell type,
  colored by direction and SpatialFDR significance), and
* an scCODA composition figure (a per-condition proportion dumbbell beside the
  posterior credibility of each cell type).

Everything is biology-free: condition labels and colors are arguments, and the
data-prep/ordering helpers take plain DataFrames (never AnnData) so they
unit-test on tiny synthetic fixtures. Generalized from the project owner's
validated ``plot_da_beeswarm.py`` and ``plot_sccoda.py`` scripts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from cellquorum.visualization.figstyle import (
    LE_RED,
    NORMAL_BLUE,
    TEXT,
    diverging_norm,
    set_style,
)

# Neutral greys shared across the DA figures (non-significant marks, connectors).
_GREY_POINT = "#cfd3d8"
_GREY_BAR = "#c2c7cd"
_GREY_MUTED = "#9aa0a6"
_ROW_BAND = "#f4f5f7"


def prepare_milo_beeswarm(da: pd.DataFrame, *, spatial_fdr: float = 0.1) -> pd.DataFrame:
    """Prepare a Milo DA table for beeswarm rendering.

    Drops neighborhoods with no majority cell type (they cannot be placed on the
    categorical axis) and adds a boolean ``significant`` column from the
    SpatialFDR cutoff.

    Args:
        da: Milo output table with at least ``majority_celltype``, ``logFC``, and
            ``SpatialFDR`` columns, as written by ``milo.R``.
        spatial_fdr: Neighborhoods with ``SpatialFDR`` strictly below this are
            flagged significant.

    Returns:
        A copy of the in-scope rows (unannotated neighborhoods removed) with an
        added ``significant`` boolean column.
    """

    prepared = da.copy()
    prepared = prepared[prepared["majority_celltype"].notna()].reset_index(drop=True)
    prepared["significant"] = prepared["SpatialFDR"].astype(float) < spatial_fdr
    return prepared


def sccoda_single_reference(da: pd.DataFrame, *, reference: str = "auto") -> pd.DataFrame:
    """Select a single reference block from an scCODA result table.

    scCODA can emit two stacked blocks (an automatic-reference fit and an
    explicit-reference fit), disambiguated by the ``reference`` column. Plots need
    exactly one block. If the requested reference is absent, falls back to the
    first reference value present so a figure is always produced.

    Args:
        da: scCODA output table; may or may not carry a ``reference`` column.
        reference: Preferred reference label to keep.

    Returns:
        The rows for a single reference (or the whole frame if there is no
        ``reference`` column), with a fresh integer index.

    Notes:
        Delegates to :func:`~cellquorum.stages.comparative.differential_abundance
        .reference_selection.split_reference_fits`, which is the one place that
        decides which stacked fit is the reported one. The figure and the reported
        metrics must not be able to disagree about that.
    """

    # Local import: reference_selection is pure pandas, but this module is the
    # matplotlib one and the method imports the rule directly from there.
    from cellquorum.stages.comparative.differential_abundance.reference_selection import (
        split_reference_fits,
    )

    primary, _ = split_reference_fits(da, reference)
    return primary


def milo_beeswarm_order(prepared: pd.DataFrame) -> list[str]:
    """Order cell types by median neighborhood logFC, descending.

    The most case-enriched lineage (highest median logFC) sits at the top row,
    matching the validated beeswarm layout.

    Args:
        prepared: Output of :func:`prepare_milo_beeswarm`.

    Returns:
        Cell-type labels, most case-enriched first.
    """

    medians = prepared.groupby("majority_celltype")["logFC"].median().sort_values(ascending=False)
    return [str(ct) for ct in medians.index]


def sccoda_composition_order(
    effects: pd.DataFrame, proportions: pd.DataFrame, *, case: str
) -> list[str]:
    """Order the tested cell types by case-arm mean proportion, ascending.

    The smallest population sits at the bottom row (largest at the top), shared
    across both panels of the scCODA figure. Only cell types present in the
    scCODA ``effects`` table are placed (those are what was tested).

    Args:
        effects: Single-reference scCODA result with a ``cell_type`` column.
        proportions: Tidy composition table with ``condition``, ``cell_type``,
            and ``proportion`` columns (from ``build_composition_proportions``).
        case: Condition label whose mean proportions drive the ordering.

    Returns:
        Cell-type labels ordered by ascending case mean proportion; ties broken
        alphabetically.
    """

    effect_cts = list(dict.fromkeys(str(ct) for ct in effects["cell_type"]))
    case_props = proportions[proportions["condition"] == case]
    mean_prop = case_props.groupby("cell_type")["proportion"].mean()
    return sorted(effect_cts, key=lambda ct: (float(mean_prop.get(ct, 0.0)), ct))


def _sina_y(
    vals: np.ndarray, center: float, *, rng: np.random.Generator, half: float = 0.40
) -> np.ndarray:
    """Quasi-random y-jitter within a KDE density envelope (violin-shaped swarm).

    Falls back to a narrow uniform jitter when there are too few points to
    estimate a density.
    """

    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return np.array([])
    if vals.size < 3 or np.allclose(vals, vals[0]):
        return center + rng.uniform(-0.12, 0.12, size=vals.size)
    try:
        kde = gaussian_kde(vals)
        dens = kde(vals)
        dens = dens / dens.max()
    except Exception:
        dens = np.full(vals.size, 0.4)
    return center + rng.uniform(-1, 1, size=vals.size) * dens * half


def plot_milo_beeswarm(
    da: pd.DataFrame,
    *,
    case: str,
    control: str,
    spatial_fdr: float = 0.1,
    case_color: str = LE_RED,
    control_color: str = NORMAL_BLUE,
    seed: int = 0,
) -> Figure:
    """Render the Milo neighborhood-level DA beeswarm.

    Each neighborhood is a point placed at its ``logFC`` within a per-cell-type
    row. Non-significant neighborhoods are drawn light grey behind; significant
    ones (``SpatialFDR`` below ``spatial_fdr``) are colored by logFC on a
    control→case diverging map.

    Args:
        da: Raw Milo output table (see :func:`prepare_milo_beeswarm`).
        case: Condition label enriched at positive logFC.
        control: Condition label enriched at negative logFC.
        spatial_fdr: Significance cutoff for coloring.
        case_color: Warm pole of the diverging map / case cue color.
        control_color: Cool pole of the diverging map / control cue color.
        seed: RNG seed for the deterministic swarm jitter.

    Returns:
        The composed matplotlib :class:`~matplotlib.figure.Figure`.
    """

    set_style()
    rng = np.random.default_rng(seed)
    prepared = prepare_milo_beeswarm(da, spatial_fdr=spatial_fdr)
    order = milo_beeswarm_order(prepared)
    row_of = {ct: i for i, ct in enumerate(order)}
    n_rows = len(order)

    cmap = LinearSegmentedColormap.from_list(
        "cond_div", [control_color, "#5b83ad", "#eef0f2", "#d17a86", case_color], N=256
    )
    sig_abs = prepared.loc[prepared["significant"], "logFC"].abs()
    vmax = float(np.quantile(sig_abs, 0.98)) if len(sig_abs) else 1.0
    vmax = max(vmax, 0.5)
    norm = diverging_norm(np.array([-vmax, vmax]), vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.6, 0.46 * n_rows + 1.4))
    for i in range(n_rows):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=_ROW_BAND, zorder=0, lw=0)
    ax.axvline(0.0, color=TEXT, lw=0.9, ls=(0, (4, 3)), zorder=1, alpha=0.7)

    # Non-significant behind (grey), then significant on top (colored by logFC).
    for ct in order:
        sub = prepared[prepared["majority_celltype"] == ct]
        ns = sub[~sub["significant"]]
        if len(ns):
            y = _sina_y(ns["logFC"].to_numpy(), row_of[ct], rng=rng)
            ax.scatter(ns["logFC"], y, s=5.0, c=_GREY_POINT, alpha=0.55, linewidths=0, zorder=2)
    for ct in order:
        sub = prepared[prepared["majority_celltype"] == ct]
        sg = sub[sub["significant"]]
        if len(sg):
            y = _sina_y(sg["logFC"].to_numpy(), row_of[ct], rng=rng)
            ax.scatter(
                sg["logFC"],
                y,
                s=9.5,
                c=sg["logFC"],
                cmap=cmap,
                norm=norm,
                alpha=0.9,
                linewidths=0.15,
                edgecolors="#33373b",
                zorder=3,
            )

    # Right-margin count: significant / total neighborhoods per lineage.
    if n_rows:
        xmax = float(prepared["logFC"].max())
        xmin = float(prepared["logFC"].min())
        pad = (xmax - xmin) * 0.02 if xmax > xmin else 0.05
        for ct in order:
            sub = prepared[prepared["majority_celltype"] == ct]
            n_sig = int(sub["significant"].sum())
            ax.text(
                xmax + pad,
                row_of[ct],
                f"{n_sig}/{len(sub)}",
                va="center",
                ha="left",
                fontsize=6.2,
                color=(TEXT if n_sig else _GREY_MUTED),
            )

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel(f"Neighborhood log fold-change  ({case} vs {control})")
    ax.set_ylabel("")
    ax.set_title("Differential abundance (Milo) — neighborhood-level", fontsize=10, pad=8)
    ax.margins(x=0.04)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.annotate(
        f"↑ enriched in {case}",
        xy=(0.995, -0.14),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7,
        color=case_color,
        fontweight="bold",
    )
    ax.annotate(
        f"enriched in {control} ↓",
        xy=(0.005, -0.14),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7,
        color=control_color,
        fontweight="bold",
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.13, aspect=22)
    cbar.set_label(f"logFC (SpatialFDR < {spatial_fdr:.2f})", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)
    return fig


def plot_sccoda_composition(
    effects: pd.DataFrame,
    proportions: pd.DataFrame,
    *,
    case: str,
    control: str,
    reference: str = "auto",
    case_color: str = LE_RED,
    control_color: str = NORMAL_BLUE,
    seed: int = 0,
) -> Figure:
    """Render the two-panel scCODA composition figure.

    Panel A is a per-cell-type dumbbell of mean proportion (control vs case,
    with faint per-sample points on a log-percent axis). Panel B is a horizontal
    bar of each cell type's posterior inclusion probability, colored in the case
    color when the effect is credible, against the spike-and-slab prior at 0.5.

    Args:
        effects: scCODA result table (may carry two reference blocks).
        proportions: Tidy composition table (from ``build_composition_proportions``).
        case: Case/disease condition label.
        control: Control/normal condition label.
        reference: scCODA reference block to display.
        case_color: Case marker / credible-bar color.
        control_color: Control marker color.
        seed: RNG seed for the deterministic per-sample jitter.

    Returns:
        The composed matplotlib :class:`~matplotlib.figure.Figure`.
    """

    set_style()
    eff = sccoda_single_reference(effects, reference=reference).set_index("cell_type")
    order = sccoda_composition_order(eff.reset_index(), proportions, case=case)
    n = len(order)

    means = (
        proportions.groupby(["condition", "cell_type"])["proportion"].mean().unstack(fill_value=0.0)
    )

    def mean_pct(cond: str, ct: str) -> float:
        try:
            return float(means.loc[cond, ct]) * 100.0
        except KeyError:
            return 0.0

    def samples_pct(cond: str, ct: str) -> np.ndarray:
        mask = (proportions["condition"] == cond) & (proportions["cell_type"] == ct)
        return proportions.loc[mask, "proportion"].to_numpy() * 100.0

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(11.4, 0.42 * n + 1.6),
        gridspec_kw={"width_ratios": [2.15, 1.0], "wspace": 0.32},
    )

    # --- Panel A: proportion dumbbell (log-percent x-axis) --------------------
    floor = 1e-2  # percent floor so a zero proportion still renders on log scale
    for i in range(n):
        if i % 2 == 0:
            ax_a.axhspan(i - 0.5, i + 0.5, color=_ROW_BAND, zorder=0, lw=0)
    for y, ct in enumerate(order):
        nl = max(mean_pct(control, ct), floor)
        le = max(mean_pct(case, ct), floor)
        pn = np.clip(samples_pct(control, ct), floor, None)
        pl = np.clip(samples_pct(case, ct), floor, None)
        ax_a.scatter(
            pn,
            np.full_like(pn, y) + np.random.default_rng(seed).uniform(-0.13, 0.13, pn.size),
            s=8,
            color=control_color,
            alpha=0.30,
            linewidths=0,
            zorder=2,
        )
        ax_a.scatter(
            pl,
            np.full_like(pl, y) + np.random.default_rng(seed + 1).uniform(-0.13, 0.13, pl.size),
            s=8,
            color=case_color,
            alpha=0.30,
            linewidths=0,
            zorder=2,
        )
        ax_a.plot([nl, le], [y, y], color="#b8bcc2", lw=1.3, zorder=3, solid_capstyle="round")
        ax_a.scatter([nl], [y], s=42, color=control_color, edgecolors="white", lw=0.6, zorder=4)
        ax_a.scatter([le], [y], s=42, color=case_color, edgecolors="white", lw=0.6, zorder=4)
    ax_a.set_xscale("log")
    ax_a.set_yticks(range(n))
    ax_a.set_yticklabels(order, fontsize=8)
    ax_a.set_ylim(-0.6, n - 0.4)
    ax_a.set_xlabel("Cell-type proportion per sample (%, log scale)")
    ax_a.set_title(f"Compositional shift ({control} vs {case})", fontsize=10, pad=8)
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=control_color,
                markersize=7,
                label=f"{control} (mean)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=case_color,
                markersize=7,
                label=f"{case} (mean)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=_GREY_MUTED,
                markersize=5,
                alpha=0.5,
                label="per sample",
                linestyle="",
            ),
        ],
        loc="lower right",
        fontsize=6.5,
        frameon=False,
    )

    # --- Panel B: posterior inclusion probability ----------------------------
    incl = np.array(
        [float(eff.loc[ct, "inclusion_probability"]) if ct in eff.index else 0.0 for ct in order]
    )
    cred = np.array(
        [bool(eff.loc[ct, "credible_effect"]) if ct in eff.index else False for ct in order]
    )
    n_cred = int(cred.sum())
    for i in range(n):
        if i % 2 == 0:
            ax_b.axhspan(i - 0.5, i + 0.5, color=_ROW_BAND, zorder=0, lw=0)
    colors = [case_color if c else _GREY_BAR for c in cred]
    ax_b.barh(
        range(n), incl, color=colors, edgecolor="#8a9096", linewidth=0.3, height=0.62, zorder=2
    )
    ax_b.axvline(0.5, color=TEXT, ls=(0, (4, 3)), lw=0.9, alpha=0.6, zorder=1)
    ax_b.set_yticks(range(n))
    ax_b.set_yticklabels([])
    ax_b.set_ylim(-0.6, n - 0.4)
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_xlabel("scCODA posterior inclusion prob.")
    ax_b.set_title(f"scCODA credibility\n{n_cred}/{n} credible", fontsize=9, pad=8)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)

    fig.suptitle(
        "Compositional differential abundance (scCODA)",
        fontsize=11,
        fontweight="bold",
        y=1.0,
    )
    return fig


__all__ = [
    "prepare_milo_beeswarm",
    "sccoda_single_reference",
    "milo_beeswarm_order",
    "sccoda_composition_order",
    "plot_milo_beeswarm",
    "plot_sccoda_composition",
]
