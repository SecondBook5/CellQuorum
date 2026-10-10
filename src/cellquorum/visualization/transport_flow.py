"""Where the mass of each Normal state goes among the LE states, as one flow diagram.

Optimal transport between a donor's Normal and LE cells sends mass from every Normal cell to
LE cells. Summed over states it gives a state-to-state flow, the diagram moscot calls a Sankey.
A raw flow mostly restates two facts that need no transport at all: how many cells each state
has in each condition, and that nearby cells are matched. So the ribbon's width is the mass it
carries but its colour is the excess over what shuffled condition labels give (log2 observed over
expected, minus the same quantity under the shuffle), and a ribbon whose excess does not clear
the donor-level test is faded. A diagram of faded ribbons is the honest picture of a transport
that moves cells no more between states than geometry predicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellquorum.visualization.figstyle import apply_theme

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from matplotlib.figure import Figure

_INK, _MUTED, _NEUTRAL = "#202428", "#7A7A7A", "#B9BEC4"
_GAP = 0.04


def _layout(sizes: dict[str, float], order: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Top and bottom of each bar, stacked downward from 1 with a gap between bars."""
    top = 1.0
    out: dict[str, tuple[float, float]] = {}
    for name in order:
        out[name] = (top, top - sizes[name])
        top -= sizes[name] + _GAP
    return out


def transport_flow(
    flow: pd.DataFrame,
    *,
    excess: pd.DataFrame | None = None,
    significant: pd.DataFrame | None = None,
    source_order: Sequence[str] | None = None,
    target_order: Sequence[str] | None = None,
    source_label: str = "Normal",
    target_label: str = "Lymphedema",
    cbar_label: str = "Excess flow over shuffled labels (log2)",
    state_colors: dict[str, str] | None = None,
) -> Figure:
    """Ribbons from Normal states (left) to LE states (right).

    Args:
        flow: Rows = Normal state, columns = LE state; non-negative mass, summing to the same
            total in every donor-averaged table (normally 1).
        excess: Same shape as ``flow``; colours the ribbons (red: more flow than shuffled labels
            give, blue: less). Without it the ribbons are neutral grey.
        significant: Boolean, same shape; ribbons that are False are faded. Without it all
            ribbons are drawn at full strength.
        source_order, target_order: Stacking order of the bars; default the table order.
        source_label, target_label: Column headings; Normal is always the left side.
        cbar_label: Name of the colour scale.
        state_colors: Optional bar colour per state name.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path

    apply_theme("compact")
    if (flow.to_numpy(dtype=float) < 0).any():
        raise ValueError("flow has negative entries")
    for name, table in (("excess", excess), ("significant", significant)):
        if table is not None and (
            list(table.index) != list(flow.index) or list(table.columns) != list(flow.columns)
        ):
            raise ValueError(f"{name} must have the same rows and columns as flow")
    src = list(source_order) if source_order is not None else list(flow.index)
    dst = list(target_order) if target_order is not None else list(flow.columns)
    state_colors = state_colors or {}

    mass_of = flow.to_numpy(dtype=float)
    ri = {s: i for i, s in enumerate(flow.index)}
    ci = {t: j for j, t in enumerate(flow.columns)}
    row_size = {s: float(mass_of[ri[s]].sum()) for s in src}
    col_size = {t: float(mass_of[:, ci[t]].sum()) for t in dst}
    total = max(sum(row_size.values()), sum(col_size.values()))
    scale = (1.0 - _GAP * (max(len(src), len(dst)) - 1)) / total
    left = _layout({s: v * scale for s, v in row_size.items()}, src)
    right = _layout({t: v * scale for t, v in col_size.items()}, dst)

    excess_of = excess.to_numpy(dtype=float) if excess is not None else None
    sig_of = significant.to_numpy(dtype=bool) if significant is not None else None
    lim = float(np.abs(excess_of).max()) if excess_of is not None else 0.0
    lim = lim or 1.0
    norm = mpl.colors.Normalize(vmin=-lim, vmax=lim)
    cmap = mpl.colormaps["RdBu_r"]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.set_label("flow")
    bar = 0.035
    s_cursor = {s: left[s][0] for s in src}
    t_cursor = {t: right[t][0] for t in dst}
    for s in src:
        for t in dst:
            mass = float(mass_of[ri[s], ci[t]])
            if mass <= 0:
                continue
            h = mass * scale
            y0t, y1t = s_cursor[s], t_cursor[t]
            s_cursor[s] -= h
            t_cursor[t] -= h
            y0b, y1b = y0t - h, y1t - h
            x0, x1, xm = 0.0, 1.0, 0.5
            path = Path(
                [
                    (x0, y0t),
                    (xm, y0t),
                    (xm, y1t),
                    (x1, y1t),
                    (x1, y1b),
                    (xm, y1b),
                    (xm, y0b),
                    (x0, y0b),
                    (x0, y0t),
                ],
                [
                    Path.MOVETO,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.LINETO,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CLOSEPOLY,
                ],
            )
            colour = (
                cmap(norm(float(excess_of[ri[s], ci[t]]))) if excess_of is not None else _NEUTRAL
            )
            called = True if sig_of is None else bool(sig_of[ri[s], ci[t]])
            patch = PathPatch(
                path, facecolor=colour, edgecolor="none", alpha=0.85 if called else 0.28
            )
            patch.set_gid(f"flow|{s}|{t}")
            ax.add_patch(patch)
    for side, layout, x in (("source", left, -bar), ("target", right, 1.0)):
        for name, (top, bottom) in layout.items():
            ax.add_patch(
                Rectangle((x, bottom), bar, top - bottom, color=state_colors.get(name, _INK))
            )
            share = (row_size if side == "source" else col_size)[name] / total
            lab = ax.text(
                x - 0.015 if side == "source" else x + bar + 0.015,
                (top + bottom) / 2,
                f"{name}\n{share:.0%}",
                ha="right" if side == "source" else "left",
                va="center",
                fontsize=8.5,
                color=_INK,
            )
            lab.set_gid(f"{side}|{name}")
    ax.text(-bar / 2, 1.06, source_label, ha="center", fontsize=10, fontweight="bold", color=_INK)
    ax.text(
        1 + bar / 2, 1.06, target_label, ha="center", fontsize=10, fontweight="bold", color=_INK
    )
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.04 - _GAP * 0, 1.12)
    ax.axis("off")
    if excess is not None:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label(cbar_label, fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
    return fig


__all__ = ["transport_flow"]
