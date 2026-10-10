"""How far apart a donor's two conditions sit, against what shuffled condition labels give.

An optimal-transport (earth mover's) distance between a donor's Normal cells and the same
donor's LE cells is always positive: two samples of the same population are never identical, and
the distance grows as cell numbers fall. A bare distance therefore says nothing. Each donor is
drawn with the distance the same cells give after their condition labels are shuffled within the
donor (grey: mean to 95th percentile), so a marker is only read against its own null. A donor
whose distance does not clear that null is an open marker. The donor is the unit; cells are never
pooled across donors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellquorum.visualization.figstyle import apply_theme

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from matplotlib.figure import Figure

_INK, _MUTED, _ACCENT, _NULL = "#202428", "#7A7A7A", "#C1121F", "#B9BEC4"
_REQUIRED = ("state", "donor", "w2", "null_mean", "null_p95", "p_perm")


def transport_distance(
    table: pd.DataFrame,
    *,
    state_order: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> Figure:
    """One row per donor, grouped by state.

    Args:
        table: One row per donor and state; columns ``state``, ``donor``, ``w2`` (observed
            distance), ``null_mean`` and ``null_p95`` (the same distance after within-donor label
            shuffles), ``p_perm`` (permutation p-value). ``n_normal`` and ``n_le`` (cells per
            condition) are written beside each donor when present.
        state_order: Order of the state blocks; defaults to first appearance in ``table``.
        alpha: p-value below which a donor's marker is filled; above it the marker is open.
    """
    import matplotlib.pyplot as plt

    missing = [c for c in _REQUIRED if c not in table.columns]
    if missing:
        raise ValueError(f"transport_distance needs columns {missing}")
    apply_theme("compact")
    states = list(state_order) if state_order is not None else list(dict.fromkeys(table["state"]))
    has_counts = {"n_normal", "n_le"} <= set(table.columns)

    layout: list[tuple[str, str | None]] = []
    for state in states:
        layout.append((state, None))
        layout.extend((state, str(d)) for d in table.loc[table["state"] == state, "donor"])
    n = len(layout)
    fig, ax = plt.subplots(figsize=(5.4, 0.28 * n + 1.3))
    ax.set_label("transport")
    lo = float(min(table["null_mean"].min(), table["w2"].min()))
    hi = float(max(table["null_p95"].max(), table["w2"].max()))
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad + (0.22 * (hi - lo) if has_counts else 0))
    right = ax.get_xlim()[1]

    labels = []
    for i, (state, donor) in enumerate(layout):
        if donor is None:
            block = table[table["state"] == state]
            above = int((block["p_perm"] < alpha).sum())
            head = ax.text(
                ax.get_xlim()[0],
                i,
                f"{state}  {above} of {len(block)} donors above the shuffled-label null",
                ha="left",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=_INK,
            )
            head.set_gid("state")
            labels.append("")
            continue
        row = table[(table["state"] == state) & (table["donor"].astype(str) == donor)].iloc[0]
        called = float(row["p_perm"]) < alpha
        ax.plot(
            [row["null_mean"], row["null_p95"]],
            [i, i],
            color=_NULL,
            lw=3.2,
            zorder=1,
            solid_capstyle="butt",
        )
        null = ax.scatter([row["null_mean"]], [i], s=16, color=_NULL, zorder=2)
        null.set_gid(f"null|{state}|{donor}")
        pt = ax.scatter(
            [row["w2"]],
            [i],
            s=46,
            facecolors=_ACCENT if called else "none",
            edgecolors=_ACCENT,
            linewidths=1.4,
            zorder=3,
        )
        pt.set_gid(f"obs|{state}|{donor}")
        labels.append(donor)
        if has_counts:
            cells = ax.text(
                right,
                i,
                f"{int(row['n_normal'])} | {int(row['n_le'])}",
                ha="right",
                va="center",
                fontsize=7.5,
                color=_MUTED,
            )
            cells.set_gid(f"cells|{state}|{donor}")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    suffix = "\nright: cells Normal | LE" if has_counts else ""
    ax.set_xlabel(
        f"Wasserstein-2 distance, Normal vs LE\n"
        f"(filled: p < {alpha:g}; grey: shuffled-label null){suffix}",
        fontsize=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


__all__ = ["transport_distance"]
