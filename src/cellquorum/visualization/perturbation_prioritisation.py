"""Which populations does a condition change most: a donor-aware distance beside Augur separability.

Two answers to one question, drawn on the same rows because each fails where the other holds.
Augur's AUC says how well a classifier separates the two conditions from cells; it ignores donors,
so a population with a few donors that differ for other reasons can score high (and a permutation
null, drawn beside every AUC, shows what an uninformative label scores). A donor-aware distance
(scDist, with donor and sample as random effects) treats the donor as the replicate, so a
difference carried by one donor does not count. A population where the AUC is high and the
distance is zero is a population Augur is reading donor structure in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellquorum.visualization.figstyle import apply_theme

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

_INK, _MUTED, _ACCENT, _NULL = "#202428", "#7A7A7A", "#C1121F", "#B9BEC4"


def perturbation_prioritisation(
    scdist: pd.DataFrame,
    augur: pd.DataFrame | None = None,
    *,
    highlight: str | None = None,
    alpha: float = 0.05,
) -> Figure:
    """One row per population, largest donor-aware distance first.

    Args:
        scdist: Index = population; columns ``dist``, ``ci_low``, ``ci_high``, ``p``.
        augur: Index = population; columns ``auc`` and ``null_auc`` (the permutation-mode AUC).
            Populations Augur skipped (too few cells) are simply absent and are written as
            "not run" rather than drawn at an AUC of one half.
        highlight: Population drawn in bold and in the accent colour.
        alpha: p-value below which a distance is a filled marker; above it the marker is open.
    """
    import matplotlib.pyplot as plt

    apply_theme("compact")
    order = list(scdist.sort_values("dist", ascending=False).index)
    n = len(order)
    panels = 2 if augur is not None else 1
    fig, axes = plt.subplots(
        1, panels, figsize=(3.4 * panels + 1.6, 0.34 * n + 1.4), sharey=True, squeeze=False
    )
    ax = axes[0][0]
    ax.set_label("scdist")
    y = list(range(n))
    for i, pop in enumerate(order):
        row = scdist.loc[pop]
        colour = _ACCENT if pop == highlight else _INK
        called = float(row["p"]) < alpha
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=colour, lw=1.4, zorder=2)
        pt = ax.scatter(
            [row["dist"]],
            [i],
            s=46,
            facecolors=colour if called else "none",
            edgecolors=colour,
            linewidths=1.4,
            zorder=3,
        )
        pt.set_gid(f"dist|{pop}")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_ylim(n - 0.5, -0.5)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold" if label.get_text() == highlight else "normal")
    ax.set_xlabel(f"Donor-aware distance, LE vs Normal\n(filled: p < {alpha:g})", fontsize=8.5)
    ax.set_xlim(left=0)
    if augur is not None:
        bx = axes[0][1]
        bx.set_label("augur")
        for i, pop in enumerate(order):
            if pop not in augur.index:
                bx.text(0.5, i, "not run", ha="left", va="center", fontsize=7.5, color=_MUTED)
                continue
            a = augur.loc[pop]
            bx.plot([a["null_auc"], a["auc"]], [i, i], color=_NULL, lw=1.2, zorder=1)
            null = bx.scatter([a["null_auc"]], [i], s=26, color=_NULL, zorder=2)
            null.set_gid(f"null|{pop}")
            pt = bx.scatter(
                [a["auc"]], [i], s=46, color=_ACCENT if pop == highlight else _INK, zorder=3
            )
            pt.set_gid(f"auc|{pop}")
        bx.axvline(0.5, color=_MUTED, lw=0.8, ls=":")
        bx.set_xlim(0.4, 1.0)
        bx.set_xlabel("Augur AUC (grey: permutation null)", fontsize=8.5)
    for a_ in axes[0]:
        a_.spines["top"].set_visible(False)
        a_.spines["right"].set_visible(False)
    return fig


__all__ = ["perturbation_prioritisation"]
