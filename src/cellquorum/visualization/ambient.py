"""Ambient-RNA (SoupX) correction QC figure.

SoupX estimates, per library, the fraction of each cell's counts that came from
the ambient "soup" of free mRNA rather than from the cell itself, then subtracts
it. That per-library fraction (rho) is the one number that says whether the
correction mattered and whether any library is an outlier — so it belongs in a
figure rather than buried in `uns`.

Reading it: a few percent is normal for fresh dissociated tissue. A library well
above its cohort is the one to look at first when a downstream result looks
lineage-confused, because ambient contamination is what puts keratin in a
fibroblast and haemoglobin in everything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from cellquorum.visualization.figstyle import apply_cellquorum_theme

if TYPE_CHECKING:
    from collections.abc import Mapping

_INK = "#1a1a1a"
_MUTED = "#6b6b6b"
_FAINT = "#c8c8cc"
# Above this the correction is doing enough work that it is worth a second look
# at the library, not a default to filter on.
_ATTENTION_RHO = 0.10


def contamination_figure(
    fractions: Mapping[str, float],
    *,
    condition_of: Mapping[str, str] | None = None,
    donor_of: Mapping[str, str] | None = None,
    case: str | None = None,
    control: str | None = None,
    condition_colors: Mapping[str, str] | None = None,  # noqa: ARG001 — kept for API compat
    attention_rho: float = _ATTENTION_RHO,
    title: str | None = None,
) -> Figure | None:
    """Ambient-correction QC as a donor x arm matrix. None when nothing to plot.

    Deliberately compact. A paired cohort's contamination is a donor x arm grid,
    so that is what gets drawn: one column per donor, one row per arm, each cell
    the fraction removed. Every value is shown, the pairing is implicit in the
    column, and an arm-level effect — the thing that would confound every
    downstream case/control comparison — would appear as one row reading
    systematically darker than the other.

    The previous version ranked all libraries on a lollipop axis. That spent an
    18-row canvas on a ~3-point spread, hid the pairing, and made the reader
    reconstruct the donor structure by reading label suffixes.

    Falls back to a single ordered row when donors/arms are not resolvable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    clean = {
        str(k): float(v) * 100.0
        for k, v in (fractions or {}).items()
        if v is not None and np.isfinite(float(v))
    }
    if not clean:
        return None

    # Resolve the donor x arm grid when possible.
    grid: dict[str, dict[str, float]] = {}
    if donor_of and condition_of:
        for library, value in clean.items():
            donor, arm = donor_of.get(library), condition_of.get(library)
            if donor and arm:
                grid.setdefault(str(donor), {})[str(arm)] = value
    arms: list[str] = []
    if grid:
        if case and control:
            arms = [control, case]
        else:
            arms = sorted({a for row in grid.values() for a in row})
    matrixed = bool(grid) and len(arms) == 2

    values = np.array(list(clean.values()))
    median = float(np.median(values))
    norm = Normalize(vmin=0.0, vmax=max(values.max(), attention_rho * 100.0 * 0.6))
    cmap = plt.get_cmap("YlOrBr")

    if matrixed:
        # Order donors by their own mean so any gradient is left-to-right, and a
        # single bad donor is at one end rather than buried mid-grid.
        donors = sorted(grid, key=lambda d: -np.mean(list(grid[d].values())))
        fig, ax = plt.subplots(figsize=(0.62 * len(donors) + 2.4, 2.65))
        for col, donor in enumerate(donors):
            for row, arm in enumerate(arms):
                value = grid[donor].get(arm)
                if value is None:
                    ax.add_patch(
                        plt.Rectangle(
                            (col, row), 1, 1, facecolor="#f4f4f6", edgecolor="white", lw=1.4
                        )
                    )
                    continue
                ax.add_patch(
                    plt.Rectangle(
                        (col, row), 1, 1, facecolor=cmap(norm(value)), edgecolor="white", lw=1.4
                    )
                )
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=_readable_on(cmap(norm(value))),
                )
        ax.set_xlim(0, len(donors))
        ax.set_ylim(0, len(arms))
        ax.set_xticks(np.arange(len(donors)) + 0.5)
        ax.set_xticklabels(donors, fontsize=8.5)
        ax.set_yticks(np.arange(len(arms)) + 0.5)
        ax.set_yticklabels(arms, fontsize=9)
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        footer = (
            f"% of counts removed as ambient · median {median:.1f}% · " f"n={len(clean)} libraries"
        )
        # Whether contamination tracks the ARM is the confound question, so answer
        # it in the figure rather than leaving it to be eyeballed.
        paired = [
            (grid[d][arms[0]], grid[d][arms[1]])
            for d in donors
            if arms[0] in grid[d] and arms[1] in grid[d]
        ]
        if len(paired) >= 3:
            lo = np.array([p[0] for p in paired])
            hi = np.array([p[1] for p in paired])
            detail = f"{int((hi > lo).sum())}/{len(paired)} higher in {arms[1]}"
            try:
                from scipy import stats

                detail += f", p={float(stats.wilcoxon(hi, lo).pvalue):.2f}"
            except Exception:  # noqa: BLE001
                pass
            footer += f"\nno arm effect: {detail}"
        ax.set_title(title or "Ambient RNA correction (SoupX)", fontsize=10.5, loc="left", pad=8)
        ax.text(0, 1.0 + 0.13, "", transform=ax.transAxes)
        fig.text(
            0.012,
            -0.02,
            footer,
            fontsize=7.5,
            color=_MUTED,
            ha="left",
            va="top",
            linespacing=1.4,
            transform=ax.transAxes,
        )
    else:
        # Fallback: one ordered row, still compact.
        order = sorted(clean, key=clean.get, reverse=True)
        fig, ax = plt.subplots(figsize=(0.52 * len(order) + 1.8, 2.0))
        for col, library in enumerate(order):
            value = clean[library]
            ax.add_patch(
                plt.Rectangle(
                    (col, 0), 1, 1, facecolor=cmap(norm(value)), edgecolor="white", lw=1.4
                )
            )
            ax.text(
                col + 0.5,
                0.5,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=_readable_on(cmap(norm(value))),
            )
        ax.set_xlim(0, len(order))
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(len(order)) + 0.5)
        ax.set_xticklabels(order, fontsize=7.5, rotation=90)
        ax.set_yticks([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title or "Ambient RNA correction (SoupX)", fontsize=10.5, loc="left", pad=8)
        fig.text(
            0.012,
            -0.06,
            f"% of counts removed as ambient · median {median:.1f}% · " f"n={len(clean)} libraries",
            fontsize=7.5,
            color=_MUTED,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    return fig


def _readable_on(rgba: tuple) -> str:
    """Ink or white, whichever is legible on this cell colour."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    return _INK if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.6 else "white"


def apply_theme() -> None:
    apply_cellquorum_theme()


__all__ = ["apply_theme", "contamination_figure"]
