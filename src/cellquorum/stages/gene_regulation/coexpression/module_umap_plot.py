"""Publication-grade re-plot of the hdWGCNA co-expression module UMAP.

hdWGCNA's built-in ModuleUMAPPlot overprints hub labels into an unreadable smear
and floats tiny blobs in a sea of white. This renders the SAME data (the
module_umap_{tag}.csv coordinate table hdWGCNA writes) as a clean Nature-style
panel: house categorical colors, crisp points, hub genes ringed and
italic-labelled with light-halo repulsion, a clean legend, whitespace trimmed to
the data.

Reads only the CSV hdWGCNA already produced (gene, UMAP1, UMAP2, module, color,
hub, kME) -> no field, no re-run. Called as a post-step of hdwgcna_modules (see
workflow) and as a standalone CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from cellquorum.visualization import figstyle

logger = logging.getLogger(__name__)


def _build_figure(
    df: pd.DataFrame,
    *,
    mods: list[str],
    mod_color: dict[str, str],
    top_hub: dict[str, str],
    tag: str,
) -> tuple[Figure, Legend, list[Text]]:
    """Draw the figure and return it (with its legend + hub-label texts) unsaved.

    Split out from :func:`plot_module_umap` so a test can inspect the legend
    and hub-label positions directly -- ``plot_module_umap`` itself closes the
    figure the moment it's written (``figstyle.save_figure``'s job, and
    correctly so), which leaves nothing to inspect afterward.
    """
    hubs = df[df["hub"] == "hub"]

    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Non-hub points
    for m in mods:
        sub = df[(df["module"].astype(str) == m) & (df["hub"] != "hub")]
        ax.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=7,
            c=mod_color[m],
            alpha=0.55,
            edgecolors="none",
            zorder=2,
            rasterized=True,
        )

    # Hub points with ring
    if len(hubs):
        ax.scatter(
            hubs["UMAP1"],
            hubs["UMAP2"],
            s=34,
            c=[mod_color[str(m)] for m in hubs["module"]],
            edgecolors="#222",
            linewidth=0.6,
            zorder=4,
        )

    def _label(m: str) -> str:
        g = top_hub.get(m, "")
        return f"{m} — {g}" if g else m

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=mod_color[m],
            markersize=8,
            markeredgecolor="#222",
            markeredgewidth=0.5,
            label=_label(m),
        )
        for m in mods
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7.5,
        frameon=True,
        title="Co-expression module",
        title_fontsize=8.5,
        borderpad=0.4,
        labelspacing=0.3,
    )
    leg.get_frame().set_edgecolor("#ccc")

    # Hub labels: top 3/module, italic, thin white halo, repelled if adjustText is available
    labels = []
    for m in mods:
        sub = hubs[hubs["module"].astype(str) == m]
        for _, r in sub.sort_values("kME", ascending=False).head(3).iterrows():
            t = ax.text(
                r["UMAP1"],
                r["UMAP2"],
                r["gene"],
                fontsize=7.5,
                color=figstyle.TEXT,
                zorder=6,
                fontstyle="italic",
            )
            t.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
            labels.append(t)
    try:
        from adjustText import adjust_text

        adjust_text(
            labels,
            ax=ax,
            only_move={"points": "xy", "text": "xy"},
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.4),
        )
    except Exception:
        pass  # labels still render, just un-repelled

    # Trim whitespace to the data extent + 6% pad
    xpad = (df.UMAP1.max() - df.UMAP1.min()) * 0.06
    ypad = (df.UMAP2.max() - df.UMAP2.min()) * 0.06
    ax.set_xlim(df.UMAP1.min() - xpad, df.UMAP1.max() + xpad)
    ax.set_ylim(df.UMAP2.min() - ypad, df.UMAP2.max() + ypad)
    ttl = f"Co-expression module architecture{(', ' + tag.replace('_', chr(8594))) if tag else ''}"
    ax.set_title(ttl, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#999")
    ax.spines["bottom"].set_color("#999")
    plt.tight_layout()

    # adjustText's own `objects=` obstacle-avoidance doesn't work for a
    # legend: it fetches the legend's bbox via get_window_extent (display /
    # pixel coordinates, since a loc="upper left" legend isn't data-
    # positioned), then feeds those pixel values through ax.convert_xunits,
    # which is a no-op for a plain numeric axis -- the "obstacle" ends up
    # miles away in data-space and never actually blocks anything (confirmed
    # directly: labels still landed under the legend with objects=[leg]
    # passed). Real fix: after the repulsion, physically push away any hub
    # label whose rendered bbox still overlaps the legend's.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = leg.get_window_extent(renderer)
    # Module hub genes cluster together in UMAP space by construction, so
    # more than one can sit near the legend's corner at once. A label
    # already just outside the legend box (not "overlapping" by a strict
    # bbox test) can still end up collided into once a neighbor gets pushed
    # toward it -- expanding the exclusion zone by a bit more than one
    # label's height catches those near-misses too, and every label caught
    # in the zone (not just the ones strictly overlapping the legend) is
    # laid out together in one packed row.
    pad = 14.0
    zone = Bbox.from_extents(
        legend_bbox.x0 - pad, legend_bbox.y0 - pad, legend_bbox.x1 + pad, legend_bbox.y1 + pad
    )
    conflicted = [t for t in labels if t.get_window_extent(renderer).overlaps(zone)]
    if conflicted:
        # Placing the packed row just below the legend (the first fix) still
        # left it INSIDE the plotted data's own coordinate range -- module
        # hub genes cluster together, so the real scatter points for this
        # exact module sit right there too, and the row could land on top of
        # one. Extending the y-axis to add headroom above every real data
        # point, then placing the row in that new strip (beside the legend,
        # not under it), guarantees it can only ever overlap whitespace.
        cur_ylim = ax.get_ylim()
        headroom = 0.16 * (cur_ylim[1] - cur_ylim[0])
        ax.set_ylim(cur_ylim[0], cur_ylim[1] + headroom)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = leg.get_window_extent(renderer)
        axes_bbox = ax.get_window_extent(renderer)
        inv = ax.transData.inverted()
        # adjustText leaves every label centered on its anchor (ha="center",
        # va="center"), not left/baseline-aligned -- set_position places the
        # MIDDLE of the glyph, so the left edge lands at position - width/2,
        # not at position itself. Missing this shifted every placed label
        # left by half its own width, under-spacing each pair by a
        # different, width-dependent amount (confirmed directly: three
        # labels placed in sequence came out short by 14px, 24px, 17px --
        # each exactly half that label's own rendered width, not a constant).
        next_x_disp = legend_bbox.x1 + 10
        row_y_disp = axes_bbox.y1 - 10
        for t in conflicted:
            bbox = t.get_window_extent(renderer)
            anchor_x = next_x_disp + bbox.width / 2
            anchor_y = row_y_disp - bbox.height / 2
            t.set_position(inv.transform((anchor_x, anchor_y)))
            next_x_disp += bbox.width + 6

    return fig, leg, labels


def plot_module_umap(
    umap_csv: str | Path,
    out_dir: str | Path,
    *,
    tag: str = "",
    palette: list[str] | None = None,
) -> list[Path]:
    """Render the clean module UMAP from hdWGCNA's module_umap CSV.

    Args:
        umap_csv: Path to the module_umap CSV (gene, UMAP1, UMAP2, module, color, hub, kME).
        out_dir: Output directory for the figures.
        tag: Optional tag for title and filename (e.g. "AAH_AIS").
        palette: Color palette for modules. Defaults to figstyle.CATEGORICAL_PALETTE.

    Returns:
        List of written figure paths (PNG+PDF), or [] when the CSV has no non-grey genes.
    """
    figstyle.set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(umap_csv)
    df = df[df["module"].astype(str).str.lower() != "grey"].copy()
    if len(df) == 0:
        logger.info(f"no non-grey genes in {umap_csv}, skipping figure")
        return []

    has_hub = "hub" in df.columns
    if not has_hub:
        df["hub"] = "other"
    if "kME" not in df.columns:
        df["kME"] = 0.5

    # Resolve palette and assign module colors in sorted order
    if palette is None:
        palette = figstyle.CATEGORICAL_PALETTE
    mods = sorted(df["module"].astype(str).unique())
    mod_color = {m: palette[i % len(palette)] for i, m in enumerate(mods)}

    hubs = df[df["hub"] == "hub"]
    top_hub = (
        hubs.sort_values("kME", ascending=False).groupby("module").first()["gene"].to_dict()
        if len(hubs)
        else {}
    )

    fig, _leg, _labels = _build_figure(df, mods=mods, mod_color=mod_color, top_hub=top_hub, tag=tag)

    stem = f"module_umap_pub_{tag}" if tag else "module_umap_pub"
    return figstyle.save_figure(fig, out_dir, stem)
