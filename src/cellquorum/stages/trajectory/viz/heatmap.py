"""Biology-agnostic condition-split pseudotime heatmap primitives.

Ported from the house Fiškin-recipe heatmap: per-condition pseudotime binning +
5-bin smoothing, per-gene 0-1 normalization (done by the caller), peak-bin gene
ordering on the combined profile, and a gridspec layout with stacked annotation
tracks (pseudotime gradient, a continuous score, a categorical state) above one
expression heatmap per condition column. No file I/O, no config, no biology.
"""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from cellquorum.visualization.figstyle import SEQUENTIAL_CMAP


def bin_masks(pt: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sort order over pt, bin id per cell in that sorted order).

    Bins by RANK (equal cell count per bin), not by raw pseudotime value.
    Real bug this fixes: dpt_pseudotime on a real cohort had 99.84% of cells
    packed into the first 5% of its raw value range, with a handful of
    straggler cells stretching the max out to 1.0. Binning by raw value
    (the previous ``np.linspace(pt.min(), pt.max(), n_bins+1)`` edges) left
    83-91/100 bins completely empty; the heatmap's forward-fill then painted
    those empty bins with whichever straggler landed last, producing large,
    confident-looking color blocks backed by a handful of outliers rather
    than real trend. Rank binning keeps every bin populated (when there are
    at least n_bins cells) regardless of how skewed the raw values are, and
    is robust to whichever pseudotime metric a caller picks -- the engine
    should not depend on one particular algorithm's value distribution being
    well-behaved.
    """
    pt = np.asarray(pt, dtype=float)
    order = np.argsort(pt)
    n = len(order)
    binid = np.clip((np.arange(n) * n_bins) // max(n, 1), 0, n_bins - 1).astype(int)
    return order, binid


def binned_profile(pt: np.ndarray, mat: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean expression per pseudotime bin, then 5-bin moving-average smoothing."""
    order, binid = bin_masks(pt, n_bins)
    m = np.asarray(mat, dtype=float)[order]
    prof = np.zeros((n_bins, m.shape[1]))
    for b in range(n_bins):
        sel = binid == b
        prof[b] = m[sel].mean(0) if sel.any() else (prof[b - 1] if b else 0)
    k = np.ones(5) / 5
    prof = np.vstack([np.convolve(prof[:, j], k, mode="same") for j in range(prof.shape[1])]).T
    return prof


def binned_score_track(pt: np.ndarray, score: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-bin mean of a continuous score (forward-filled over empty bins)."""
    order, binid = bin_masks(pt, n_bins)
    sc_o = np.asarray(score, dtype=float)[order]
    track = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = binid == b
        if sel.any():
            track[b] = np.nanmean(sc_o[sel])
        elif b:
            track[b] = track[b - 1]
    for b in range(1, n_bins):
        if np.isnan(track[b]):
            track[b] = track[b - 1]
    return track


def binned_categorical_track(pt: np.ndarray, codes: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-bin modal categorical code (forward-filled over empty bins)."""
    order, binid = bin_masks(pt, n_bins)
    st_o = np.asarray(codes)[order]
    track = np.full(n_bins, -1)
    for b in range(n_bins):
        sel = binid == b
        if sel.any():
            vals, cnts = np.unique(st_o[sel], return_counts=True)
            track[b] = vals[np.argmax(cnts)]
        elif b:
            track[b] = track[b - 1]
    return track.astype(int)


def binned_tracks(
    pt: np.ndarray, score: np.ndarray, state_codes: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated combo of binned_score_track + binned_categorical_track.

    Kept only so any external caller built against the old two-in-one signature
    keeps working; new code should call the two functions directly.
    """
    return (
        binned_score_track(pt, score, n_bins),
        binned_categorical_track(pt, state_codes, n_bins),
    )


def normalize_profiles_pooled(profiles: dict) -> dict:
    """0-1 normalize each gene using min/max pooled across ALL conditions.

    Real bug this fixes: normalizing each condition panel independently (each
    condition's own min/max) hides a true cross-condition magnitude
    difference -- a gene uniformly higher in one condition still gets
    stretched to look like the same 0-1 range as every other condition, which
    can make a real disease-vs-healthy effect invisible. Pooling the min/max
    across all conditions before normalizing keeps a real magnitude
    difference visible, matching how a published reference figure of this
    same plot type normalizes (globally, on the pooled data, not per panel).

    Args:
        profiles: {condition -> (n_bins, n_genes) RAW (not yet normalized)
            binned expression array}.

    Returns:
        {condition -> same-shape array, each gene scaled into [0, 1] using
        min/max pooled across every condition's array}.
    """
    stacked = np.concatenate(list(profiles.values()), axis=0)
    gmin = stacked.min(0)
    span = stacked.max(0) - gmin
    span = np.where(span > 1e-9, span, 1.0)
    return {c: (p - gmin) / span for c, p in profiles.items()}


def peak_bin_order(combined_profile: np.ndarray) -> np.ndarray:
    """Gene order by peak (argmax) bin of the combined profile."""
    return np.argsort(np.argmax(np.asarray(combined_profile), axis=0))


def grouped_peak_order(
    combined_profile: np.ndarray, gene_labels: list[str], groups: dict[str, list[str]]
) -> tuple[np.ndarray, list[str]]:
    """Gene order that keeps each declared group together, in the declared group order.

    Within a group genes run by peak bin (earliest first), so the groups carry the biology and the
    peak order still shows progression along pseudotime. Group members that are not columns of the
    profile are ignored. Returns (column indices to display, group name per displayed row).
    """
    peak = np.argmax(np.asarray(combined_profile), axis=0)
    column_of = {g: i for i, g in enumerate(gene_labels)}
    order: list[int] = []
    row_groups: list[str] = []
    seen: set[int] = set()
    for name, members in groups.items():
        cols = [column_of[g] for g in members if g in column_of and column_of[g] not in seen]
        cols.sort(key=lambda j: (peak[j], j))
        seen.update(cols)
        order.extend(cols)
        row_groups.extend([name] * len(cols))
    return np.asarray(order, dtype=int), row_groups


def _draw_gene_groups(
    expr_axes: list[Axes], row_groups: list[str], n_bins: int, *, fontsize: float = 7.5
) -> None:
    """Divide the gene rows into their groups: a white divider across every panel at each group
    boundary, and a bracket plus the group name in the left margin of the first panel."""
    runs: list[tuple[str, int, int]] = []
    for i, name in enumerate(row_groups):
        if runs and runs[-1][0] == name:
            runs[-1] = (name, runs[-1][1], i)
        else:
            runs.append((name, i, i))
    for ax in expr_axes:
        for _, start, _ in runs[1:]:
            ax.plot(
                [-0.5, n_bins - 0.5],
                [start - 0.5, start - 0.5],
                color="white",
                lw=2.2,
                scalex=False,
                scaley=False,
                zorder=3,
            )
    first = expr_axes[0]
    span = n_bins
    for name, start, end in runs:
        first.plot(
            [-0.5 - 0.02 * span] * 2,
            [start - 0.4, end + 0.4],
            color="#444444",
            lw=1.2,
            clip_on=False,
            scalex=False,
            scaley=False,
        )
        first.text(
            -0.5 - 0.035 * span,
            (start + end) / 2,
            name,
            ha="right",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            clip_on=False,
        )


def _leader_line_labels(ax: Axes, gene_labels_ordered: list[str], *, fontsize: float = 6.5) -> None:
    """Evenly-spaced right-margin gene labels, each on a thin leader line back
    to its real row, instead of a plain y-tick list.

    Drawn in the axes' own data coordinates just past its right edge, with
    ``clip_on=False`` so matplotlib does not clip them off -- the caller must
    leave enough right margin (see ``condition_split_heatmap``'s gridspec) for
    them to land in.
    """
    n = len(gene_labels_ordered)
    if n == 0:
        return
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    line_end_x = x1 + 0.06 * span
    text_x = x1 + 0.09 * span
    label_y = np.linspace(0, n - 1, n)
    for i, label in enumerate(gene_labels_ordered):
        ax.plot(
            [x1, line_end_x],
            [i, label_y[i]],
            color="#9a9a9a",
            lw=0.5,
            clip_on=False,
            zorder=1,
        )
        ax.text(
            text_x,
            label_y[i],
            label,
            fontsize=fontsize,
            fontstyle="italic",
            va="center",
            ha="left",
            clip_on=False,
        )

    # Outside leader lines must not autoscale the image shorter than its tracks.
    ax.set_xlim(x0, x1)


def condition_split_heatmap(
    profiles: dict,
    score_tracks: dict,
    categorical_tracks: list,
    gene_labels: list,
    gene_order: np.ndarray,
    *,
    condition_order: list,
    expr_cmap: str = SEQUENTIAL_CMAP,
    title: str = "",
    gene_group_labels: list[str] | None = None,
) -> Figure:
    """Build the stacked-annotation, condition-split expression heatmap.

    Args:
        profiles: {condition -> (n_bins, n_genes) binned expression array}.
        score_tracks: {condition -> (n_bins,) continuous score array}, or an
            empty dict / all-NaN to omit the "Score" row entirely.
        categorical_tracks: zero or more stacked categorical annotation rows,
            each a dict with keys ``label`` (row title), ``cats`` (ordered
            category names), ``colors`` (one hex per category, same order),
            ``tracks`` ({condition -> (n_bins,) int code array, -1 = none}),
            and ``present_codes`` (which codes actually occur, for the legend).
            Rendered top-to-bottom in list order, each with its own legend.
        gene_labels: gene name per column of ``profiles``.
        gene_order: column order to display (e.g. from ``peak_bin_order``).
        condition_order: which conditions to render, left to right.
        expr_cmap: colormap for the expression heatmap.
        title: figure suptitle.
        gene_group_labels: optional group name per displayed row (same length as
            ``gene_order``, each group a contiguous run). Draws dividers between groups and
            a bracketed group name in the left margin.

    Returns:
        The assembled Figure.
    """
    if gene_group_labels is not None and len(gene_group_labels) != len(gene_order):
        raise ValueError(
            f"gene_group_labels has {len(gene_group_labels)} entries for "
            f"{len(gene_order)} displayed genes."
        )
    n_cols = len(condition_order)
    n_bins = next(iter(profiles.values())).shape[0]

    any_score = bool(score_tracks) and any(
        not np.all(np.isnan(score_tracks[c])) for c in condition_order
    )

    rows = ["pt"]
    if any_score:
        rows.append("score")
    rows += [f"cat{i}" for i in range(len(categorical_tracks))]
    rows.append("expr")
    n_ann = len(rows) - 1

    hm_h = max(3.0, 0.30 * len(gene_labels))
    fig = Figure(figsize=(5.5 * n_cols, hm_h + 1.0 + 0.22 * len(categorical_tracks)))
    height_ratios = [0.28] * n_ann + [hm_h]
    gs = gridspec.GridSpec(
        len(rows),
        n_cols,
        height_ratios=height_ratios,
        hspace=0.08,
        wspace=0.06,
        left=0.21 if gene_group_labels else 0.14,
        right=0.82,
        top=0.90,
        bottom=0.08,
    )

    score_all = np.concatenate([score_tracks[c] for c in condition_order]) if any_score else None
    if score_all is not None and np.isfinite(score_all).any():
        s_vmin, s_vmax = np.nanpercentile(score_all[np.isfinite(score_all)], [2, 98])
    else:
        s_vmin, s_vmax = 0.0, 1.0

    pt_grad = np.linspace(0, 1, n_bins)[None, :]
    im_score = None
    im_expr = None
    # x-axis tick positions/labels shared by every row -- the point of this
    # heatmap is progression along pseudotime, so leaving the axis blank
    # (the prior behaviour) hid the one thing a reader most needs to read off.
    tick_pos = [0, (n_bins - 1) * 0.25, (n_bins - 1) * 0.5, (n_bins - 1) * 0.75, n_bins - 1]
    tick_lab = ["0%", "25%", "50%", "75%", "100%"]
    for j, c in enumerate(condition_order):
        r = 0
        ax_pt = fig.add_subplot(gs[r, j])
        ax_pt.imshow(pt_grad, aspect="auto", cmap="Spectral_r", vmin=0, vmax=1)
        ax_pt.set_xticks([])
        ax_pt.set_yticks([])
        ax_pt.set_title(str(c), fontsize=11, fontweight="bold", pad=6)
        if j == 0:
            ax_pt.set_ylabel("Pseudotime", rotation=0, ha="right", va="center", fontsize=7.5)
        r += 1
        if any_score:
            ax_s = fig.add_subplot(gs[r, j])
            im_score = ax_s.imshow(
                score_tracks[c][None, :], aspect="auto", cmap="Reds", vmin=s_vmin, vmax=s_vmax
            )
            ax_s.set_xticks([])
            ax_s.set_yticks([])
            if j == 0:
                ax_s.set_ylabel("Score", rotation=0, ha="right", va="center", fontsize=7.5)
            r += 1
        for cat_track in categorical_tracks:
            cats, colors = cat_track["cats"], cat_track["colors"]
            cmap = ListedColormap(colors) if colors else None
            norm = BoundaryNorm(np.arange(-0.5, len(cats) + 0.5), len(cats)) if cats else None
            ax_ct = fig.add_subplot(gs[r, j])
            ax_ct.imshow(cat_track["tracks"][c][None, :], aspect="auto", cmap=cmap, norm=norm)
            ax_ct.set_xticks([])
            ax_ct.set_yticks([])
            if j == 0:
                ax_ct.set_ylabel(
                    cat_track["label"], rotation=0, ha="right", va="center", fontsize=7.5
                )
            r += 1
        ax = fig.add_subplot(gs[r, j])
        im_expr = ax.imshow(
            profiles[c][:, gene_order].T,
            aspect="auto",
            cmap=expr_cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=7.5)
        ax.set_xlabel("pseudotime (percent of range)", fontsize=8)
        ax.set_yticks([])
        if j == n_cols - 1:
            last_expr_ax = ax

    # Gene names as evenly-spaced, leader-lined labels in the right margin
    # after the LAST condition panel (matching the reference layout this was
    # modeled on), rather than a plain y-tick list on the first panel -- past
    # a few dozen genes a tick list reads as a cramped, illegible block.
    _leader_line_labels(last_expr_ax, [gene_labels[i] for i in gene_order])
    if gene_group_labels:
        expr_axes = [a for a in fig.axes if a.get_xlabel().startswith("pseudotime")]
        _draw_gene_groups(expr_axes, gene_group_labels, n_bins)

    if im_score is not None:
        cax_g = fig.add_axes((0.93, 0.74, 0.014, 0.12))
        cb_g = fig.colorbar(im_score, cax=cax_g, ticks=[s_vmin, s_vmax])
        cb_g.ax.set_yticklabels(["Min", "Max"], fontsize=7)
        cb_g.set_label("Score", fontsize=7.5)
    if im_expr is not None:
        cax = fig.add_axes((0.93, 0.40, 0.014, 0.24))
        fig.colorbar(im_expr, cax=cax, label="scaled expr")

    # One stacked legend block per categorical track, top to bottom.
    legend_y = 0.32
    for cat_track in categorical_tracks:
        cats, colors, present = (
            cat_track["cats"],
            cat_track["colors"],
            cat_track.get("present_codes") or [],
        )
        handles = [
            Patch(facecolor=colors[i], edgecolor="none", label=cats[i])
            for i in present
            if 0 <= i < len(cats)
        ]
        if not handles:
            continue
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.90, legend_y),
            frameon=False,
            fontsize=6.8,
            title=cat_track["label"],
            title_fontsize=7.5,
            handlelength=1.0,
            handleheight=1.0,
            labelspacing=0.35,
        )
        legend_y -= 0.05 + 0.028 * len(handles)

    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    return fig


__all__ = [
    "bin_masks",
    "binned_profile",
    "binned_score_track",
    "binned_categorical_track",
    "binned_tracks",
    "normalize_profiles_pooled",
    "peak_bin_order",
    "grouped_peak_order",
    "condition_split_heatmap",
]
