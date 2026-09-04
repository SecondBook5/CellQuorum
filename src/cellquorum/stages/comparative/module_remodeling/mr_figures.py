"""Figures for the module-remodeling stage.

``state_scoring`` computes per-cell module activity and draws nothing, so before
this module the only way to *see* a curated module set was to open the CSV. Three
panels close that, flagship first:

1. ``plot_module_dotgrid`` — module x group dot-grid. Colour is the mixed-model
   condition effect, dot area is ``-log10(FDR)``, a dark ring marks FDR < 0.05,
   and an open cross marks a cell that could not be tested at all. That last mark
   matters: a blank cell in a dot-grid reads as "no effect", which is the opposite
   of "underpowered", and the two are indistinguishable without it.
2. ``plot_contrast_index`` — the signed identity axis, by group and condition.
3. ``plot_permanova`` — multivariate condition effect (R²) per group.

The stage's fourth question — how the programs are co-organized — is drawn by
:mod:`cellquorum.visualization.program_correlation` instead. It used to be a
heatmap of ``scores.corr()`` here, and that panel showed a coefficient with no
unit, no significance and no disclosure of which pairs share genes; the shared
module draws the tested table instead, and is reusable outside this stage.

Every panel is driven entirely by the stage's own CSVs, so a figure can be redrawn
from a finished run directory without recomputing a statistic. Colour carries
signed magnitude through a diverging colormap rather than a categorical palette,
which is both the right encoding for an effect size and the reason these panels
need no categorical-palette CVD gate: the only categorical colours used anywhere
here are the two house condition colours.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from cellquorum.visualization import figstyle

# Dot area (points²) spanned across the -log10(FDR) range. The floor is visible
# rather than invisible on purpose: a tested-but-null cell must still read as
# tested, which is what distinguishes it from the cross drawn for an untestable one.
_DOT_AREA_MIN = 18.0
_DOT_AREA_MAX = 340.0

_SIG_RING = figstyle.TEXT
_NULL_RING = "#B7BDC4"
_UNTESTED = "#8A9099"
_BAND = "#F2F4F6"


def display_labels(names: list[str], labels: dict[str, str] | None = None) -> list[str]:
    """Render identifiers as axis labels: the study's wording, else the name opened up.

    Program and group names are obs column keys — ``integrin_focal_adhesion`` — and
    a manuscript row label is not an identifier. The mapping is the caller's,
    because "EndoMT (LEC)" is this study's wording and no rule derives it; what the
    engine supplies is the fallback that at least does not print underscores.
    """
    mapping = labels or {}
    return [str(mapping.get(name, str(name).replace("_", " "))) for name in names]


def _wrap(text: str, width: int = 34) -> str:
    """Wrap a long axis label onto more lines instead of letting it run off the panel.

    A rotated y label longer than the axes is taller than the figure, and the
    overflow is cut rather than expanded — the LEC arm's "EndoMT index (z
    EndoMT+mesenchymal - z LEC identity)" lost its last two words. Wrapping keeps
    the whole label on the panel, which is what makes it a label.
    """
    return "\n".join(textwrap.wrap(str(text), width=width)) or str(text)


def ordered_programs(
    programs: list[str], categories: dict[str, list[str]]
) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Order programs by category and return the row bands for annotation.

    Categories are drawn in the order the config declares them, which is how a
    figure keeps a biological grouping (identity before mechanics before barrier)
    that alphabetical order would scramble. Anything the config forgot lands in a
    trailing band rather than vanishing.

    Public because the stage sorts its effect table with the same call. The table
    is what a replot reads, so if it carried a different row order from the panel
    the two would drift apart the first time anyone redrew the figure.
    """
    ordered: list[str] = []
    bands: list[tuple[str, int, int]] = []
    for category, members in (categories or {}).items():
        present = [m for m in members if m in programs and m not in ordered]
        if not present:
            continue
        bands.append((category, len(ordered), len(ordered) + len(present) - 1))
        ordered.extend(present)
    leftover = [p for p in programs if p not in ordered]
    if leftover:
        label = "other" if bands else ""
        bands.append((label, len(ordered), len(ordered) + len(leftover) - 1))
        ordered.extend(leftover)
    return ordered, bands


def _dot_areas(fdr: np.ndarray, *, max_exponent: float) -> np.ndarray:
    """Map ``-log10(FDR)`` onto dot area, clipped at ``max_exponent``.

    Without the clip a single FDR of 1e-40 sets the scale and every other dot
    collapses to the floor — the panel then shows one finding and hides ten.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = -np.log10(np.clip(fdr.astype(float), 1e-300, 1.0))
    exponent = np.nan_to_num(exponent, nan=0.0, posinf=max_exponent)
    fraction = np.clip(exponent / max(max_exponent, 1e-9), 0.0, 1.0)
    return _DOT_AREA_MIN + fraction * (_DOT_AREA_MAX - _DOT_AREA_MIN)


def plot_module_dotgrid(
    effects: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str = "module_remodeling_dotgrid",
    categories: dict[str, list[str]] | None = None,
    group_order: list[str] | None = None,
    group_label: str | None = None,
    program_labels: dict[str, str] | None = None,
    case: str | None = None,
    control: str | None = None,
    effect_cap: float | None = None,
    max_dot_fdr_exponent: float = 6.0,
    title: str | None = None,
) -> list[Path]:
    """Flagship: module x group dot-grid of mixed-model condition effects.

    Args:
        effects: The stage's effect-size table — one row per (group, program)
            with ``effect``, ``fdr`` and ``method`` columns.
        out_dir: Directory to write into.
        stem: Filename stem; ``.pdf`` and ``.png`` are both written.
        categories: Category -> programs, for row grouping and order.
        group_order: Column order; defaults to the table's order of appearance.
        group_label: Name of the group axis — what the columns *are*. Omitting it
            leaves the axis unlabelled, which on a partition of bare cluster
            numbers leaves the reader guessing.
        program_labels: Program -> row label, for manuscript wording. Anything
            missing falls back to the program name with underscores opened up.
        case: Case label, for the colourbar's directional wording.
        control: Control label, likewise.
        effect_cap: Symmetric colour cap; ``None`` uses the 98th percentile of
            ``|effect|`` so one extreme cell cannot flatten the rest. Derived per
            panel, which means two panels drawn from different runs are NOT colour
            comparable unless this is passed explicitly.
        max_dot_fdr_exponent: Clip for the ``-log10(FDR)`` size scale.
        title: Panel title; the study-specific wording belongs to the caller.

    Returns:
        The written paths.
    """
    figstyle.set_style()

    programs = list(dict.fromkeys(effects["program"]))
    groups = list(group_order) if group_order else list(dict.fromkeys(effects["group"]))
    programs, bands = ordered_programs(programs, categories or {})

    grid = effects.set_index(["program", "group"])
    effect = np.full((len(programs), len(groups)), np.nan)
    fdr = np.ones((len(programs), len(groups)))
    for i, program in enumerate(programs):
        for j, group in enumerate(groups):
            if (program, group) in grid.index:
                row = grid.loc[(program, group)]
                effect[i, j] = float(row["effect"])
                fdr[i, j] = float(row["fdr"]) if np.isfinite(row["fdr"]) else 1.0

    finite = effect[np.isfinite(effect)]
    if effect_cap is None:
        effect_cap = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    norm = figstyle.diverging_norm(finite, vmax=effect_cap or 1.0)

    # Floors, not just per-element scaling: the header size key needs about 5.5in
    # of width and the tallest marker about 0.5in of height regardless of how few
    # programs or groups there are, and a grid narrower than its own legend reads
    # as a mistake. The category labels get their own reserved width from the
    # longest name, because "actomyosin contractility" set beside the grid is
    # wider than the gap a fixed colourbar pad would leave it.
    longest_category = max((len(label) for label, _, _ in bands if label), default=0)
    label_inches = (0.12 + 0.049 * longest_category) if longest_category else 0.0
    height = max(3.1, 0.34 * len(programs) + 2.0)
    width = max(5.6, 0.95 * len(groups) + 3.4 + label_inches)
    fig, ax = plt.subplots(figsize=(width, height))

    # Category bands behind the dots, so a reader can see the grouping without
    # reading eleven row labels.
    for index, (_label, start, end) in enumerate(bands):
        if index % 2 == 0:
            ax.axhspan(start - 0.5, end + 0.5, color=_BAND, zorder=0, linewidth=0)

    tested = np.isfinite(effect)
    xs, ys = np.meshgrid(np.arange(len(groups)), np.arange(len(programs)))
    scatter = ax.scatter(
        xs[tested],
        ys[tested],
        s=_dot_areas(fdr[tested], max_exponent=max_dot_fdr_exponent),
        c=effect[tested],
        cmap="RdBu_r",
        norm=norm,
        edgecolors=np.where(fdr[tested] < 0.05, _SIG_RING, _NULL_RING),
        linewidths=np.where(fdr[tested] < 0.05, 1.1, 0.5),
        zorder=3,
    )
    if (~tested).any():
        ax.scatter(
            xs[~tested],
            ys[~tested],
            marker="x",
            s=26,
            c=_UNTESTED,
            linewidths=1.0,
            zorder=3,
        )

    ax.set_xticks(np.arange(len(groups)))
    # Rotate only when the labels need it. Cluster numbers set at 30 degrees read
    # as a rendering accident; names long enough to collide genuinely do need it.
    group_text = display_labels(groups)
    rotate = max((len(text) for text in group_text), default=0) > 4
    ax.set_xticklabels(
        group_text,
        rotation=30 if rotate else 0,
        ha="right" if rotate else "center",
    )
    ax.set_yticks(np.arange(len(programs)))
    ax.set_yticklabels(display_labels(programs, program_labels))
    if group_label:
        ax.set_xlabel(_wrap(group_label), fontsize=8.5, labelpad=6)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(len(programs) - 0.5, -0.5)  # first module at the top
    ax.set_axisbelow(True)
    ax.grid(axis="both", color="#E3E6E9", linewidth=0.5, zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Category names on the right, so they do not compete with the row labels, and
    # set horizontally with a bracket. Rotated labels only work on a band deep
    # enough to hold them: a one-row band renders its name taller than itself, so
    # two adjacent single-module categories printed on top of each other.
    blended = blended_transform_factory(ax.transAxes, ax.transData)
    for label, start, end in bands:
        if not label:
            continue
        ax.plot(
            [1.008, 1.008],
            [start - 0.42, end + 0.42],
            transform=blended,
            color=figstyle.TEXT,
            linewidth=0.8,
            solid_capstyle="butt",
            clip_on=False,
            zorder=4,
        )
        ax.text(
            1.022,
            (start + end) / 2.0,
            label.replace("_", " "),
            transform=blended,
            ha="left",
            va="center",
            fontsize=7.5,
            color=figstyle.TEXT,
        )

    direction = f"{case} − {control}" if case and control else "case − control"
    bar_pad = 0.05 + label_inches / max(width * 0.8, 1.0)
    bar = fig.colorbar(scatter, ax=ax, fraction=0.030, pad=bar_pad, shrink=0.85)
    bar.set_label(f"effect, {direction}", fontsize=8, labelpad=8)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_linewidth(0.5)

    # The size key runs horizontally across the header. Stacked vertically and
    # anchored outside the axes it landed on top of the colourbar's label and its
    # own largest marker; laid out in a row, matplotlib measures each handle's
    # width and nothing can collide.
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=_SIG_RING,
            markersize=np.sqrt(_dot_areas(np.array([q]), max_exponent=max_dot_fdr_exponent)[0]),
            label=text,
        )
        for q, text in (
            (1.0, "FDR 1.0"),
            (0.05, "FDR 0.05"),
            (10.0**-max_dot_fdr_exponent, f"FDR ≤ 1e−{max_dot_fdr_exponent:g}"),
        )
    ]
    if (~tested).any():
        # Only key the cross when a cross is on the panel. Advertising a marker
        # that is not drawn sends a reader hunting for untestable cells in a grid
        # where every cell was tested.
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                linestyle="none",
                color=_UNTESTED,
                markersize=5,
                label="not testable",
            )
        )
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=len(handles),
        frameon=False,
        fontsize=7,
        borderpad=0.0,
        handletextpad=0.7,
        columnspacing=1.5,
    )

    if title:
        # Above the size key, not behind it: set_title would sit in the same band.
        fig.suptitle(title, fontsize=10, x=0.02, ha="left")

    return figstyle.save_figure(fig, out_dir, stem)


def plot_contrast_index(
    values: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str = "module_contrast_index",
    group_col: str,
    condition_col: str,
    index_col: str,
    group_label: str | None = None,
    index_label: str | None = None,
    significance: dict[str, float] | None = None,
    case: str | None = None,
    control: str | None = None,
    title: str | None = None,
) -> list[Path]:
    """Signed contrast index by group and condition (paired violins).

    The two conditions sit side by side within each group rather than in separate
    panels, which is what makes the *shift* legible instead of two distributions
    the reader has to align by eye.

    Args:
        values: Per-cell frame with the group, condition and index columns. Its
            group order is the panel's column order.
        out_dir: Directory to write into.
        stem: Filename stem; ``.pdf`` and ``.png`` are both written.
        group_col: Column holding the group.
        condition_col: Column holding the condition.
        index_col: Column holding the per-cell index.
        group_label: Name of the group axis.
        index_label: Name of the *quantity* — "signed contrast index" describes
            the formula, and every study's index measures something different.
        significance: Group -> FDR, from the stage's per-group test of this index.
            Drawn as stars above each pair, so the panel carries the evidence for
            the shift it shows rather than leaving the reader to judge an overlap.
        case: Case label; fixes which violin sits on the right.
        control: Control label, likewise.
        title: Panel title.

    Returns:
        The written paths.
    """
    figstyle.set_style()

    groups = list(dict.fromkeys(values[group_col]))
    conditions = [c for c in (control, case) if c] or list(dict.fromkeys(values[condition_col]))
    palette = figstyle.condition_palette(case, control)

    fig, ax = plt.subplots(figsize=(max(4.0, 1.25 * len(groups) + 2.0), 3.4))
    tops: dict[int, float] = {}
    for offset, condition in zip((-1, 1), conditions[:2], strict=False):
        data, positions = [], []
        for i, group in enumerate(groups):
            selected = values[(values[group_col] == group) & (values[condition_col] == condition)][
                index_col
            ].to_numpy(dtype=float)
            selected = selected[np.isfinite(selected)]
            if selected.size < 2:
                continue
            data.append(selected)
            positions.append(i + offset * 0.19)
            tops[i] = max(tops.get(i, -np.inf), float(selected.max()))
        if not data:
            continue
        parts = ax.violinplot(data, positions=positions, widths=0.34, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(palette.get(condition, figstyle.CELLQUORUM_GRAY))
            body.set_edgecolor("none")
            body.set_alpha(0.85)
        ax.scatter(
            positions,
            [float(np.median(d)) for d in data],
            s=12,
            color="white",
            edgecolors=figstyle.TEXT,
            linewidths=0.6,
            zorder=4,
        )

    ax.axhline(0.0, color=_NULL_RING, linewidth=0.7, zorder=1)

    # The verdict above each pair, from the stage's own test of this index. Placed
    # off the taller of the two violins so it never lands inside a distribution, and
    # the ceiling is lifted to make room rather than letting it clip the frame.
    # "ns" is printed rather than left blank: a bare gap cannot be told apart from a
    # group that was never tested.
    if significance and tops:
        span = max(tops.values()) - min(
            float(values[index_col].min()), 0.0
        )  # data range, for a proportional gap
        gap = 0.06 * (span if np.isfinite(span) and span > 0 else 1.0)
        # One row, above the tallest violin on the panel. Per-group heights track
        # each pair's own maximum and come out jagged, which reads as a pattern in
        # the annotations that is really a pattern in the tails.
        row = max(tops.values()) + gap
        highest = -np.inf
        for i, group in enumerate(groups):
            fdr = significance.get(str(group))
            if fdr is None or i not in tops:
                continue
            stars = figstyle.significance_stars(fdr)
            y = row
            ax.text(
                i,
                y,
                stars,
                ha="center",
                va="bottom",
                # "ns" set smaller and greyer than a star: it is the absence of a
                # finding and should not read with the same weight as one.
                fontsize=7.0 if stars == "ns" else 9.0,
                color=_NULL_RING if stars == "ns" else figstyle.TEXT,
            )
            highest = max(highest, y)
        if np.isfinite(highest):
            low, high = ax.get_ylim()
            ax.set_ylim(low, max(high, highest + 2.2 * gap))

    ax.set_xticks(np.arange(len(groups)))
    group_text = display_labels([str(g) for g in groups])
    rotate = max((len(text) for text in group_text), default=0) > 4
    ax.set_xticklabels(group_text, rotation=20 if rotate else 0, ha="right" if rotate else "center")
    if group_label:
        ax.set_xlabel(_wrap(group_label), fontsize=8.5, labelpad=6)
    ax.set_ylabel(_wrap(index_label or "signed contrast index", width=30))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Condition key in a row above the axes, not inside them. ``loc="best"`` put it
    # in the top-right corner, which is exactly where the significance row for the
    # last group sits — the legend landed on top of that group's stars.
    ax.legend(
        handles=[
            Line2D([], [], marker="s", linestyle="none", color=palette.get(c, "#999999"), label=c)
            for c in conditions[:2]
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8,
        borderpad=0.0,
        columnspacing=1.6,
        handletextpad=0.6,
    )
    if title:
        # Above the key, for the same reason: set_title would sit in its band.
        fig.suptitle(title, fontsize=10, x=0.02, ha="left")

    return figstyle.save_figure(fig, out_dir, stem)


def plot_permanova(
    permanova: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str = "module_permanova",
    group_label: str | None = None,
    title: str | None = None,
) -> list[Path]:
    """Multivariate condition effect per group: R² bars with significance.

    R² is the share of between-sample module-profile variance the condition
    explains, so the bar height is the effect and the annotation is the evidence
    — kept separate on purpose, because a tall bar from four samples and a tall
    bar from eighteen are not the same claim.
    """
    figstyle.set_style()

    frame = permanova.dropna(subset=["R2"]).copy()
    if frame.empty:
        raise ValueError("no finite R2 values to plot")
    groups = frame["group"].astype(str).tolist()
    r2 = frame["R2"].to_numpy(dtype=float)

    # Star the adjusted p-value: one PERMANOVA per group is a family, and the
    # unadjusted version stars whichever null group permuted lowest. Tables written
    # before ``fdr`` existed still replot -- a run directory is the replot source,
    # so an old one must not raise -- but the panel then says which value it drew.
    adjusted = "fdr" in frame.columns and frame["fdr"].notna().any()
    evidence = frame["fdr" if adjusted else "p_value"].to_numpy(dtype=float)

    # 0.62in per group, not 0.95: the sample count moved off the bar and onto the
    # tick below (it describes the GROUP, not that bar's height), so the panel no
    # longer has to be wide enough to keep two "18 samples" labels from colliding.
    fig, ax = plt.subplots(figsize=(max(3.4, 0.62 * len(groups) + 1.7), 3.2))
    significant = evidence < 0.05
    ax.bar(
        np.arange(len(groups)),
        r2,
        width=0.62,
        color=np.where(significant, figstyle.LE_RED, _NULL_RING),
        edgecolor=figstyle.TEXT,
        linewidth=0.6,
    )
    counts = frame.get("n_samples", pd.Series(np.nan, index=frame.index))
    for i, (value, p) in enumerate(zip(r2, evidence, strict=False)):
        ax.text(
            i,
            value + 0.02,
            figstyle.significance_stars(p),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax.set_xticks(np.arange(len(groups)))
    group_text = display_labels(groups)
    # Rotation is decided on the group NAME alone; the count is a second line and
    # would otherwise tip every panel into rotating single-digit cluster numbers.
    rotate = max((len(text) for text in group_text), default=0) > 4
    tick_text = [
        f"{text}\n({int(n)})" if np.isfinite(n) else text
        for text, n in zip(group_text, counts, strict=False)
    ]
    ax.set_xticklabels(tick_text, rotation=20 if rotate else 0, ha="right" if rotate else "center")
    if group_label:
        ax.set_xlabel(_wrap(group_label), fontsize=8.5, labelpad=6)
    ax.set_ylabel("R² (condition, PERMANOVA)")
    ax.text(
        0.0,
        1.02,
        ("stars: BH-FDR" if adjusted else "stars: p, unadjusted") + "   (n) = samples",
        transform=ax.transAxes,
        fontsize=7.0,
        color=_UNTESTED if adjusted else figstyle.LE_RED,
        va="bottom",
    )
    ax.set_ylim(0, min(1.0, float(r2.max()) * 1.35 + 0.08))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)

    return figstyle.save_figure(fig, out_dir, stem)


__all__ = [
    "ordered_programs",
    "plot_contrast_index",
    "plot_module_dotgrid",
    "plot_permanova",
]
