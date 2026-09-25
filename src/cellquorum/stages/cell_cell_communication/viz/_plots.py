"""Biology-agnostic plotting primitives for ccc_viz.

Each function takes a tidy DataFrame plus explicit column-name arguments and
draws. No file I/O, no config objects, no biological literals. Optional plotting
deps (pycirclize, plotly) are import-guarded inside the functions that use them,
with a matplotlib fallback so a missing dep never raises.
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Wedge

from cellquorum.visualization.figstyle import CATEGORICAL_PALETTE, SEQUENTIAL_CMAP
from cellquorum.visualization.figstyle import get_group_palette as _get_group_palette

_OTHER_GRAY = "#9E9E9E"
_SEQ_CMAP = SEQUENTIAL_CMAP


def _clean_id(v: object) -> str:
    """String form of a node/cluster id, stripping a spurious trailing '.0'.

    Real bug this fixes: cluster ids ("0", "1", ..., "6") all look numeric,
    so a CSV read without an explicit dtype=str silently infers a float64
    column -- str(6.0) == "6.0", not "6", so every node label in a rendered
    network figure read "6.0" instead of "6". A whole-number float is
    rendered as its integer form regardless of which dtype the caller's
    DataFrame happens to carry.
    """
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


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
    """Deterministic sorted-celltype -> validated categorical color. Entity-keyed.

    Delegates to figstyle's audited CATEGORICAL_PALETTE (via get_group_palette)
    rather than a second, never-audited color list: that list capped out at 8
    colors with flat-gray overflow past it, which silently made a 9th cell
    type (the real fibroblast cohort has 9 subclusters) indistinguishable from
    background in every CCC figure. palette_colors never cycles or grays out.
    """
    return _get_group_palette(list(cell_types))


def _empty_fig(msg: str = "no data") -> Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.set_axis_off()
    return fig


def _interaction_dotplot_data(
    lr: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
    weight_col: str = "weight",
    sample_col: str = "sample",
    top_k: int = 15,
    max_pairs: int = 30,
) -> dict:
    """Curate rows (LR pairs) AND columns (source->target combos) to a
    readable size, then aggregate for plotting.

    Curating only the rows and leaving every distinct source->target combo
    on the x-axis is how dotplot_liana.png ended up 12,664px wide on a real
    9-subcluster run: 9 clusters is up to 72 combos, and even a handful of
    kept LR pairs routinely touches most of them. max_pairs caps that axis
    the same way top_k caps the LR-pair axis, keeping the top combos by
    total weight across the kept LR pairs.
    """
    if lr is None or lr.empty:
        return {"xs": [], "ys": [], "agg": pd.DataFrame(), "x_pos": {}, "y_pos": {}}
    df = lr.copy()
    df["__lr"] = df[ligand_col].astype(str) + "->" + df[receptor_col].astype(str)
    df["__st"] = df[source_col].astype(str) + "->" + df[target_col].astype(str)
    pair_weight = (
        df.groupby("__lr")[weight_col].sum().sort_values(kind="mergesort", ascending=False)
    )
    keep = list(pair_weight.head(top_k).index)
    df = df[df["__lr"].isin(keep)]
    if df.empty:
        return {"xs": [], "ys": [], "agg": pd.DataFrame(), "x_pos": {}, "y_pos": {}}

    st_weight = df.groupby("__st")[weight_col].sum().sort_values(kind="mergesort", ascending=False)
    keep_st = set(st_weight.head(max_pairs).index)
    df = df[df["__st"].isin(keep_st)]

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
    return {"xs": xs, "ys": ys, "agg": agg, "x_pos": x_pos, "y_pos": y_pos}


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
    max_pairs: int = 30,
) -> Figure:
    """Top-N LR pairs x top-N source->target combos; color=mean, size=#samples."""
    if lr is None or lr.empty:
        return _empty_fig()
    data = _interaction_dotplot_data(
        lr,
        source_col=source_col,
        target_col=target_col,
        ligand_col=ligand_col,
        receptor_col=receptor_col,
        weight_col=weight_col,
        sample_col=sample_col,
        top_k=top_k,
        max_pairs=max_pairs,
    )
    xs, ys, agg, x_pos, y_pos = data["xs"], data["ys"], data["agg"], data["x_pos"], data["y_pos"]
    if not xs or not ys:
        return _empty_fig()
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(xs)), max(4, 0.4 * len(ys))))
    w = agg["mean_w"].to_numpy(dtype=float)
    finite_w = w[np.isfinite(w)]
    if finite_w.size:
        vmin, vmax = float(finite_w.min()), float(finite_w.max())
        if vmax == vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
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
        # weight >= 0 by canonical LR contract; sequential ramp anchored at 0
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


def plot_mnn_sample_activity(
    sample_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    top_n: int = 20,
    cluster_labels: dict[str, str] | None = None,
) -> Figure:
    """MultiNicheNet's own headline figure: per-sample expression, scaled and
    raw ligand activity in the receiver, and ligand/receptor celltype
    specificity, for the same top-N prioritized LR interactions.

    Reproduces multinichenetr's own make_sample_lr_prod_activity_plots() in
    matplotlib/house style rather than importing its ggplot output: same 4
    panels, same source columns (sample_prioritization_tbl for panel 1;
    group_prioritization_table_source for panels 2-3 and most of panel 4).

    ``source_df`` must additionally carry scaled_pb_ligand/scaled_pb_receptor
    (panel 4's color) merged in by the caller from group_prioritization_tbl
    (mnn_prioritization.csv) on (id, group) -- confirmed against a real
    MultiNicheNet rerun that those two columns are NOT part of
    group_prioritization_table_source itself, despite both being read from
    the same multi_nichenet_analysis() call's prioritization_tables.

    Deliberately simplified from the reference:
    rows are a flat, sorted "sender -> receiver: ligand - receptor" list
    instead of nested facet strips (matplotlib has no facet_grid equivalent,
    and free per-facet row heights add real complexity for a readability
    gain a sorted flat list already gets most of), and panels 2/3 show one
    activity value per group rather than splitting by up/down direction
    (redundant with activity_scaled's own sign in the near-totality of
    real rows). A deliberate exception to one-plot-per-file, like
    plot_tensor_factors: comparing these 4 views for the SAME interactions
    is the entire point of the figure.
    """
    if sample_df is None or sample_df.empty or source_df is None or source_df.empty:
        return _empty_fig()

    labels = cluster_labels or {}

    def _route(r: pd.Series) -> str:
        sender = labels.get(str(r["sender"]), str(r["sender"]))
        receiver = labels.get(str(r["receiver"]), str(r["receiver"]))
        return f"{sender} -> {receiver}: {r['ligand']} - {r['receptor']}"

    top_ids = (
        source_df.groupby("id")["prioritization_score"]
        .max()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    sample_d = sample_df[sample_df["id"].isin(top_ids)].copy()
    source_d = source_df[source_df["id"].isin(top_ids)].copy()
    if sample_d.empty or source_d.empty:
        return _empty_fig()

    id_route = source_d.drop_duplicates("id").set_index("id").apply(_route, axis=1).to_dict()
    rows = sorted(set(id_route.values()))
    row_pos = {r: i for i, r in enumerate(rows)}
    sample_d["__row"] = sample_d["id"].map(id_route)
    source_d["__row"] = source_d["id"].map(id_route)

    # sender -> receiver prefix only (drop the ": ligand - receptor" suffix),
    # to find where one route's block of rows ends and the next begins --
    # rows are already sorted so same-route rows are contiguous.
    route_of_row = {r: r.split(": ", 1)[0] for r in rows}
    group_boundaries = [
        i + 0.5 for i in range(len(rows) - 1) if route_of_row[rows[i]] != route_of_row[rows[i + 1]]
    ]

    groups = sorted(source_d["group"].unique())
    # Sorted by (group, sample), not bare sample name: a plain alphabetical
    # sort interleaves the two conditions by patient (P10_LE, P10_Normal,
    # P12_LE, ...), confirmed on a real render -- samples for the same group
    # need to render as one contiguous block, matching the reference
    # figure's per-group facet columns.
    sample_group = sample_d.drop_duplicates("sample").set_index("sample")["group"]
    samples = sorted(sample_d["sample"].unique(), key=lambda s: (sample_group[s], s))
    sample_x_range: dict[str, tuple[float, float]] = {}
    for g in groups:
        xs = [i for i, s in enumerate(samples) if sample_group[s] == g]
        if xs:
            sample_x_range[g] = (min(xs) - 0.5, max(xs) + 0.5)

    fig, axes = plt.subplots(
        1,
        4,
        # 0.5in/row (was 0.35) and extra top margin for the group header --
        # confirmed too cramped on a real render next to the reference.
        figsize=(17, max(3.5, 0.5 * len(rows) + 1.0)),
        squeeze=True,
        gridspec_kw={
            "width_ratios": [max(2, len(samples)), len(groups), len(groups), 2 * len(groups)],
            "wspace": 0.35,
        },
    )

    # Panel 1: per-sample scaled LR pseudobulk product, sized by
    # sender/receiver presence.
    ax = axes[0]
    keep_sizes = {
        "Sender & Receiver absent": 20,
        "Receiver absent": 45,
        "Sender absent": 70,
        "Sender & Receiver present": 140,
    }
    x_pos = {s: i for i, s in enumerate(samples)}
    vmax = float(sample_d["scaled_LR_pb_prod"].abs().max()) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    ax.scatter(
        [x_pos[s] for s in sample_d["sample"]],
        [row_pos[r] for r in sample_d["__row"]],
        c=sample_d["scaled_LR_pb_prod"],
        cmap="RdBu_r",
        norm=norm,
        s=[keep_sizes.get(k, 40) for k in sample_d["keep_sender_receiver"]],
        edgecolor="0.3",
        linewidth=0.3,
    )
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels(samples, rotation=90, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title("Sample L-R product", fontsize=9)
    ax.set_ylim(-0.5, len(rows) - 0.5 + max(1.0, 0.06 * len(rows)))
    for g, (x0, x1) in sample_x_range.items():
        header_y = len(rows) - 0.5 + max(1.0, 0.06 * len(rows)) * 0.5
        ax.text((x0 + x1) / 2, header_y, g, ha="center", va="center", fontweight="bold", fontsize=9)
        ax.plot(
            [x0 + 0.05, x1 - 0.05], [len(rows) - 0.5, len(rows) - 0.5], color="0.4", linewidth=1.0
        )

    # Panel 2: scaled ligand activity in receiver, one column per group.
    ax = axes[1]
    vmax = float(source_d["activity_scaled"].abs().max()) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    mat = source_d.pivot_table(index="__row", columns="group", values="activity_scaled")
    mat = mat.reindex(index=rows, columns=groups)
    # origin="lower": imshow's default puts array row 0 at the TOP, but
    # panel 1's scatter (matplotlib's default, non-inverted y-axis) puts
    # row_pos 0 at the BOTTOM -- confirmed mismatched on a real render,
    # which silently paired every row's activity with the WRONG sample-
    # product/specificity row one panel over.
    ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=90, fontsize=8)
    ax.set_yticks([])
    ax.set_title("Scaled activity", fontsize=9)

    # Panel 3: raw ligand activity in receiver, one column per group.
    ax = axes[2]
    mat = source_d.pivot_table(index="__row", columns="group", values="activity")
    mat = mat.reindex(index=rows, columns=groups)
    vals = mat.to_numpy(dtype=float)
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    ax.imshow(
        vals,
        cmap="Oranges",
        norm=Normalize(vmin=0.0, vmax=vmax or 1.0),
        aspect="auto",
        origin="lower",  # see panel 2's comment -- same row-order fix
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=90, fontsize=8)
    ax.set_yticks([])
    ax.set_title("Ligand activity", fontsize=9)

    # Panel 4: ligand/receptor celltype specificity, colored by scaled
    # pseudobulk expression, sized by fraction of group expressing.
    ax = axes[3]
    cols = [f"{lr}\n{g}" for g in groups for lr in ("ligand", "receptor")]
    col_pos = {c: i for i, c in enumerate(cols)}
    cs_rows = []
    for _, r in source_d.iterrows():
        cs_rows.append(
            (f"ligand\n{r['group']}", r["__row"], r["scaled_pb_ligand"], r["fraction_ligand_group"])
        )
        cs_rows.append(
            (
                f"receptor\n{r['group']}",
                r["__row"],
                r["scaled_pb_receptor"],
                r["fraction_receptor_group"],
            )
        )
    cs = pd.DataFrame(cs_rows, columns=["col", "row", "specificity", "fraction"])
    ax.scatter(
        [col_pos[c] for c in cs["col"]],
        [row_pos[r] for r in cs["row"]],
        c=cs["specificity"],
        cmap="viridis",
        s=200.0 * cs["fraction"].clip(lower=0),
        edgecolor="0.3",
        linewidth=0.3,
    )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7)
    ax.set_yticks([])
    ax.set_title("Celltype specificity", fontsize=9)

    # Separator between each sender->receiver route's block of rows, on
    # every panel -- confirmed too flat/hard to scan as a plain sorted list
    # next to the reference figure's bordered per-route groups.
    for ax in axes:
        xlim = ax.get_xlim()
        for y in group_boundaries:
            ax.plot(xlim, [y, y], color="0.75", linewidth=0.7, zorder=0)
        ax.set_xlim(xlim)

    fig.tight_layout()
    return fig


def cluster_focused_circos(
    df: pd.DataFrame,
    focus: str,
    *,
    source_col: str = "sender",
    target_col: str = "receiver",
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
    weight_col: str = "prioritization_score",
    palette: dict[str, str],
    top_k: int = 8,
    cluster_labels: dict[str, str] | None = None,
    title: str | None = None,
) -> Figure:
    """Circos scoped to ONE cluster's own edges, with the dominant
    ligand-receptor pair for every remaining ribbon listed alongside it.

    The whole-cohort circos (chord_diagram) puts every cluster's edges in
    one ring -- even with the fairness-guaranteed top_k cap, a real
    multi-subcluster cohort still crowds ribbons too thin to label, and
    pycirclize has no reliable way to print an LR pair directly on a curved
    ribbon at that density. Restricting to one cluster's own edges first
    solves both problems: few enough ribbons to read, and few enough rows
    to name every one's LR pair in a legend instead of guessing from color.
    """
    if df is None or df.empty:
        return _empty_fig()
    focus = str(focus)
    d = df.copy()
    d[source_col] = d[source_col].astype(str)
    d[target_col] = d[target_col].astype(str)
    d = d[(d[source_col] == focus) | (d[target_col] == focus)]
    if d.empty:
        return _empty_fig()
    # One ribbon per partner: each (source, target) route's single
    # strongest LR pair, ranked, capped to top_k partners.
    d = (
        d.sort_values(weight_col, ascending=False, kind="mergesort")
        .drop_duplicates(subset=[source_col, target_col])
        .head(top_k)
    )

    # Relabel BEFORE building the ring: chord_diagram's matrix is indexed by
    # whatever is in these two columns, so cluster_labels has to land here,
    # not just in the side legend -- a real render otherwise kept the ring's
    # own sector labels as bare codes ("6") even with a real name mapping
    # given, because only the legend text used it.
    labels = cluster_labels or {}
    d_display = d.copy()
    d_display[source_col] = d_display[source_col].map(lambda v: labels.get(v, v))
    d_display[target_col] = d_display[target_col].map(lambda v: labels.get(v, v))
    display_palette = {labels.get(k, k): v for k, v in palette.items()}

    fig = chord_diagram(
        d_display,
        source_col=source_col,
        target_col=target_col,
        weight_col=weight_col,
        palette=display_palette,
        top_k=top_k,
    )
    if title is None:
        title = labels.get(focus, focus)
    fig.suptitle(title, fontsize=11)

    lines = [
        f"{labels.get(r[source_col], r[source_col])} -> "
        f"{labels.get(r[target_col], r[target_col])}: "
        f"{r[ligand_col]} - {r[receptor_col]}"
        for _, r in d.sort_values(weight_col, ascending=False, kind="mergesort").iterrows()
    ]
    fig.text(
        1.02,
        0.5,
        "\n".join(lines),
        transform=fig.transFigure,
        fontsize=7,
        va="center",
        ha="left",
    )
    return fig


def mnn_top_lr_lollipop(
    df: pd.DataFrame,
    *,
    group_col: str = "group",
    sender_col: str = "sender",
    receiver_col: str = "receiver",
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
    score_col: str = "prioritization_score",
    top_n: int = 30,
    cluster_labels: dict[str, str] | None = None,
) -> Figure:
    """Top-N prioritized LR interactions, one lollipop per row, colored by
    sender->receiver, named by the actual ligand-receptor pair.

    The circos/chord view of this same data answers "which cell types talk
    more in disease" but actively hides WHICH molecules are doing it --
    everything collapses into one aggregate ribbon per cell-type pair. This
    is the figure that names names: each row is one real LR pair, ranked by
    MultiNicheNet's own prioritization_score, not a generic magnitude.

    When ``group_col`` is present, the top ``top_n`` rows are taken
    independently PER group (e.g. Normal, Lymphedema) so a low-signal group
    isn't crowded out of the ranking by a high-signal one.
    """
    if df is None or df.empty:
        return _empty_fig()
    d = df.copy()
    d[sender_col] = d[sender_col].astype(str)
    d[receiver_col] = d[receiver_col].astype(str)
    if group_col in d.columns:
        parts = [
            g.sort_values(score_col, ascending=False).head(top_n) for _, g in d.groupby(group_col)
        ]
        d = pd.concat(parts, ignore_index=True) if parts else d.iloc[0:0]
    else:
        d = d.sort_values(score_col, ascending=False).head(top_n)
    if d.empty:
        return _empty_fig()

    labels = cluster_labels or {}
    d = d.assign(
        __pair=d[ligand_col].astype(str) + " - " + d[receptor_col].astype(str),
        __route=(
            d[sender_col].map(lambda v: labels.get(v, v))
            + " -> "
            + d[receiver_col].map(lambda v: labels.get(v, v))
        ),
    )
    # Sort bottom-to-top so the strongest interaction renders at the top.
    d = d.sort_values(score_col, ascending=True, kind="mergesort")

    routes = sorted(d["__route"].unique())
    palette = celltype_palette(routes)

    # A real run's top-N routinely has 20-30 distinct sender->receiver
    # routes; that legend does not fit inside the axes without covering
    # data, so it gets its own column to the right instead. Width scales
    # with route count for the same reason.
    legend_w = max(2.2, 0.028 * len(routes) * max(len(r) for r in routes))
    height = min(7.0, max(3.0, 0.2 * len(d)))
    fig, ax = plt.subplots(figsize=(7 + legend_w, height))
    y = np.arange(len(d))
    colors = [palette[r] for r in d["__route"]]
    ax.hlines(y, 0, d[score_col], color=colors, linewidth=1.5, zorder=1)
    ax.scatter(d[score_col], y, c=colors, s=60, zorder=2, edgecolor="0.3", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(d["__pair"])
    ax.set_xlabel(score_col)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=palette[r], label=r) for r in routes
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=7,
        frameon=False,
        title="sender -> receiver",
    )
    fig.tight_layout()
    return fig


def ligand_activity_bar(
    activities: pd.DataFrame,
    *,
    ligand_col: str = "test_ligand",
    score_col: str = "aupr_corrected",
    top_n: int = 20,
) -> Figure:
    """Ranked ligand-activity leaderboard: NicheNet's own headline output.

    predict_ligand_activities() scores every candidate ligand by how well its
    predicted targets explain the receiver's real DE geneset -- this is the
    number the whole analysis exists to produce, and nothing in the generic
    canonical-LR pipeline (built for source/target/weight edges) can show it.
    """
    if activities is None or activities.empty:
        return _empty_fig()
    df = activities[[ligand_col, score_col]].dropna()
    if df.empty:
        return _empty_fig()
    df = df.sort_values(score_col, ascending=False, kind="mergesort").head(top_n)
    # Bottom-to-top so the strongest ligand renders at the top of the chart.
    df = df.iloc[::-1]
    scores = df[score_col].to_numpy(dtype=float)
    finite = scores[np.isfinite(scores)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if hi == lo:
        hi = lo + 1.0
    norm = Normalize(vmin=lo, vmax=hi)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(df))))
    ax.barh(
        df[ligand_col].astype(str),
        scores,
        color=plt.get_cmap("Oranges")(norm(scores)),
    )
    ax.set_xlabel(score_col)
    fig.tight_layout()
    return fig


def ligand_target_heatmap(
    weights: pd.DataFrame,
    *,
    ligand_col: str = "ligand",
    target_col: str = "target",
    weight_col: str = "weight",
) -> Figure:
    """Ligand x target regulatory-potential matrix for NicheNet's top ligands.

    get_weighted_ligand_target_links() already restricts rows to the top
    ligands' top target genes, so a 0 fill for an absent (ligand, target)
    pair means "not among this ligand's top predicted targets", matching how
    NicheNet's own reference figures render this matrix.
    """
    if weights is None or weights.empty:
        return _empty_fig()
    mat = weights.pivot_table(
        index=ligand_col, columns=target_col, values=weight_col, aggfunc="mean", fill_value=0.0
    )
    mat = mat.sort_index(kind="mergesort").sort_index(axis=1, kind="mergesort")
    if mat.empty:
        return _empty_fig()
    fig, ax = plt.subplots(figsize=(max(4, 0.4 * mat.shape[1]), max(3, 0.4 * mat.shape[0])))
    vals = mat.to_numpy(dtype=float)
    norm = Normalize(vmin=0.0, vmax=float(np.nanmax(vals)) or 1.0)
    # A single-hue white->purple ramp (nichenetr's own convention), not the
    # shared multi-hue viridis: this matrix is mostly near-zero (only a
    # ligand's own top targets are non-zero), and a multi-hue map drowns
    # that in what reads as noise.
    im = ax.imshow(vals, cmap="Purples", norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(list(mat.columns), rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(list(mat.index))
    ax.set_xlabel(target_col)
    ax.set_ylabel(ligand_col)
    fig.colorbar(im, ax=ax, label=weight_col)
    fig.tight_layout()
    return fig


def ligand_receptor_heatmap(
    weights: pd.DataFrame,
    *,
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
    weight_col: str = "weight",
) -> Figure:
    """Ligand x receptor prior-interaction-potential matrix (over every expressed
    receptor, not one arbitrary cognate pick -- see nichenet.R's
    get_weighted_ligand_receptor_links() call).

    A distinct single-hue scale (pink/RdPu) from ligand_target_heatmap's
    purple: the two matrices answer different questions (which receptor
    binds this ligand vs. which target gene it regulates) and share a
    figure set, so a reader should be able to tell them apart at a glance.
    """
    if weights is None or weights.empty:
        return _empty_fig()
    mat = weights.pivot_table(
        index=ligand_col, columns=receptor_col, values=weight_col, aggfunc="mean", fill_value=0.0
    )
    mat = mat.sort_index(kind="mergesort").sort_index(axis=1, kind="mergesort")
    if mat.empty:
        return _empty_fig()
    fig, ax = plt.subplots(figsize=(max(4, 0.4 * mat.shape[1]), max(3, 0.4 * mat.shape[0])))
    vals = mat.to_numpy(dtype=float)
    norm = Normalize(vmin=0.0, vmax=float(np.nanmax(vals)) or 1.0)
    im = ax.imshow(vals, cmap="RdPu", norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(list(mat.columns), rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(list(mat.index))
    ax.set_xlabel(receptor_col)
    ax.set_ylabel(ligand_col)
    fig.colorbar(im, ax=ax, label=weight_col)
    fig.tight_layout()
    return fig


def sender_ligand_dotplot(
    expr: pd.DataFrame,
    *,
    sender_col: str = "sender",
    ligand_col: str = "ligand",
    frac_col: str = "fraction_expressing",
    mean_col: str = "mean_expression",
) -> Figure:
    """Sender x ligand dotplot: size=fraction expressing, color=mean expression.

    Answers "which cell type actually sends this ligand" -- a question the
    activity ranking and ligand-target matrix don't, since both are computed
    over a pooled sender pool (see nichenet_method.py's module docstring).
    """
    if expr is None or expr.empty:
        return _empty_fig()
    df = expr[[sender_col, ligand_col, frac_col, mean_col]].dropna(subset=[sender_col, ligand_col])
    if df.empty:
        return _empty_fig()
    senders = sorted(df[sender_col].astype(str).unique())
    ligands = sorted(df[ligand_col].astype(str).unique())
    x_pos = {v: i for i, v in enumerate(senders)}
    y_pos = {v: i for i, v in enumerate(ligands)}
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(senders)), max(3, 0.35 * len(ligands))))
    mean_vals = df[mean_col].to_numpy(dtype=float)
    finite = mean_vals[np.isfinite(mean_vals)]
    vmax = float(finite.max()) if finite.size else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax or 1.0)
    sizes = 300.0 * df[frac_col].to_numpy(dtype=float)
    sc = ax.scatter(
        [x_pos[s] for s in df[sender_col].astype(str)],
        [y_pos[lg] for lg in df[ligand_col].astype(str)],
        s=sizes,
        c=mean_vals,
        cmap=_SEQ_CMAP,
        norm=norm,
        edgecolor="0.3",
        linewidth=0.4,
    )
    ax.set_xticks(range(len(senders)))
    ax.set_xticklabels(senders, rotation=45, ha="right")
    ax.set_yticks(range(len(ligands)))
    ax.set_yticklabels(ligands)
    fig.colorbar(sc, ax=ax, label=mean_col)
    fig.tight_layout()
    return fig


def _guarantee_node_seats(
    agg: pd.DataFrame, source_col: str, target_col: str, weight_col: str, top_k: int
) -> pd.DataFrame:
    """From an already-deduplicated (source, target) edge table, keep the
    top_k strongest -- but never at the cost of dropping a node from the
    figure entirely.

    top_k is a flat cap on PAIRS; on a real, densely-connected graph (a
    fibroblast_subcluster CCC run: 9 clusters, up to 72 real pairs, or a
    36-edge complete CCI curvature graph) the globally top-15 pairs can
    easily all involve the same handful of clusters, silently erasing every
    other cluster from the figure with no warning. Every node's own single
    strongest edge is guaranteed a seat first; top_k then fills (or trims)
    the remainder.
    """
    agg = agg.sort_values([weight_col, source_col, target_col], kind="mergesort", ascending=False)
    if len(agg) <= top_k:
        return agg

    # Each node's single best edge (as either source or target), by weight.
    best_as_source = agg.sort_values(weight_col, ascending=False).drop_duplicates(source_col)
    best_as_target = agg.sort_values(weight_col, ascending=False).drop_duplicates(target_col)
    guaranteed = pd.concat([best_as_source, best_as_target]).drop_duplicates(
        subset=[source_col, target_col]
    )

    if len(guaranteed) >= top_k:
        return guaranteed.sort_values(
            [weight_col, source_col, target_col], kind="mergesort", ascending=False
        )

    remaining = agg.merge(
        guaranteed[[source_col, target_col]],
        on=[source_col, target_col],
        how="left",
        indicator=True,
    )
    remaining = remaining[remaining["_merge"] == "left_only"].drop(columns="_merge")
    fill = remaining.head(top_k - len(guaranteed))
    return pd.concat([guaranteed, fill]).sort_values(
        [weight_col, source_col, target_col], kind="mergesort", ascending=False
    )


def _top_edges(
    lr: pd.DataFrame, source_col: str, target_col: str, weight_col: str, top_k: int
) -> pd.DataFrame:
    """Aggregate to (source, target) pairs, then apply _guarantee_node_seats."""
    agg = lr.groupby([source_col, target_col])[weight_col].sum().reset_index()
    return _guarantee_node_seats(agg, source_col, target_col, weight_col, top_k)


def chord_diagram(
    lr: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    palette: dict[str, str],
    top_k: int = 15,
) -> Figure:
    """Circos chord (pycirclize) or matplotlib arc fallback. Never raises."""
    if lr is None or lr.empty:
        return _empty_fig()
    edges = _top_edges(lr, source_col, target_col, weight_col, top_k)
    if edges.empty:
        return _empty_fig()
    try:
        from pycirclize import Circos  # noqa: F401

        matrix = edges.pivot_table(
            index=source_col,
            columns=target_col,
            values=weight_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        matrix = matrix.sort_index(kind="mergesort").sort_index(axis=1, kind="mergesort")
        circos = Circos.initialize_from_matrix(
            matrix,
            cmap={
                ct: palette.get(ct, _OTHER_GRAY)
                for ct in sorted(set(matrix.index) | set(matrix.columns))
            },
        )
        return circos.plotfig()
    except Exception:  # noqa: BLE001  (import OR pycirclize runtime -> fallback)
        return _chord_fallback(edges, source_col, target_col, weight_col, palette)


def _chord_fallback(
    edges: pd.DataFrame, source_col: str, target_col: str, weight_col: str, palette: dict[str, str]
) -> Figure:
    """Matplotlib ring-of-arcs fallback (nodes on a circle, chords as lines)."""
    nodes = sorted(set(edges[source_col]) | set(edges[target_col]))
    ang = {n: 2 * np.pi * i / max(1, len(nodes)) for i, n in enumerate(nodes)}
    fig, ax = plt.subplots(figsize=(6, 6))
    for n, a in ang.items():
        ax.add_patch(Wedge((np.cos(a), np.sin(a)), 0.06, 0, 360, color=palette.get(n, _OTHER_GRAY)))
        ax.text(1.12 * np.cos(a), 1.12 * np.sin(a), str(n), ha="center", va="center", fontsize=8)
    wmax = float(edges[weight_col].max()) or 1.0
    for _, r in edges.iterrows():
        a0, a1 = ang[r[source_col]], ang[r[target_col]]
        ax.plot(
            [np.cos(a0), np.cos(a1)],
            [np.sin(a0), np.sin(a1)],
            color=palette.get(r[source_col], _OTHER_GRAY),
            linewidth=0.5 + 3.0 * r[weight_col] / wmax,
            alpha=0.6,
        )
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


def _select_sankey_flows(
    df: pd.DataFrame,
    *,
    source_col: str,
    ligand_col: str,
    receptor_col: str,
    target_col: str,
    weight_col: str,
    top_k: int,
) -> pd.DataFrame:
    """Rows for the sankey diagram: every real (source, target) cluster pair
    guaranteed a seat, keeping each guaranteed pair's single strongest
    ligand-receptor row.

    Real bug this fixes: a flat ``df.sort_values(weight).head(top_k)`` cut
    across the WHOLE LR results table (thousands of source-ligand-receptor-
    target rows on a real cohort) let the top-15 strongest individual rows
    concentrate in one or two source clusters, leaving another cluster with
    literally zero flow -- completely isolated in the rendered diagram, with
    no warning. chord_diagram already guards against exactly this via
    _top_edges/_guarantee_node_seats; this reuses the same guarantee so the
    sankey and the chord diagram agree about which cluster pairs matter.
    """
    edges = _top_edges(df, source_col, target_col, weight_col, top_k)
    if edges.empty:
        return edges
    return (
        df.sort_values(weight_col, ascending=False, kind="mergesort")
        .merge(edges[[source_col, target_col]], on=[source_col, target_col])
        .drop_duplicates(subset=[source_col, target_col])
    )


def sankey_flow(
    lr: pd.DataFrame,
    *,
    source_col: str = "source",
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
    target_col: str = "target",
    weight_col: str = "weight",
    palette: dict[str, str],
    top_k: int = 15,
) -> Figure:
    """source->ligand->receptor->target flow. plotly+kaleido, else matplotlib fallback."""
    if lr is None or lr.empty:
        return _empty_fig()
    df = lr.copy()
    df["__w"] = df[weight_col].astype(float)
    df = _select_sankey_flows(
        df,
        source_col=source_col,
        ligand_col=ligand_col,
        receptor_col=receptor_col,
        target_col=target_col,
        weight_col="__w",
        top_k=top_k,
    )
    if df.empty:
        return _empty_fig()
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        cols = [source_col, ligand_col, receptor_col, target_col]
        labels: list[str] = []
        idx: dict[tuple[int, str], int] = {}
        for ci, col in enumerate(cols):
            for v in sorted(df[col].astype(str).unique()):
                idx[(ci, v)] = len(labels)
                labels.append(v)
        src, tgt, val = [], [], []
        for ci in range(len(cols) - 1):
            g = df.groupby([cols[ci], cols[ci + 1]])["__w"].sum().reset_index()
            for _, r in g.iterrows():
                src.append(idx[(ci, str(r[cols[ci]]))])
                tgt.append(idx[(ci + 1, str(r[cols[ci + 1]]))])
                val.append(float(r["__w"]))
        figp = go.Figure(
            go.Sankey(
                node={"label": labels},
                link={"source": src, "target": tgt, "value": val},
            )
        )
        png = pio.to_image(figp, format="png", width=900, height=600, scale=2)
        import io

        import matplotlib.image as mpimg

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(mpimg.imread(io.BytesIO(png)))
        ax.set_axis_off()
        return fig
    except Exception:  # noqa: BLE001
        return _sankey_fallback(
            df, [source_col, ligand_col, receptor_col, target_col], weight_col, palette
        )


def _sankey_fallback(
    df: pd.DataFrame, cols: list[str], weight_col: str, palette: dict[str, str]
) -> Figure:
    """Matplotlib 4-column node/flow diagram."""
    fig, ax = plt.subplots(figsize=(9, 6))
    positions = {}
    for ci, col in enumerate(cols):
        vals = sorted(df[col].astype(str).unique())
        for vi, v in enumerate(vals):
            y = 1.0 - (vi + 0.5) / max(1, len(vals))
            positions[(ci, v)] = (ci, y)
            ax.scatter([ci], [y], s=200, color=palette.get(v, _OTHER_GRAY), zorder=3)
            ax.text(ci, y + 0.03, str(v), ha="center", va="bottom", fontsize=7)
    wmax = float(df[weight_col].max()) or 1.0
    for ci in range(len(cols) - 1):
        g = df.groupby([cols[ci], cols[ci + 1]])[weight_col].sum().reset_index()
        for _, r in g.iterrows():
            x0, y0 = positions[(ci, str(r[cols[ci]]))]
            x1, y1 = positions[(ci + 1, str(r[cols[ci + 1]]))]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="0.6",
                linewidth=0.5 + 3.0 * r[weight_col] / wmax,
                alpha=0.5,
                zorder=1,
            )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks([])
    ax.set_axis_off()
    return fig


def curvature_network(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    curvature_col: str = "ricci_curvature",
    weight_col: str = "weight",
    node_curv_col: str = "ricci_curvature",
    top_k: int = 30,
) -> Figure:
    """Curvature-colored network: kamada-kawai layout, curved edges, halo'd labels.

    Three changes from the original straight-line/spring-layout version, all
    aimed at the same complaint on real output (a real regulon network:
    "not advanced looking, and the labels are being covered by the network
    rather than clearly standing out"):

    - spring layout, but with room to breathe: k (optimal node distance) is
      scaled up from networkx's crowded 1/sqrt(n) default. kamada-kawai was
      tried first and looked cleaner on a well-connected graph, but a real
      regulon network (confirmed directly) is mostly disconnected TF-target
      dyads after the top_k trim, not one component -- kamada-kawai
      collapses each disconnected pair down to near-zero separation and
      scatters the pairs across mostly-empty canvas, which is worse than
      the original crowding, not better. Spring layout with a larger k
      keeps every component reliably legible regardless of how connected
      the graph turns out to be.
    - Curved edges (FancyArrowPatch, connectionstyle="arc3") instead of
      straight lines. Two edges between nearby nodes no longer draw
      exactly on top of each other, and the slight arcs make individual
      connections traceable by eye even where several cross near a hub --
      the standard technique real network-viz tools (Gephi, Cytoscape) use
      for exactly this readability problem.
    - Labels get a white stroke halo (same technique already used for
      module_umap_plot's hub labels) so text stays legible over a busy
      tangle of edges instead of visually merging with whatever crosses
      behind it.
    """
    if edges is None or edges.empty:
        return _empty_fig()
    # Guard for missing weight_col (differential curvature frames lack it)
    if weight_col not in edges.columns:
        e = edges.copy()
        e[weight_col] = 1.0
    else:
        e = edges
    e = _guarantee_node_seats(e, source_col, target_col, weight_col, top_k)
    G = nx.DiGraph()
    for _, r in e.iterrows():
        G.add_edge(
            _clean_id(r[source_col]),
            _clean_id(r[target_col]),
            curv=float(r[curvature_col]),
            weight=float(r[weight_col]),
        )
    if G.number_of_edges() == 0:
        return _empty_fig()
    n_nodes = max(G.number_of_nodes(), 1)
    pos = nx.spring_layout(G, seed=0, weight="weight", k=2.2 / n_nodes**0.5, iterations=100)
    curvs = np.array([d["curv"] for _, _, d in G.edges(data=True)], dtype=float)
    norm = signed_norm(curvs)
    cmap = plt.get_cmap("RdBu_r")
    fig, ax = plt.subplots(figsize=(9, 8))
    wmax = (curvs.size and float(e[weight_col].max())) or 1.0
    for (u, v, d), c in zip(G.edges(data=True), curvs, strict=False):
        arrow = FancyArrowPatch(
            pos[u],
            pos[v],
            connectionstyle="arc3,rad=0.12",
            color=cmap(norm(c)),
            linewidth=0.5 + 2.3 * d["weight"] / wmax,
            alpha=0.75,
            zorder=1,
            shrinkA=0,
            shrinkB=0,
            arrowstyle="-",
        )
        ax.add_patch(arrow)
    ncurv = {}
    if nodes is not None and not nodes.empty and node_curv_col in nodes.columns:
        node_strs = nodes["node"].map(_clean_id)
        node_vals = nodes[node_curv_col].astype(float)
        ncurv = dict(zip(node_strs, node_vals, strict=False))
    # Node size carries degree -- a hub reads as a hub at a glance, not just
    # from counting converging edges.
    degree = dict(G.degree())
    max_degree = max(degree.values()) if degree else 1
    label_texts = []
    for n, (x, y) in pos.items():
        size = 90.0 + 220.0 * (degree.get(n, 1) / max_degree) if max_degree else 140.0
        ax.scatter(
            [x],
            [y],
            s=size,
            color=cmap(norm(ncurv.get(n, 0.0))),
            edgecolor="0.15",
            linewidth=0.8,
            zorder=3,
        )
        t = ax.text(x, y, str(n), fontsize=7.5, ha="center", va="center", zorder=4)
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
        label_texts.append(t)
    # A wider k still leaves close-together nodes (a real, dense TF/target
    # pair) with their labels dead-centered on the marker and touching each
    # other -- the same repulsion module_umap_plot's hub labels already use.
    try:
        from adjustText import adjust_text

        adjust_text(
            label_texts,
            ax=ax,
            only_move={"points": "xy", "text": "xy"},
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.4),
        )
    except Exception:
        pass  # labels still render, just un-repelled
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=curvature_col, shrink=0.8)
    ax.set_axis_off()
    ax.margins(0.1)
    fig.tight_layout()
    return fig


def topology_facet_figures(
    topo: pd.DataFrame,
    *,
    node_col: str = "node",
    metric_cols: tuple[str, ...] = ("Listener", "Influencer", "Mediator", "Pagerank"),
    top_k: int = 15,
) -> dict[str, Figure]:
    """One figure per topology metric: top-N nodes as horizontal bars.

    One plot per file (house style), not a combined small-multiple grid --
    each metric (who receives most, who sends most, who mediates, who's most
    central) answers a different question a reader asks separately.

    A metric with zero variance across every node (e.g. betweenness
    centrality -- "Mediator" -- on a near-complete graph, where every
    shortest path is a direct edge and no node ever mediates anything: a
    genuine result, confirmed on a real 9-cluster CCC network with 81 edges)
    renders as an explained empty panel instead of silent blank bars.
    """
    if topo is None or topo.empty:
        return {}
    metrics = [m for m in metric_cols if m in topo.columns]
    if not metrics:
        return {}
    figures: dict[str, Figure] = {}
    for m in metrics:
        sub = topo.reindex(topo[m].abs().sort_values(kind="mergesort", ascending=False).index).head(
            top_k
        )
        sub = sub.iloc[::-1]
        vals = sub[m].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(5, max(2.5, 0.4 * len(sub) + 1)))
        finite = vals[np.isfinite(vals)]
        if finite.size == 0 or float(np.nanmax(finite) - np.nanmin(finite)) == 0.0:
            ax.text(
                0.5,
                0.5,
                f"{m}: no variation across nodes\n(degenerate for this network's structure)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                wrap=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            straddles = float(np.nanmin(vals)) < 0 < float(np.nanmax(vals))
            if straddles:
                norm = signed_norm(vals)
                colors = plt.get_cmap("RdBu_r")(norm(vals))
            else:
                norm = Normalize(vmin=float(np.nanmin(vals)), vmax=float(np.nanmax(vals)))
                colors = plt.get_cmap(_SEQ_CMAP)(norm(vals))
            y = np.arange(len(sub))
            ax.barh(y, vals, color=colors, edgecolor="0.3", linewidth=0.4)
            ax.set_yticks(y)
            ax.set_yticklabels(sub[node_col].astype(str).tolist())
        ax.set_title(m)
        fig.tight_layout()
        figures[m] = fig
    return figures


def plot_tensor_factors(
    contexts: pd.DataFrame,
    senders: pd.DataFrame,
    receivers: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    *,
    context_palette: dict[str, str] | None = None,
    node_palette: dict[str, str] | None = None,
    top_k_lr: int = 200,
) -> Figure:
    """Standard tensor-cell2cell factor-loading plot: one row per factor,
    four columns (contexts, ligand-receptor pairs, senders, receivers).

    Unlike most figures in this module, this is a deliberate exception to
    one-plot-per-file: comparing loadings ACROSS factors is the entire
    analytical point of a tensor decomposition (which factor is
    condition-specific, which senders/receivers/LR-pairs define it), so
    splitting factors into separate files would remove the thing the figure
    exists to show.

    contexts/senders/receivers/lr_pairs each have entities on the index and
    one column per factor (as tensor_c2c_method.py writes them). lr_pairs is
    typically hundreds to low thousands of pairs -- too many to label
    individually -- so it renders as an unlabeled bar-code-style strip
    (magnitude only), capped at top_k_lr by max loading across factors,
    while contexts/senders/receivers (a handful to a few dozen entities)
    render as normal labeled bars.
    """
    frames = {
        "contexts": contexts,
        "senders": senders,
        "receivers": receivers,
        "lr_pairs": lr_pairs,
    }
    if any(df is None or df.empty for df in frames.values()):
        return _empty_fig()

    factor_cols = list(contexts.columns)
    n_factors = len(factor_cols)
    if n_factors == 0:
        return _empty_fig()

    # Cap lr_pairs to its top_k_lr strongest rows (by max loading across
    # factors) so the barcode strip stays a reasonable size to render.
    lr_ranked = lr_pairs.loc[lr_pairs.max(axis=1).sort_values(ascending=False).index]
    lr_ranked = lr_ranked.head(top_k_lr)

    fig, axes = plt.subplots(
        n_factors,
        4,
        figsize=(16, max(2.2, 1.8 * n_factors)),
        squeeze=False,
    )
    col_frames = [
        ("Contexts", contexts, context_palette),
        ("Ligand-Receptor Pairs", lr_ranked, None),
    ]
    col_frames += [
        ("Sender Cells", senders, node_palette),
        ("Receiver Cells", receivers, node_palette),
    ]

    for row, factor in enumerate(factor_cols):
        for col, (col_title, df, palette) in enumerate(col_frames):
            ax = axes[row][col]
            vals = df[factor].to_numpy(dtype=float)
            x = np.arange(len(df))
            if palette:
                colors = [palette.get(str(i), _OTHER_GRAY) for i in df.index]
            else:
                colors = [CATEGORICAL_PALETTE[0]] * len(df)
            ax.bar(x, vals, color=colors, width=1.0 if col == 1 else 0.8, edgecolor="none")
            if col == 1:
                # Barcode strip: magnitude only, no per-pair labels (hundreds
                # to thousands of genes cannot be individually ticked).
                ax.set_xticks([])
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(list(df.index), rotation=90 if len(df) > 6 else 45, fontsize=7)
            ax.set_ylim(0, None)
            if row == 0:
                ax.set_title(col_title, fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    f"Factor {row + 1}"
                    if not str(factor).lower().startswith("factor")
                    else str(factor),
                    fontsize=10,
                )
    fig.tight_layout()
    return fig


__all__ = [
    "signed_norm",
    "celltype_palette",
    "interaction_dotplot",
    "cci_heatmap",
    "chord_diagram",
    "sankey_flow",
    "curvature_network",
    "topology_facet_figures",
    "plot_tensor_factors",
    "_OTHER_GRAY",
]
