"""The radial DoRothEA/CollecTRI GRN figure ("the cool circle plot").

Ports a reference lab script (a hand-rolled Mehrara-lab analysis, not a
CellQuorum module) into an engine primitive: for one compartment's paired
case-vs-control contrast, the most active transcription factors (regulators)
sit on an inner ring, grouped into data-driven sectors clustered on shared
targets; their regulated genes sit on the outer ring; edges run TF -> target,
colored by mode of regulation. Regulator squares are filled by TF activity,
target circles by their log2FC.

TF activity here is decoupler ULM run on a single DE CONTRAST (signed
-log10 p-ranked genes from ``de_table_to_ranking``), not per-cell AUCell --
a different, complementary question from the rest of the GRN stage's
pySCENIC figures: "given the observed case-vs-control fold changes, which
known regulators (DoRothEA/CollecTRI) explain that signature", versus
pySCENIC's "which co-expression-derived regulons are active in this cell".

``radial_regulon_curvature`` reuses the exact same sector/radial layout
engine to redraw the GRN stage's existing Ollivier-Ricci regulon-curvature
figure (today a force-directed hairball, ``regulon_curvature.py``'s
``curvature_network``) in the same polished style, coloring edges/nodes by
curvature instead of DE direction. It operates on the pySCENIC data-driven
regulon graph (real structure from this cohort), not the curated prior net.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Patch, PathPatch
from matplotlib.path import Path as MPath

from cellquorum.stats.multiple_testing import bh_fdr
from cellquorum.visualization import figstyle

if TYPE_CHECKING:
    import networkx as nx
    from matplotlib.axes import Axes

ACTIVATING = "#2E8B9E"  # teal -- mode of regulation +
REPRESSING = "#9B5FB0"  # violet -- mode of regulation -
R_IN, R_OUT = 0.60, 1.00
# Distinct, muted sector-arc tints (identity only, not a value ramp).
_SECTOR_TINTS = ["#E8A0B0", "#9CC77E", "#D9C08A", "#9FB6D4", "#D9A066", "#8FC7C0"]


# ---------------------------------------------------------------------------
# TF activity from a DE contrast
# ---------------------------------------------------------------------------


def tf_activity_from_de(
    de: pd.DataFrame,
    net: pd.DataFrame,
    *,
    tmin: int = 5,
    fdr: float = 0.05,
    gene_col: str = "gene",
    lfc_col: str = "logFC",
    p_col: str = "PValue",
) -> pd.DataFrame:
    """decoupler ULM TF activity from a single DE contrast (bulk, not per-cell).

    Mirrors the reference lab script's local ``tf_activity()`` helper
    (grn_subcluster.py), built from already-validated engine primitives:
    ``de_table_to_ranking`` turns the DE table into decoupler's single-row
    preranked-contrast shape, then ULM scores every TF (``net['source']``)
    against it.

    Args:
        de: DE result table (gene_col/lfc_col/p_col).
        net: Long-format source/target[/weight] regulon net (e.g. DoRothEA).
        tmin: Minimum overlapping targets required to score a TF (decoupler
            ``tmin``).
        fdr: BH-adjusted-p threshold for the ``significant`` flag.
        gene_col, lfc_col, p_col: DE table column names.

    Returns:
        DataFrame indexed by TF (``source``), columns ``score``, ``pvalue``,
        ``padj``, ``significant``. Empty (same columns, no rows) only when there
        is nothing to score: the DE table is empty, or no TF has ``tmin`` targets in it.

    Raises:
        Exception: Whatever decoupler or the DE ranking raises, unchanged. An error is not
            an empty result, and the caller reports it with its real cause.
    """
    cols = ["score", "pvalue", "padj", "significant"]
    if de is None or de.empty:
        return pd.DataFrame(columns=cols)

    from cellquorum.stages.comparative.enrichment.ranking import de_table_to_ranking

    ranking = de_table_to_ranking(de, gene_col=gene_col, lfc_col=lfc_col, p_col=p_col)
    if ranking.shape[1] == 0:
        return pd.DataFrame(columns=cols)

    # decoupler rewrites matplotlib's legend settings the first time it is imported. Contain
    # that so it cannot override the theme applied before this figure is drawn.
    with mpl.rc_context():
        import decoupler as dc

    es, pv = dc.mt.ulm(ranking, net, tmin=tmin, verbose=False)
    if es.shape[1] == 0:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(
        {"score": es.iloc[0].to_numpy(), "pvalue": pv.iloc[0].to_numpy()}, index=es.columns
    )
    out["padj"] = bh_fdr(out["pvalue"].to_numpy())
    out["significant"] = out["padj"] < fdr
    return out


# ---------------------------------------------------------------------------
# TF/target selection (DoRothEA path)
# ---------------------------------------------------------------------------


def select_tf_targets(
    tf_activity: pd.DataFrame,
    de: pd.DataFrame,
    net: pd.DataFrame,
    *,
    n_tf: int = 12,
    per_tf: int = 6,
    lfc_col: str = "logFC",
    fdr_col: str = "FDR",
) -> tuple[list[str], list[tuple[str, str, float, float]], dict[str, tuple[str, float]]]:
    """Top TFs by significance then |activity|, then each one's top DE-backed targets.

    Args:
        tf_activity: Output of :func:`tf_activity_from_de` (index=TF).
        de: DE table indexed by gene (``lfc_col``/``fdr_col`` columns).
        net: Long-format source/target[/weight] net.
        n_tf: Number of top TFs kept.
        per_tf: Number of targets kept per TF, ranked by |logFC| with a bonus
            for FDR significance.

    Returns:
        ``(tfs, edges, targets)`` where ``edges`` is a list of
        ``(tf, target_gene, sign, logFC)`` and ``targets`` maps each target
        gene to the ``(owner_tf, logFC)`` of its strongest-|logFC| regulator.
    """
    if tf_activity is None or tf_activity.empty or net is None or net.empty:
        return [], [], {}
    if de is None or de.empty:
        return [], [], {}

    tf = tf_activity.copy()
    tf["absscore"] = tf["score"].abs()
    sort_cols = ["absscore"]
    ascending = [False]
    if "significant" in tf.columns:
        sort_cols = ["significant", *sort_cols]
        ascending = [False, False]
    tf = tf.sort_values(sort_cols, ascending=ascending)
    tfs = [t for t in tf.index if t in set(net["source"])][:n_tf]

    edges: list[tuple[str, str, float, float]] = []
    targets: dict[str, tuple[str, float]] = {}
    for t in tfs:
        sub = net[net["source"] == t].copy()
        sub = sub[sub["target"].isin(de.index)]
        if sub.empty:
            continue
        sub["lfc"] = de.loc[sub["target"], lfc_col].to_numpy()
        sub["fdr"] = de.loc[sub["target"], fdr_col].to_numpy() if fdr_col in de.columns else np.nan
        sub = sub[np.isfinite(sub["lfc"])]
        sub["rank"] = sub["lfc"].abs() + 3.0 * (sub["fdr"] < 0.05).fillna(False)
        sub = sub.sort_values("rank", ascending=False).head(per_tf)
        weight_col = "weight" if "weight" in sub.columns else None
        for _, r in sub.iterrows():
            sign = float(np.sign(r[weight_col]) or 1.0) if weight_col else 1.0
            edges.append((t, r["target"], sign, float(r["lfc"])))
            if r["target"] not in targets or abs(r["lfc"]) > abs(targets[r["target"]][1]):
                targets[r["target"]] = (t, float(r["lfc"]))
    return tfs, edges, targets


# ---------------------------------------------------------------------------
# Shared radial layout (used by both the DoRothEA figure and the curvature one)
# ---------------------------------------------------------------------------


def _cluster_by_shared_children(
    members: list[str], children_of: dict[str, set[str]], n_sectors: int
) -> tuple[dict[str, int], int]:
    """Cluster members into sectors by Jaccard distance on their child sets.

    Falls back to an even round-robin split when the distance matrix is
    degenerate (every pair equally dissimilar) -- confirmed on the real
    fibroblast pySCENIC regulon graph: unlike a curated net (DoRothEA),
    whose TFs commonly share pathway targets, a data-driven regulon
    graph's top-K-by-importance targets are very often COMPLETELY
    disjoint across TFs, making every pairwise Jaccard distance exactly
    1.0. Average-linkage + fcluster's maxclust criterion then collapses
    everything to ONE cluster regardless of n_sectors (verified directly
    against scipy), producing a figure with one giant sector and a
    misleading "N target-sharing modules" subtitle. A constant distance
    matrix carries no real grouping signal either way, so spreading
    members evenly is the more honest -- and more legible -- fallback.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = len(members)
    k = int(min(n_sectors, max(1, n)))
    if n <= 2 or k <= 1:
        return {m: 0 for m in members}, k
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = children_of.get(members[i], set()), children_of.get(members[j], set())
            jac = 1.0 - (len(a & b) / len(a | b) if (a | b) else 0.0)
            dist[i, j] = dist[j, i] = jac
    Z = linkage(squareform(dist, checks=False), method="average")
    lab = fcluster(Z, t=k, criterion="maxclust")
    if len(set(lab)) == 1:
        return {members[i]: i % k for i in range(n)}, k
    return {members[i]: int(lab[i]) for i in range(n)}, k


def _layout_sectors(
    sec_ids: list[int],
    sec_members: dict[int, list[str]],
    sec_children: dict[int, list[str]],
    *,
    gap_deg: float = 6.0,
) -> tuple[dict[str, float], dict[str, float], dict[int, tuple[float, float]]]:
    """Angular layout: each sector gets an arc sized by its child count.

    Returns ``(ang_member, ang_child, sec_span)`` -- radians, member and
    child names to angle, sector id to ``(start, end)`` angle.
    """
    sizes = {s: max(len(sec_children.get(s, [])), 2) for s in sec_ids}
    total = sum(sizes.values()) or 1
    gap = np.deg2rad(gap_deg)
    avail = 2 * np.pi - gap * len(sec_ids)

    ang_member: dict[str, float] = {}
    ang_child: dict[str, float] = {}
    sec_span: dict[int, tuple[float, float]] = {}
    cur = np.pi / 2  # start at top, go clockwise
    for s in sec_ids:
        width = avail * sizes[s] / total
        a1, a0 = cur, cur - width
        sec_span[s] = (a0, a1)
        children = sec_children.get(s, [])
        for i, g in enumerate(children):
            frac = (i + 0.5) / max(len(children), 1)
            ang_child[g] = a1 - frac * width
        members = sec_members.get(s, [])
        mid, half = (a0 + a1) / 2, width * 0.30
        for i, m in enumerate(members):
            frac = (i + 0.5) / max(len(members), 1)
            ang_member[m] = (mid + half) - frac * (2 * half) if len(members) > 1 else mid
        cur = a0 - gap
    return ang_member, ang_child, sec_span


def _bezier(p0: tuple[float, float], p1: tuple[float, float], bow: float = 0.55) -> MPath:
    """Quadratic bezier, control point pulled toward the origin (bundled look)."""
    ctrl = ((p0[0] + p1[0]) / 2 * bow, (p0[1] + p1[1]) / 2 * bow)
    return MPath([p0, ctrl, p1], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])


def _pos(ang: float, r: float) -> tuple[float, float]:
    return (r * np.cos(ang), r * np.sin(ang))


def _draw_sector_arcs(
    ax: Axes,
    sec_ids: list[int],
    sec_span: dict[int, tuple[float, float]],
    sec_labels: dict[int, str],
) -> None:
    r_arc = R_OUT + 0.10
    for s in sec_ids:
        a0, a1 = sec_span[s]
        tint = _SECTOR_TINTS[s % len(_SECTOR_TINTS)]
        ax.add_patch(
            Arc(
                (0, 0),
                2 * r_arc,
                2 * r_arc,
                angle=0,
                theta1=np.rad2deg(a0),
                theta2=np.rad2deg(a1),
                color=tint,
                lw=6,
                alpha=0.85,
                zorder=1,
            )
        )
        amid = (a0 + a1) / 2
        x, y = _pos(amid, r_arc + 0.16)
        deg = np.rad2deg(amid) % 360
        rot = deg - 90 if 90 <= deg <= 270 else deg + 90
        ax.text(
            x,
            y,
            sec_labels.get(s, f"module {s}"),
            rotation=rot,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=8.4,
            color=tint,
            fontweight="bold",
        )


# ---------------------------------------------------------------------------
# The DoRothEA/CollecTRI radial figure
# ---------------------------------------------------------------------------


def radial_grn(
    tf_activity: pd.DataFrame,
    de: pd.DataFrame,
    net: pd.DataFrame,
    *,
    title: str = "Regulatory network",
    subtitle: str | None = None,
    collection_label: str = "Dorothea",
    case: str | None = None,
    control: str | None = None,
    case_color: str | None = None,
    control_color: str | None = None,
    n_tf: int = 12,
    per_tf: int = 6,
    n_sectors: int = 4,
    lfc_col: str = "logFC",
    fdr_col: str = "FDR",
) -> Figure:
    """Render the radial DoRothEA/CollecTRI GRN figure for one contrast.

    Regulators (TFs) sit on an inner ring, grouped into sectors clustered on
    shared targets; their targets sit on an outer ring colored by log2FC.
    Edge color encodes mode of regulation (activating/repressing); node color
    scales are independent (TF activity vs target log2FC), so they get two
    separate colorbars, never one shared bar.
    """
    tfs, edges, targets = select_tf_targets(
        tf_activity, de, net, n_tf=n_tf, per_tf=per_tf, lfc_col=lfc_col, fdr_col=fdr_col
    )
    if not edges:
        return figstyle.empty_figure("no TF -> target edges for this contrast")

    children_of: dict[str, set[str]] = {}
    for t, g, _sign, _lfc in edges:
        children_of.setdefault(t, set()).add(g)
    sec_of, k = _cluster_by_shared_children(tfs, children_of, n_sectors)
    sec_ids = sorted(set(sec_of.values()))
    sec_members = {s: [t for t in tfs if sec_of[t] == s] for s in sec_ids}
    sec_children = {
        s: [g for g, (owner, _lfc) in targets.items() if sec_of[owner] == s] for s in sec_ids
    }
    ang_tf, ang_tgt, sec_span = _layout_sectors(sec_ids, sec_members, sec_children)
    sec_labels = {
        s: f"{sec_members[s][0]} module" if sec_members[s] else f"module {s}" for s in sec_ids
    }

    case_color = case_color or figstyle.LE_RED
    control_color = control_color or figstyle.NORMAL_BLUE
    cmap = plt.get_cmap("RdBu_r")
    lfcs = np.array([lf for _, _, _, lf in edges])
    vlfc = max(0.5, float(np.percentile(np.abs(lfcs), 95)))
    nlfc = plt.Normalize(-vlfc, vlfc)
    acts = tf_activity.loc[tfs, "score"].to_numpy()
    vact = max(0.5, float(np.percentile(np.abs(acts), 95))) if acts.size else 0.5
    nact = plt.Normalize(-vact, vact)

    fig, ax = plt.subplots(figsize=(11.5, 11.0))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    for t, g, sign, _lfc in edges:
        if t not in ang_tf or g not in ang_tgt:
            continue
        p0, p1 = _pos(ang_tf[t], R_IN), _pos(ang_tgt[g], R_OUT)
        col = ACTIVATING if sign >= 0 else REPRESSING
        ax.add_patch(
            PathPatch(_bezier(p0, p1), fill=False, edgecolor=col, lw=0.7, alpha=0.5, zorder=1)
        )

    _draw_sector_arcs(ax, sec_ids, sec_span, sec_labels)

    for g, ang in ang_tgt.items():
        lf = targets[g][1]
        x, y = _pos(ang, R_OUT)
        ax.scatter([x], [y], s=90, c=[cmap(nlfc(lf))], edgecolors="white", linewidths=0.6, zorder=3)
        lx, ly = _pos(ang, R_OUT + 0.035)
        deg = np.rad2deg(ang) % 360
        ha = "left" if -90 < ((deg + 180) % 360 - 180) < 90 else "right"
        rot = deg if ha == "left" else deg - 180
        ax.text(
            lx,
            ly,
            g,
            rotation=rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=6.2,
            color=figstyle.TEXT,
            zorder=4,
        )

    for t in tfs:
        if t not in ang_tf:
            continue
        x, y = _pos(ang_tf[t], R_IN)
        ax.scatter(
            [x],
            [y],
            s=180,
            marker="s",
            c=[cmap(nact(float(cast("Any", tf_activity.loc[t, "score"]))))],
            edgecolors=figstyle.TEXT,
            linewidths=0.8,
            zorder=5,
        )
        ax.text(
            x,
            y - 0.055,
            t,
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color=figstyle.TEXT,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#c8ccd0", lw=0.6),
        )

    sm_lfc = plt.cm.ScalarMappable(cmap=cmap, norm=nlfc)
    sm_lfc.set_array([])
    sm_act = plt.cm.ScalarMappable(cmap=cmap, norm=nact)
    sm_act.set_array([])
    cax1 = fig.add_axes((0.055, 0.50, 0.016, 0.30))
    cax2 = fig.add_axes((0.055, 0.16, 0.016, 0.22))
    cb1 = fig.colorbar(sm_lfc, cax=cax1)
    cb1.set_label("target log$_2$FC  (●)", fontsize=8.5)
    cb1.ax.tick_params(labelsize=7.5)
    cast("Any", cb1.outline).set_visible(False)  # stubs type Spine as not callable
    cb2 = fig.colorbar(sm_act, cax=cax2)
    cb2.set_label("TF activity, ULM  (■)", fontsize=8.5)
    cb2.ax.tick_params(labelsize=7.5)
    cast("Any", cb2.outline).set_visible(False)  # stubs type Spine as not callable
    case_label = f"up in {case}" if case else "up in case"
    control_label = f"up in {control}" if control else "up in control"
    cax1.text(
        -0.6,
        1.0,
        case_label,
        transform=cax1.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=case_color,
        fontweight="bold",
        rotation=90,
    )
    cax1.text(
        -0.6,
        0.0,
        control_label,
        transform=cax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color=control_color,
        fontweight="bold",
        rotation=90,
    )

    shape_leg = ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=10,
                markerfacecolor="#b9bec4",
                markeredgecolor=figstyle.TEXT,
                label="Regulator (TF)",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=9,
                markerfacecolor="#b9bec4",
                markeredgecolor="white",
                label="Measured target",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.02, 1.02),
        frameon=False,
        fontsize=8.2,
        title="node",
        title_fontsize=8.6,
    )
    ax.add_artist(shape_leg)
    ax.legend(
        handles=[
            Patch(color=ACTIVATING, label="Activating"),
            Patch(color=REPRESSING, label="Repressing"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.02, -0.02),
        frameon=False,
        fontsize=8.2,
        title=f"{collection_label} edge",
        title_fontsize=8.6,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.925,
        subtitle
        or (
            f"top {len(tfs)} {collection_label} TFs by activity  ·  "
            f"grouped into {k} target-sharing modules  ·  {len(ang_tgt)} regulated targets"
        ),
        ha="center",
        va="top",
        fontsize=9,
        color="#6b7075",
    )
    return fig


# ---------------------------------------------------------------------------
# Curvature-on-radial-layout (pySCENIC regulon graph)
# ---------------------------------------------------------------------------


def select_curvature_targets(
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    G: nx.DiGraph,
    *,
    n_tf: int = 12,
    per_tf: int = 6,
) -> tuple[list[str], list[tuple[str, str, float, float]], dict[str, float], dict[str, float]]:
    """Top TFs by |mean node curvature|, then their strongest-weight targets.

    ``compute_ricci_curvature`` runs Ollivier-Ricci on an UNDIRECTED copy of
    the regulon graph, so ``edge_df``'s own source/target columns are not
    guaranteed to preserve the real TF -> target direction. Orientation is
    re-derived here from the original directed ``G`` (from
    ``build_regulon_graph``), not trusted from ``edge_df`` as given.

    Returns:
        ``(tfs, edges, tf_curvature, target_curvature)`` where ``edges`` is
        ``(tf, target, curvature, weight)``.
    """
    empty: tuple[
        list[str], list[tuple[str, str, float, float]], dict[str, float], dict[str, float]
    ] = (
        [],
        [],
        {},
        {},
    )
    if edge_df is None or edge_df.empty or G is None or G.number_of_edges() == 0:
        return empty

    node_curv = (
        dict(zip(node_df["node"], node_df["ricci_curvature"], strict=False))
        if node_df is not None and not node_df.empty
        else {}
    )

    oriented: list[tuple[str, str, float, float]] = []
    for _, r in edge_df.iterrows():
        u, v = str(r["source"]), str(r["target"])
        curv = float(r["ricci_curvature"])
        weight = float(r.get("weight", 1.0))
        if G.has_edge(u, v):
            oriented.append((u, v, curv, weight))
        elif G.has_edge(v, u):
            oriented.append((v, u, curv, weight))
    if not oriented:
        return empty

    tf_names = {t for t, _g, _c, _w in oriented}
    tf_rank = sorted(tf_names, key=lambda t: abs(node_curv.get(t, 0.0)), reverse=True)[:n_tf]
    keep = set(tf_rank)

    edges: list[tuple[str, str, float, float]] = []
    per_tf_rows: dict[str, list[tuple[str, float, float]]] = {}
    for t, g, curv, weight in oriented:
        if t not in keep:
            continue
        per_tf_rows.setdefault(t, []).append((g, curv, weight))
    for t, rows in per_tf_rows.items():
        rows = sorted(rows, key=lambda row: row[2], reverse=True)[:per_tf]
        for g, curv, weight in rows:
            edges.append((t, g, curv, weight))

    tf_curvature = {t: node_curv.get(t, 0.0) for t in tf_rank}
    target_curvature = {g: node_curv.get(g, 0.0) for _t, g, _c, _w in edges}
    return tf_rank, edges, tf_curvature, target_curvature


def radial_regulon_curvature(
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    G: nx.DiGraph,
    *,
    title: str = "Regulon curvature (Ollivier-Ricci)",
    n_tf: int = 12,
    per_tf: int = 6,
    n_sectors: int = 4,
) -> Figure:
    """Redraw the GRN stage's regulon-curvature graph in the radial layout.

    Same sector/bezier engine as :func:`radial_grn`, but edges and both node
    rings are colored by Ollivier-Ricci curvature (negative = bridges
    disjoint regulatory neighborhoods, positive = redundant co-regulated
    core) instead of DE direction -- operating on the pySCENIC data-driven
    regulon graph, not the curated prior net.
    """
    tfs, edges, tf_curv, tgt_curv = select_curvature_targets(
        edge_df, node_df, G, n_tf=n_tf, per_tf=per_tf
    )
    if not edges:
        return figstyle.empty_figure("no curvature edges for this graph")

    children_of: dict[str, set[str]] = {}
    for t, g, _c, _w in edges:
        children_of.setdefault(t, set()).add(g)
    sec_of, k = _cluster_by_shared_children(tfs, children_of, n_sectors)
    sec_ids = sorted(set(sec_of.values()))
    sec_members = {s: [t for t in tfs if sec_of[t] == s] for s in sec_ids}
    sec_children = {s: sorted({g for t, g, _c, _w in edges if sec_of[t] == s}) for s in sec_ids}
    ang_tf, ang_tgt, sec_span = _layout_sectors(sec_ids, sec_members, sec_children)
    sec_labels = {
        s: f"{sec_members[s][0]} module" if sec_members[s] else f"module {s}" for s in sec_ids
    }

    cmap = plt.get_cmap("RdBu_r")
    curvs = np.array(
        [c for _t, _g, c, _w in edges] + list(tf_curv.values()) + list(tgt_curv.values())
    )
    vmax = max(0.05, float(np.percentile(np.abs(curvs), 95))) if curvs.size else 0.5
    norm = plt.Normalize(-vmax, vmax)
    weights = np.array([w for _t, _g, _c, w in edges])
    wmax = float(weights.max()) if weights.size else 1.0

    fig, ax = plt.subplots(figsize=(11.5, 11.0))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    for t, g, curv, weight in edges:
        if t not in ang_tf or g not in ang_tgt:
            continue
        p0, p1 = _pos(ang_tf[t], R_IN), _pos(ang_tgt[g], R_OUT)
        ax.add_patch(
            PathPatch(
                _bezier(p0, p1),
                fill=False,
                edgecolor=cmap(norm(curv)),
                lw=0.5 + 2.0 * (weight / wmax if wmax else 0.0),
                alpha=0.7,
                zorder=1,
            )
        )

    _draw_sector_arcs(ax, sec_ids, sec_span, sec_labels)

    for g, ang in ang_tgt.items():
        x, y = _pos(ang, R_OUT)
        ax.scatter(
            [x],
            [y],
            s=90,
            c=[cmap(norm(tgt_curv.get(g, 0.0)))],
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        lx, ly = _pos(ang, R_OUT + 0.035)
        deg = np.rad2deg(ang) % 360
        ha = "left" if -90 < ((deg + 180) % 360 - 180) < 90 else "right"
        rot = deg if ha == "left" else deg - 180
        ax.text(
            lx,
            ly,
            g,
            rotation=rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=6.2,
            color=figstyle.TEXT,
            zorder=4,
        )

    for t in tfs:
        if t not in ang_tf:
            continue
        x, y = _pos(ang_tf[t], R_IN)
        ax.scatter(
            [x],
            [y],
            s=180,
            marker="s",
            c=[cmap(norm(tf_curv.get(t, 0.0)))],
            edgecolors=figstyle.TEXT,
            linewidths=0.8,
            zorder=5,
        )
        ax.text(
            x,
            y - 0.055,
            t,
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color=figstyle.TEXT,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#c8ccd0", lw=0.6),
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes((0.055, 0.35, 0.016, 0.30))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Ollivier-Ricci curvature", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    cast("Any", cb.outline).set_visible(False)  # stubs type Spine as not callable
    cax.text(
        -0.6,
        1.0,
        "bridging",
        transform=cax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=to_rgb(cmap(0.05)),
        fontweight="bold",
        rotation=90,
    )
    cax.text(
        -0.6,
        0.0,
        "redundant core",
        transform=cax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color=to_rgb(cmap(0.95)),
        fontweight="bold",
        rotation=90,
    )

    shape_leg = ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=10,
                markerfacecolor="#b9bec4",
                markeredgecolor=figstyle.TEXT,
                label="Regulator (TF)",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=9,
                markerfacecolor="#b9bec4",
                markeredgecolor="white",
                label="Target gene",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.02, 1.02),
        frameon=False,
        fontsize=8.2,
        title="node",
        title_fontsize=8.6,
    )
    ax.add_artist(shape_leg)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.925,
        f"top {len(tfs)} TFs by |curvature|  ·  grouped into {k} target-sharing modules  ·  "
        f"{len(ang_tgt)} target genes",
        ha="center",
        va="top",
        fontsize=9,
        color="#6b7075",
    )
    return fig


__all__ = [
    "radial_grn",
    "radial_regulon_curvature",
    "select_curvature_targets",
    "select_tf_targets",
    "tf_activity_from_de",
]
