"""Publication-grade RNA-velocity stream rendering.

Why this exists rather than calling scVelo's plotter
---------------------------------------------------
``scv.pl.velocity_embedding_stream(adata)`` with no arguments — which is what this
stage used to do — produces a figure with no group colouring, no labels, no
legend, no axes and no title: uniform grey blobs under a hairball of arrows. It
also emits non-finite coordinates that break PDF export.

So the computation is kept and the plotting is replaced: the velocity is projected
into the embedding, gridded with scVelo's own ``compute_velocity_on_grid``, and
drawn with matplotlib ``streamplot`` over group-coloured, directly-labelled cells.

Read the accompanying coherence panel before trusting arrow directions. A stream
plot draws arrows everywhere, including where the model has no support, and in
non-differentiating tissue a smooth rotational field is the signature of no real
directional process rather than of flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from cellquorum.visualization.figstyle import (
    apply_cellquorum_theme,
    get_group_palette,
)

if TYPE_CHECKING:
    import anndata as ad

_INK = "#1a1a1a"
_MUTED = "#6b6b6b"


class VelocityRenderError(RuntimeError):
    """Raised when the object lacks what a velocity figure needs."""


def resolve_group_key(
    adata: ad.AnnData,
    configured: str | None,
    candidates: tuple[str, ...],
    *,
    min_cells: int = 10,
    min_fraction: float = 0.01,
) -> str | None:
    """First categorical obs column that actually PARTITIONS the object.

    A configured key is honored on presence and 2+ populated levels: the caller
    named it, so this does not second-guess it. The candidates are inferred, and
    for those "2+ levels" turned out to be too weak a test.

    On a per-lineage arm — the whole point of the hypothesis-repo layout — the
    LEC velocity object's ``cell_type_granular`` had 13 levels of which LEC held
    1840 of 1864 cells and the other twelve held one to three stray cells each.
    That passes "2+ levels" and produces a figure that is one uniform colour with
    a dozen invisible singletons, plus a twelve-entry legend for them: the
    grey-blob-with-a-legend variant of the failure this module exists to prevent.
    Meanwhile ``leiden``, last in the candidate list, had 15 populated clusters
    that would have shown the within-lineage structure the figure is for.

    So a level counts only if it holds at least ``min_cells`` cells AND at least
    ``min_fraction`` of the object, and a candidate is usable when 2+ of its
    levels count. Candidate ORDER still decides between usable columns, because
    it encodes the caller's preference for the most specific naming available.

    If no candidate clears that bar, this falls back to the first with 2+
    populated levels at all — a weakly coloured figure still beats no figure.
    """
    if configured and configured in adata.obs:
        if adata.obs[configured].astype(str).nunique(dropna=True) >= 2:
            return configured

    threshold = max(int(min_cells), int(np.ceil(min_fraction * adata.n_obs)))
    weak_fallback: str | None = None
    for key in candidates:
        if not key or key not in adata.obs:
            continue
        counts = adata.obs[key].astype(str).value_counts()
        if int((counts >= threshold).sum()) >= 2:
            return key
        if weak_fallback is None and len(counts) >= 2:
            weak_fallback = key
    return weak_fallback


def _corner_axes(ax: Axes, xy: np.ndarray, labels: tuple[str, str]) -> None:
    """Frameless embedding axes with corner axis-name arrows."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    x_extent, y_extent = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
    x0 = xy[:, 0].min() - 0.03 * x_extent
    y0 = xy[:, 1].min() - 0.03 * y_extent
    arrow = {"arrowstyle": "-|>", "color": _INK, "lw": 0.9}
    ax.annotate("", xy=(x0 + 0.13 * x_extent, y0), xytext=(x0, y0), arrowprops=arrow)
    ax.annotate("", xy=(x0, y0 + 0.13 * y_extent), xytext=(x0, y0), arrowprops=arrow)
    ax.text(x0 + 0.14 * x_extent, y0, labels[0], fontsize=7, va="center", color=_INK)
    ax.text(x0, y0 + 0.14 * y_extent, labels[1], fontsize=7, ha="center", rotation=90, color=_INK)


def _label_groups(
    ax: Axes,
    xy: np.ndarray,
    groups: np.ndarray,
    order: list[str],
    palette: dict[str, str],
    *,
    min_cells: int = 15,
) -> None:
    """Boxed group labels at per-group medians, de-collided when adjustText exists."""
    texts = []
    for group in order:
        mask = groups == group
        if mask.sum() < min_cells:
            continue
        cx, cy = np.median(xy[mask, 0]), np.median(xy[mask, 1])
        texts.append(
            ax.text(
                cx,
                cy,
                group,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=10,
                color=_INK,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": palette.get(group, _INK),
                    "lw": 1.0,
                    "alpha": 0.92,
                },
            )
        )
    if len(texts) > 1:
        try:
            from adjustText import adjust_text

            adjust_text(
                texts,
                ax=ax,
                only_move={"text": "xy"},
                expand=(1.12, 1.22),
                force_text=(0.28, 0.36),
                arrowprops={"arrowstyle": "-", "color": "#9a9a9a", "lw": 0.4},
            )
        except Exception:  # noqa: BLE001 — cosmetic only
            pass


def velocity_stream_figure(
    adata: ad.AnnData,
    *,
    group_key: str,
    basis: str = "umap",
    palette: dict[str, str] | None = None,
    title: str | None = None,
    density: float = 2.0,
    smooth: float = 0.6,
    min_mass: float = 1.5,
    stream_density: float = 1.5,
    point_size: float = 14.0,
    figsize: tuple[float, float] = (8.8, 8.2),
) -> Figure:
    """Velocity stream over group-coloured cells. Raises VelocityRenderError if unusable.

    Requires ``obsm['X_<basis>']`` and a reprojected ``obsm['velocity_<basis>']``.
    The reprojection is attempted here when absent, which is also the guard that
    catches an object whose velocity was never projected into any embedding.
    """
    import matplotlib.pyplot as plt

    embedding_key, velocity_key = f"X_{basis}", f"velocity_{basis}"
    if embedding_key not in adata.obsm:
        raise VelocityRenderError(
            f"no obsm['{embedding_key}']; velocity cannot be drawn without an "
            "embedding (carry the caller's embeddings onto the velocity object)"
        )
    if velocity_key not in adata.obsm:
        try:
            import scvelo as scv

            scv.tl.velocity_embedding(adata, basis=basis)
        except Exception as exc:  # noqa: BLE001
            raise VelocityRenderError(f"velocity not projected onto '{basis}': {exc}") from exc
    if group_key not in adata.obs:
        raise VelocityRenderError(f"group_key '{group_key}' not in obs")

    from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid

    xy = np.asarray(adata.obsm[embedding_key], dtype=float)[:, :2]
    velocity = np.asarray(adata.obsm[velocity_key], dtype=float)[:, :2]
    finite = np.isfinite(velocity).all(axis=1) & np.isfinite(xy).all(axis=1)
    if finite.sum() < 10:
        raise VelocityRenderError("fewer than 10 cells with finite velocity")

    groups = adata.obs[group_key].astype(str).to_numpy()
    order = (
        [str(c) for c in adata.obs[group_key].cat.categories]
        if hasattr(adata.obs[group_key], "cat")
        else sorted(set(groups))
    )
    order = [g for g in order if (groups == g).any()]
    palette = palette or get_group_palette(order)

    X_grid, V_grid = compute_velocity_on_grid(
        X_emb=xy[finite],
        V_emb=velocity[finite],
        density=density,
        smooth=smooth,
        min_mass=min_mass,
        adjust_for_stream=True,
    )
    X_grid = np.asarray(X_grid, dtype=float)
    V_grid = np.asarray(V_grid, dtype=float)
    # The gridder returns a regular grid in float32, whose spacing is not
    # bit-exactly equal; matplotlib's streamplot rejects that outright. Rebuilding
    # the axes as an exact linspace over the same range is the same grid.
    xs = np.linspace(X_grid[0][0], X_grid[0][-1], X_grid[0].size)
    ys = np.linspace(X_grid[1][0], X_grid[1][-1], X_grid[1].size)
    # ``adjust_for_stream=True`` NaNs out every grid cell below ``min_mass`` — on a
    # real LEC object that is over half of them — which is how streamplot knows not
    # to draw in empty regions: it masks non-finite u/v itself. It does NOT mask
    # linewidth, which it interpolates per streamline vertex, so a NaN speed
    # propagates into the linewidth of any streamline passing a masked cell and
    # matplotlib's PDF backend then refuses the figure with "Can only output finite
    # numbers in PDF" — after writing part of the file. Hence: keep the NaNs in
    # V_grid (they carry the mask) and make the linewidth finite everywhere.
    speed = np.sqrt((V_grid**2).sum(0))
    peak = float(np.nanmax(speed)) if np.isfinite(speed).any() else 0.0
    scale = peak if peak > 0 else 1.0
    linewidth = 0.4 + 2.2 * np.nan_to_num(speed / scale, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=figsize)
    for group in order:
        mask = groups == group
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=point_size,
            c=palette.get(group, "#9a9a9a"),
            alpha=0.80,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    ax.streamplot(
        xs,
        ys,
        V_grid[0],
        V_grid[1],
        color=_INK,
        density=stream_density,
        linewidth=linewidth,
        arrowsize=1.15,
        arrowstyle="-|>",
        zorder=5,
    )
    _label_groups(ax, xy, groups, order, palette)
    ax.set_aspect("equal", adjustable="datalim")
    _corner_axes(ax, xy, (f"{basis.upper()}1", f"{basis.upper()}2"))
    ax.set_title(title or f"RNA velocity ({adata.n_obs:,} cells)", fontsize=12)
    ax.text(
        0.5,
        -0.035,
        "arrows = scVelo velocity projected into the embedding; " "line width ∝ local speed",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=_MUTED,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=palette.get(g, "#9a9a9a"),
            markeredgecolor="none",
            label=g,
        )
        for g in order
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=min(len(order), 3),
        frameon=False,
        fontsize=9.5,
    )
    fig.tight_layout()
    return fig


def velocity_diagnostics_figure(
    adata: ad.AnnData,
    *,
    group_key: str,
    basis: str = "umap",
    palette: dict[str, str] | None = None,
    title: str | None = None,
) -> Figure | None:
    """Speed and coherence maps plus coherence per group; None if neither exists.

    Reported because a stream plot always looks confident. Coherence is what says
    whether the arrows are supported by the data.
    """
    import matplotlib.pyplot as plt

    metrics = [
        (k, lab)
        for k, lab in (
            ("velocity_length", "velocity speed"),
            ("velocity_confidence", "velocity coherence"),
        )
        if k in adata.obs
    ]
    if not metrics or f"X_{basis}" not in adata.obsm:
        return None

    xy = np.asarray(adata.obsm[f"X_{basis}"], dtype=float)[:, :2]
    groups = adata.obs[group_key].astype(str).to_numpy()
    order = sorted(set(groups))
    palette = palette or get_group_palette(order)

    fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(5.2 * (len(metrics) + 1), 4.9))
    axes = np.atleast_1d(axes)
    for ax, (key, label) in zip(axes, metrics, strict=False):
        values = adata.obs[key].to_numpy(dtype=float)
        # Draw high values last so hotspots are not hidden under the bulk.
        rank = np.argsort(values)
        scatter = ax.scatter(
            xy[rank, 0],
            xy[rank, 1],
            c=values[rank],
            s=10,
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.7, aspect=18)
        ax.set_aspect("equal", adjustable="datalim")
        _corner_axes(ax, xy, (f"{basis.upper()}1", f"{basis.upper()}2"))
        ax.set_title(label, fontsize=10)

    ax = axes[-1]
    key = "velocity_confidence" if "velocity_confidence" in adata.obs else metrics[0][0]
    data = [adata.obs.loc[groups == g, key].to_numpy(dtype=float) for g in order]
    parts = ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=False)
    for patch, group in zip(parts["boxes"], order, strict=False):
        patch.set_facecolor(palette.get(group, "#9a9a9a"))
        patch.set_edgecolor(_INK)
        patch.set_linewidth(0.8)
    for element in parts["whiskers"] + parts["caps"] + parts["medians"]:
        element.set_color(_INK)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(key.replace("_", " "))
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
    ax.set_title("per group", fontsize=10)

    fig.suptitle(title or "velocity diagnostics", fontsize=12, y=1.0)
    fig.tight_layout()
    return fig


def apply_theme() -> None:
    """House theme for velocity figures."""
    apply_cellquorum_theme()


__all__ = [
    "VelocityRenderError",
    "apply_theme",
    "resolve_group_key",
    "velocity_diagnostics_figure",
    "velocity_stream_figure",
]
