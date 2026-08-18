"""Mast-cell/LE-KC style publication QC figures.

This module intentionally mirrors the visual grammar used in the user's
``mast_cell_scrna/scripts/03_figures/supplementary_qc/qc_diagnostics.py`` and
``le_kc_signaling_hubs/src/lekc/figstyle.py`` projects:

* individual compact panels, not a generic multi-metric grid;
* Normal/LE colors ``#24608F`` and ``#C52A45``;
* 7--8 pt journal typography with embedded editable PDF/SVG fonts;
* per-sample boxplots with horizontal QC cutoff bars;
* ECDF doublet-score panel and optional UMAP/scree panels.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.visualization.figstyle import (
    DOUBLET_COLOR,
    LE_RED,
    NORMAL_COLOR,
    QC_FAIL_COLOR,
    SEQUENTIAL_CMAP,
    TEXT_COLOR,
    add_panel_label,
    set_publication_style,
)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes


NORMAL = NORMAL_COLOR
LE = LE_RED
QC_FAIL = QC_FAIL_COLOR
DOUBLET = DOUBLET_COLOR
TEXT = TEXT_COLOR


class QCPublicationFigureError(CellQuorumDataError):
    """Report failures while building publication-style QC figures."""


def write_publication_qc_figures(
    adata: ad.AnnData,
    output_dir: str | Path,
    *,
    thresholds: str | Path | pd.DataFrame | None = None,
    patient_key: str = "patient_id",
    sample_key: str = "sample_id",
    condition_key: str = "condition",
    normal_label: str = "Normal",
    disease_label: str = "Lymphedema",
    dpi: int = 500,
    combined_qa: bool = True,
    formats: Sequence[str] = ("png", "pdf", "svg"),
) -> list[Path]:
    """Write mast-cell-style publication QC panels.

    Args:
        adata: QC AnnData with cell-level metrics in ``obs``.
        output_dir: Directory where figures are written.
        thresholds: Optional CellQuorum ``thresholds.csv`` path or DataFrame.
        patient_key: Observation column containing patient IDs.
        sample_key: Observation column containing library/sample IDs.
        condition_key: Observation column containing condition labels.
        normal_label: Label for the control condition.
        disease_label: Label for the disease condition.
        dpi: Raster output resolution. The mast-cell QC script used 500.
        combined_qa: Also write a compact combined A--G QA sheet.

    Returns:
        Written figure paths.
    """

    if not isinstance(adata, ad.AnnData):
        raise QCPublicationFigureError(
            "write_publication_qc_figures expected an AnnData object. "
            f"Received: {type(adata).__name__}."
        )

    required = [patient_key, condition_key]
    missing = [col for col in required if col not in adata.obs.columns]
    if missing:
        raise QCPublicationFigureError(
            "Publication QC figures require obs column(s): " + ", ".join(missing)
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _set_style()
    frame = _metric_frame(
        adata,
        patient_key=patient_key,
        sample_key=sample_key,
        condition_key=condition_key,
        normal_label=normal_label,
        disease_label=disease_label,
    )
    threshold_table = _read_thresholds(thresholds)
    threshold_map = _threshold_map(threshold_table)

    written: list[Path] = []
    written.extend(
        _render_individual_panels(
            adata,
            frame,
            output_path,
            threshold_map=threshold_map,
            threshold_table=threshold_table,
            normal_label=normal_label,
            disease_label=disease_label,
            dpi=dpi,
            formats=formats,
        )
    )
    if combined_qa:
        written.extend(
            _render_combined_qa(
                adata,
                frame,
                output_path,
                threshold_map=threshold_map,
                normal_label=normal_label,
                disease_label=disease_label,
                dpi=dpi,
                formats=formats,
            )
        )
    return written


def _set_style() -> None:
    """Apply the shared mast-cell/LE-KC small-panel publication style."""

    set_publication_style(dpi=500, small=True)


def _metric_frame(
    adata: ad.AnnData,
    *,
    patient_key: str,
    sample_key: str,
    condition_key: str,
    normal_label: str,
    disease_label: str,
) -> pd.DataFrame:
    """Return CellQuorum QC obs as the mast-cell plotting frame."""

    metric_aliases = {
        "pct_counts_mito": ["pct_counts_mito", "pct_counts_mt"],
        "pct_counts_ribo": ["pct_counts_ribo"],
        "pct_counts_hemoglobin": ["pct_counts_hemoglobin", "pct_counts_hb"],
        "doublet_score": ["doublet_score"],
        "predicted_doublet": ["predicted_doublet"],
        "n_genes_by_counts": ["n_genes_by_counts"],
        "log1p_n_genes_by_counts": ["log1p_n_genes_by_counts"],
        "total_counts": ["total_counts"],
        "log1p_total_counts": ["log1p_total_counts"],
        "pct_counts_in_top_20_genes": ["pct_counts_in_top_20_genes"],
        "cellquorum_qc_keep": ["cellquorum_qc_keep", "keep"],
    }
    resolved: dict[str, str] = {}
    for target, candidates in metric_aliases.items():
        for candidate in candidates:
            if candidate in adata.obs.columns:
                resolved[target] = candidate
                break

    required_metrics = [
        "pct_counts_mito",
        "pct_counts_ribo",
        "pct_counts_hemoglobin",
        "n_genes_by_counts",
        "total_counts",
    ]
    missing_metrics = [metric for metric in required_metrics if metric not in resolved]
    if missing_metrics:
        raise QCPublicationFigureError(
            "Publication QC figures require metric column(s): " + ", ".join(missing_metrics)
        )

    if sample_key not in adata.obs.columns:
        sample_key = patient_key

    decision_columns = [
        col
        for col in adata.obs.columns
        if col.startswith("cellquorum_qc_mad_") or col.startswith("mad_")
    ]
    columns = [patient_key, sample_key, condition_key, *resolved.values(), *decision_columns]
    df = adata.obs.loc[:, list(dict.fromkeys(columns))].copy()
    rename = {source: target for target, source in resolved.items()}
    df = df.rename(columns=rename)
    df[patient_key] = df[patient_key].astype(str)
    df[sample_key] = df[sample_key].astype(str)
    df[condition_key] = df[condition_key].astype(str)
    df["patient_id"] = df[patient_key]
    df["sample_id"] = df[sample_key]
    df["condition"] = df[condition_key]
    df["condition_display"] = (
        df[condition_key]
        .map(
            {
                normal_label: "N",
                disease_label: "LE",
                "Normal": "N",
                "Lymphedema": "LE",
                "LE": "LE",
            }
        )
        .fillna(df[condition_key])
    )
    df["sample_label"] = df[patient_key] + " " + df["condition_display"]
    return df


def _read_thresholds(thresholds: str | Path | pd.DataFrame | None) -> pd.DataFrame:
    """Load a CellQuorum threshold table without inventing rows."""

    if thresholds is None:
        return pd.DataFrame()
    return pd.read_csv(thresholds) if isinstance(thresholds, str | Path) else thresholds.copy()


def _threshold_map(table: pd.DataFrame) -> dict[str, list[float]]:
    """Parse CellQuorum thresholds into plotting values by metric."""

    defaults: dict[str, list[float]] = {
        "pct_counts_mito": [20.0],
        "doublet_score": [0.40],
        "n_genes_by_counts": [200.0, 6000.0],
    }
    if table.empty or "metric" not in table.columns:
        return defaults

    out: dict[str, list[float]] = {}
    for metric, aliases in {
        "pct_counts_mito": ["pct_counts_mito", "pct_counts_mt"],
        "pct_counts_ribo": ["pct_counts_ribo"],
        "pct_counts_hemoglobin": ["pct_counts_hemoglobin", "pct_counts_hb"],
        "doublet_score": ["doublet_score"],
        "n_genes_by_counts": ["n_genes_by_counts"],
        "total_counts": ["total_counts"],
    }.items():
        rows = table.loc[table["metric"].isin(aliases)]
        values: list[float] = []
        for bound in ("lower", "upper"):
            if bound in rows.columns:
                values.extend(rows[bound].dropna().astype(float).tolist())
        if values:
            out[metric] = sorted(set(values))

    # Preserve mast-cell visual defaults when CellQuorum has no explicit threshold.
    for key, values in defaults.items():
        out.setdefault(key, values)
    return out


def _sample_order(frame: pd.DataFrame) -> list[str]:
    """Order samples as P1 N, P1 LE, P2 N, P2 LE, ..."""

    def patient_sort(value: str) -> tuple[str, int | str]:
        prefix = "".join(ch for ch in value if not ch.isdigit())
        digits = "".join(ch for ch in value if ch.isdigit())
        return prefix, int(digits) if digits else value

    pairs = (
        frame[["patient_id", "condition_display", "sample_label"]]
        .drop_duplicates()
        .sort_values(
            ["patient_id", "condition_display"],
            key=lambda col: col.map(patient_sort) if col.name == "patient_id" else col,
        )
    )
    order: list[str] = []
    for patient in sorted(frame["patient_id"].unique(), key=patient_sort):
        sub = pairs.loc[pairs["patient_id"].eq(patient)]
        for condition in ("N", "LE"):
            labels = sub.loc[sub["condition_display"].eq(condition), "sample_label"].tolist()
            order.extend(labels)
        extras = sub.loc[~sub["condition_display"].isin(["N", "LE"]), "sample_label"].tolist()
        order.extend(extras)
    return order


def _clean_axes(ax: Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#E9ECEF", linewidth=0.45)


def _panel_label(ax: Axes, value: str) -> None:
    add_panel_label(ax, value, x=-0.14, y=1.10)


def _plot_metric(
    ax: Axes,
    frame: pd.DataFrame,
    metric: str,
    *,
    thresholds: Iterable[float] | None,
    title: str,
    ylabel: str,
    normal_label: str,
    disease_label: str,
) -> None:
    """Mast-cell-style per-sample boxplot with QC cutoff bars."""

    from matplotlib.lines import Line2D

    order = _sample_order(frame)
    values = [
        frame.loc[frame["sample_label"].eq(label), metric].dropna().to_numpy(dtype=float)
        for label in order
    ]
    boxes = ax.boxplot(
        values,
        positions=np.arange(len(order)),
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        boxprops={"linewidth": 0.65, "edgecolor": "#4D5357"},
        medianprops={"linewidth": 1.0, "color": "white"},
        whiskerprops={"linewidth": 0.65, "color": "#4D5357"},
        capprops={"linewidth": 0.65, "color": "#4D5357"},
    )
    for box, label in zip(boxes["boxes"], order, strict=False):
        box.set_facecolor(LE if label.endswith("LE") else NORMAL)
        box.set_alpha(0.88)

    threshold_values = [float(v) for v in thresholds or [] if np.isfinite(float(v))]
    for cutoff in threshold_values:
        for i in range(len(order)):
            ax.plot(
                [i - 0.34, i + 0.34],
                [cutoff, cutoff],
                color="#111111",
                linewidth=1.1,
                zorder=5,
            )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(order)), order)
    ax.tick_params(axis="x", rotation=45)
    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=5, color=NORMAL, label=normal_label),
        Line2D([], [], marker="s", linestyle="", markersize=5, color=LE, label="LE"),
    ]
    if threshold_values:
        handles.append(
            Line2D([], [], linestyle="-", linewidth=1.1, color="#111111", label="QC cutoff")
        )
    ax.legend(handles=handles, loc="upper right", frameon=False, handletextpad=0.3)
    _clean_axes(ax)


def _plot_doublet_scores(ax: Axes, frame: pd.DataFrame) -> None:
    """ECDF doublet separation panel, matching the mast-cell script."""

    if "doublet_score" not in frame.columns:
        ax.text(0.5, 0.5, "No doublet scores", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    if "predicted_doublet" in frame.columns:
        calls = frame["predicted_doublet"].fillna(False).astype(bool).to_numpy()
    elif "cellquorum_qc_keep" in frame.columns:
        calls = ~frame["cellquorum_qc_keep"].fillna(True).astype(bool).to_numpy()
    else:
        calls = np.zeros(len(frame), dtype=bool)

    scores = frame["doublet_score"].to_numpy(dtype=float)
    classes = {"Singlet": ~calls, "Doublet": calls}
    colors = {"Singlet": "#66727A", "Doublet": DOUBLET}
    for label, mask in classes.items():
        values = np.sort(scores[mask & np.isfinite(scores)])
        if len(values) == 0:
            continue
        cumulative = np.arange(1, len(values) + 1) / len(values)
        ax.plot(
            values,
            cumulative,
            color=colors[label],
            linewidth=1.4,
            label=f"{label} (n={len(values):,})",
        )
        median = float(np.median(values))
        ax.scatter(
            median,
            0.5,
            s=18,
            color=colors[label],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores):
        ax.set_xlim(left=0, right=float(np.quantile(finite_scores, 0.9995)) * 1.03)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Doublet score")
    ax.set_ylabel("Cumulative fraction of cells")
    ax.set_title("Doublet-score separation (ECDF)")
    ax.legend(frameon=False, loc="lower right")
    _clean_axes(ax)


def _plot_doublet_umap(ax: Axes, adata: ad.AnnData, frame: pd.DataFrame) -> None:
    """Optional doublet/fail overlay on UMAP using LE-KC embedding style."""

    if "X_umap" not in adata.obsm:
        ax.text(0.5, 0.5, "No UMAP embedding", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    xy = np.asarray(adata.obsm["X_umap"])
    if "predicted_doublet" in frame.columns:
        calls = frame["predicted_doublet"].fillna(False).astype(bool).to_numpy()
        label = f"{calls.sum():,} predicted doublets"
    elif "cellquorum_qc_keep" in frame.columns:
        calls = ~frame["cellquorum_qc_keep"].fillna(True).astype(bool).to_numpy()
        label = f"{calls.sum():,} QC-failed cells"
    else:
        calls = np.zeros(adata.n_obs, dtype=bool)
        label = "No doublet/QC calls"

    rng = np.random.default_rng(42)
    background = np.flatnonzero(~calls)
    if len(background) > 45_000:
        background = rng.choice(background, 45_000, replace=False)
    ax.scatter(
        xy[background, 0],
        xy[background, 1],
        s=0.25,
        color="#C7CCD0",
        alpha=0.5,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        xy[calls, 0],
        xy[calls, 1],
        s=0.8,
        color=DOUBLET,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("QC flags on embedding")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.02, 0.02, label, transform=ax.transAxes, fontsize=5.8, color=DOUBLET)


def _plot_scree(ax: Axes, adata: ad.AnnData) -> None:
    """PCA variance panel using mast-cell supplementary-QC styling."""

    pca = adata.uns.get("pca", {})
    variance_ratio = np.asarray(pca.get("variance_ratio", []), dtype=float)
    if variance_ratio.size == 0:
        ax.text(0.5, 0.5, "No PCA variance", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    variance = variance_ratio[:50] * 100
    cumulative = np.cumsum(variance_ratio[:50]) * 100
    x = np.arange(1, len(variance) + 1)
    ax.bar(x, variance, color="#A4B9CF", width=0.82, edgecolor="white", linewidth=0.2)
    ax.plot(x, variance, color=NORMAL, linewidth=0.8, marker="o", markersize=1.8)
    if len(x) >= 30:
        ax.axvline(30, color=LE, linestyle="--", linewidth=0.8)
        ax.text(30.7, float(variance.max()) * 0.92, "30 PCs", color=LE, fontsize=5.8, va="top")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    _clean_axes(ax)
    ax2 = ax.twinx()
    ax2.plot(x, cumulative, color="#2A9D8F", linewidth=1.0, marker="o", markersize=1.7)
    ax2.set_ylabel("Cumulative variance (%)", rotation=270, labelpad=10)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, max(100, float(cumulative.max()) * 1.05))
    ax.set_title("PCA variance and selected dimensions")


def _mad_specs(table: pd.DataFrame) -> list[dict[str, object]]:
    """Return MAD threshold specs in the canonical CellQuorum QC order."""

    if table.empty:
        return []
    required = {"metric", "rule_name", "source", "lower", "upper"}
    if not required.issubset(table.columns):
        return []
    mad_rows = table.loc[
        table["source"].astype(str).isin(["mad", "mad_mito"])
        | table["rule_name"].astype(str).str.startswith("mad_")
    ].copy()
    if mad_rows.empty:
        return []
    order = [
        ("log1p_total_counts", "log1p total counts"),
        ("log1p_n_genes_by_counts", "log1p detected genes"),
        ("pct_counts_in_top_20_genes", "Top-20 gene fraction (%)"),
        ("pct_counts_mito", "Mitochondrial reads (%)"),
    ]
    specs: list[dict[str, object]] = []
    for metric, title in order:
        rows = mad_rows.loc[mad_rows["metric"].eq(metric)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        lower = row["lower"] if pd.notna(row["lower"]) else None
        upper = row["upper"] if pd.notna(row["upper"]) else None
        specs.append(
            {
                "metric": metric,
                "title": title,
                "rule_name": str(row["rule_name"]),
                "lower": None if lower is None else float(lower),
                "upper": None if upper is None else float(upper),
            }
        )
    return specs


def _plot_mad_thresholds(frame: pd.DataFrame, specs: list[dict[str, object]]) -> plt.Figure:
    """Render the explicit MAD outlier threshold audit panel."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_panels = len(specs)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 3.05 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, spec in enumerate(specs):
        ax = axes_flat[idx]
        metric = str(spec["metric"])
        rule_name = str(spec["rule_name"])
        if metric not in frame.columns:
            ax.text(0.5, 0.5, f"Missing {metric}", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        values = frame[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        fail_col = f"cellquorum_qc_{rule_name}"
        if fail_col not in frame.columns and rule_name in frame.columns:
            fail_col = rule_name
        if fail_col in frame.columns:
            fail_mask = frame[fail_col].fillna(False).astype(bool).to_numpy()
        else:
            fail_mask = np.zeros(len(frame), dtype=bool)
            lower = spec.get("lower")
            upper = spec.get("upper")
            metric_values = frame[metric].to_numpy(dtype=float)
            if lower is not None:
                fail_mask |= metric_values < float(lower)
            if upper is not None:
                fail_mask |= metric_values > float(upper)

        all_values = frame[metric].to_numpy(dtype=float)
        finite_mask = np.isfinite(all_values)
        pass_values = all_values[finite_mask & ~fail_mask]
        fail_values = all_values[finite_mask & fail_mask]
        finite_values = all_values[finite_mask]
        if finite_values.size == 0:
            ax.text(
                0.5,
                0.5,
                f"No values for {metric}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            continue

        bins = np.linspace(
            float(np.quantile(finite_values, 0.002)),
            float(np.quantile(finite_values, 0.998)),
            54,
        )
        ax.hist(
            pass_values,
            bins=bins,
            color="#AEB7BE",
            alpha=0.70,
            label=f"Within bounds (n={len(pass_values):,})",
        )
        if fail_values.size:
            ax.hist(
                fail_values,
                bins=bins,
                color=DOUBLET,
                alpha=0.82,
                label=f"MAD outlier (n={len(fail_values):,})",
            )

        for bound_name, color, align in [("lower", NORMAL, "right"), ("upper", LE, "left")]:
            bound = spec.get(bound_name)
            if bound is None:
                continue
            ax.axvline(float(bound), color=color, linestyle="--", linewidth=1.1)
            ymax = ax.get_ylim()[1]
            x_offset = -0.012 if align == "right" else 0.012
            ax.text(
                float(bound) + x_offset * (bins.max() - bins.min()),
                ymax * 0.92,
                f"{bound_name}: {float(bound):.2f}",
                rotation=90,
                ha=align,
                va="top",
                fontsize=5.8,
                color=color,
            )

        ax.set_title(str(spec["title"]))
        ax.set_xlabel(metric)
        ax.set_ylabel("Cells")
        ax.text(
            0.02,
            0.96,
            rule_name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color=TEXT,
        )
        _clean_axes(ax)

    for ax in axes_flat[n_panels:]:
        ax.set_axis_off()
    axes_flat[0].legend(
        handles=[
            Line2D(
                [],
                [],
                marker="s",
                linestyle="",
                markersize=5,
                color="#AEB7BE",
                label="Within MAD bounds",
            ),
            Line2D(
                [],
                [],
                marker="s",
                linestyle="",
                markersize=5,
                color=DOUBLET,
                label="MAD outlier",
            ),
            Line2D([], [], linestyle="--", linewidth=1.1, color="#111111", label="MAD cutoff"),
        ],
        loc="upper right",
        frameon=False,
        handletextpad=0.3,
    )
    fig.suptitle("MAD-based QC outlier thresholds", y=0.995, fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def _plot_umi_detected_gene_gradient(
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    normal_label: str,
    disease_label: str,
    dpi: int,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    max_points: int = 65_000,
) -> list[Path]:
    """Render mast-cell-style UMI versus detected-gene mito-gradient panels."""

    import matplotlib.pyplot as plt

    required = {"condition", "total_counts", "n_genes_by_counts", "pct_counts_mito"}
    if not required.issubset(frame.columns):
        return []

    columns = list(required)
    if "cellquorum_qc_keep" in frame.columns:
        columns.append("cellquorum_qc_keep")
    plot_frame = frame.loc[:, columns].copy()
    plot_frame["condition"] = plot_frame["condition"].astype(str)
    plot_frame["retained"] = (
        plot_frame.get(
            "cellquorum_qc_keep",
            pd.Series(True, index=plot_frame.index),
        )
        .fillna(True)
        .astype(bool)
    )

    vmax = max(1.0, float(np.nanpercentile(plot_frame["pct_counts_mito"], 99.5)))
    xlim = (
        max(1.0, float(plot_frame["total_counts"].min()) * 0.85),
        float(plot_frame["total_counts"].max()) * 1.15,
    )
    ylim = (
        max(1.0, float(plot_frame["n_genes_by_counts"].min()) * 0.85),
        float(plot_frame["n_genes_by_counts"].max()) * 1.15,
    )
    rng = np.random.default_rng(42)
    written: list[Path] = []

    condition_specs = [
        (normal_label, "Normal", "qc_panel_J_umi_detected_genes_normal"),
        (disease_label, "LE", "qc_panel_K_umi_detected_genes_le"),
    ]
    for condition_value, display_label, stem in condition_specs:
        sub = plot_frame.loc[plot_frame["condition"].eq(condition_value)].copy()
        if sub.empty and condition_value == disease_label:
            sub = plot_frame.loc[plot_frame["condition"].eq("LE")].copy()
        if sub.empty:
            continue
        if len(sub) > max_points:
            sub = sub.iloc[rng.choice(len(sub), max_points, replace=False)]
        sub = sub.iloc[np.argsort(sub["pct_counts_mito"].to_numpy())]

        fig, ax = plt.subplots(figsize=(4.1, 3.7))
        points = ax.scatter(
            sub["total_counts"],
            sub["n_genes_by_counts"],
            c=sub["pct_counts_mito"],
            cmap=SEQUENTIAL_CMAP,
            vmin=0,
            vmax=vmax,
            s=1.0,
            alpha=0.58,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Total UMI counts")
        ax.set_ylabel("Detected genes per cell")
        ax.set_title(display_label, fontweight="bold", pad=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#E6E9EB", linewidth=0.45, alpha=0.8)
        retained = int(sub["retained"].sum())
        total = int(len(sub))
        ax.text(
            0.02,
            0.98,
            f"{total:,} input cells | {retained:,} retained cells",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
            color=TEXT,
        )
        cbar = fig.colorbar(points, ax=ax, pad=0.025, fraction=0.045)
        cbar.set_label("Mitochondrial reads (%)")
        cbar.outline.set_linewidth(0.5)
        fig.tight_layout()
        written.extend(_save_panel(fig, out_dir, stem, dpi=dpi, formats=formats))

    return written


def _plot_condition_publication_violins(
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    normal_label: str,
    disease_label: str,
    dpi: int,
    formats: Sequence[str] = ("png", "pdf", "svg"),
) -> list[Path]:
    """Render the LE-KC cohort condition-comparison QC violin panel."""

    import matplotlib.pyplot as plt

    try:
        from scipy.stats import mannwhitneyu
    except Exception:  # pragma: no cover - scipy is expected in supported envs
        mannwhitneyu = None

    condition_col = "condition"
    metrics = [
        ("n_genes_by_counts", "Genes per cell"),
        ("total_counts", "Counts per cell"),
        ("pct_counts_mito", "% Mitochondrial"),
    ]
    if condition_col not in frame.columns or any(
        metric not in frame.columns for metric, _ in metrics
    ):
        return []

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, (metric, label) in zip(axes, metrics, strict=False):
        plot_df = frame[[condition_col, metric]].dropna()
        n_vals = plot_df.loc[plot_df[condition_col].eq(normal_label), metric]
        le_vals = plot_df.loc[plot_df[condition_col].eq(disease_label), metric]
        if n_vals.empty or le_vals.empty:
            ax.text(0.5, 0.5, "Missing condition", transform=ax.transAxes, ha="center")
            ax.set_axis_off()
            continue

        parts = ax.violinplot(
            [n_vals, le_vals],
            positions=[0, 1],
            showmeans=False,
            showmedians=False,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#6B8EAD", "#C8817A"][i])
            pc.set_edgecolor("black")
            pc.set_linewidth(1.2)
            pc.set_alpha(0.7)

        ax.boxplot(
            [n_vals, le_vals],
            positions=[0, 1],
            widths=0.15,
            patch_artist=False,
            boxprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 1.5},
            capprops={"color": "black", "linewidth": 1.5},
            medianprops={"color": "black", "linewidth": 2},
        )

        if mannwhitneyu is not None and len(n_vals) and len(le_vals):
            _, p_val = mannwhitneyu(n_vals, le_vals, alternative="two-sided")
            y_max = float(plot_df[metric].max())
            ax.text(0.5, y_max * 1.05, f"Wilcoxon p = {p_val:.2e}", ha="center", fontsize=9)
            ax.set_ylim(top=y_max * 1.16)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"{normal_label}\n(n={len(n_vals)})", f"LE\n(n={len(le_vals)})"],
            fontsize=10,
        )
        ax.set_ylabel(label, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return _save_panel(
        fig, out_dir, "qc_panel_L_by_condition_publication", dpi=dpi, formats=formats
    )


def _plot_cells_per_sample(
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    normal_label: str,
    disease_label: str,
    dpi: int,
    formats: Sequence[str] = ("png", "pdf", "svg"),
) -> list[Path]:
    """Render the LE-KC cohort cells-per-sample QC bar chart."""

    import matplotlib.pyplot as plt

    required = {"sample_id", "condition"}
    if not required.issubset(frame.columns):
        return []

    sample_counts = frame.groupby("sample_id")["condition"].first().to_frame("condition")
    sample_counts["n_cells"] = frame["sample_id"].value_counts()
    sample_counts = sample_counts.sort_values("n_cells", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        normal_label: "#4C72B0",
        disease_label: "#C44E52",
        "Normal": "#4C72B0",
        "Lymphedema": "#C44E52",
        "LE": "#C44E52",
    }
    ax.bar(
        range(len(sample_counts)),
        sample_counts["n_cells"],
        color=[colors.get(c, "#888888") for c in sample_counts["condition"]],
    )
    ax.set_xticks(range(len(sample_counts)))
    ax.set_xticklabels(sample_counts.index, rotation=90, fontsize=8)
    ax.set_ylabel("Cells")
    ax.set_title("Cells per sample")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_panel(fig, out_dir, "qc_panel_M_cells_per_sample", dpi=dpi, formats=formats)


def _save_panel(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    *,
    dpi: int,
    formats: Sequence[str] = ("png", "pdf", "svg"),
) -> list[Path]:
    paths = [out_dir / f"{stem}.{ext}" for ext in formats]
    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def _render_individual_panels(
    adata: ad.AnnData,
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    threshold_map: dict[str, list[float]],
    threshold_table: pd.DataFrame,
    normal_label: str,
    disease_label: str,
    dpi: int,
    formats: Sequence[str] = ("png", "pdf", "svg"),
) -> list[Path]:
    import matplotlib.pyplot as plt

    written: list[Path] = []
    metric_specs = [
        (
            "A",
            "pct_counts_mito",
            "Mitochondrial content",
            "Mitochondrial reads (%)",
            "qc_panel_A_mitochondrial_content",
        ),
        (
            "B",
            "pct_counts_ribo",
            "Ribosomal content",
            "Ribosomal reads (%)",
            "qc_panel_B_ribosomal_content",
        ),
        (
            "C",
            "pct_counts_hemoglobin",
            "Hemoglobin content",
            "Hemoglobin reads (%)",
            "qc_panel_C_hemoglobin_content",
        ),
    ]
    for letter, metric, title, ylabel, stem in metric_specs:
        if metric not in frame.columns:
            continue
        fig, ax = plt.subplots(figsize=(4.2, 3.3))
        _plot_metric(
            ax,
            frame,
            metric,
            thresholds=threshold_map.get(metric),
            title=title,
            ylabel=ylabel,
            normal_label=normal_label,
            disease_label=disease_label,
        )
        if metric == "pct_counts_hemoglobin":
            ax.set_yscale("symlog", linthresh=0.01)
            max_value = max(float(frame[metric].max()), *(threshold_map.get(metric) or [0.0]), 0.01)
            ax.set_ylim(0, max_value * 1.18)
        _panel_label(ax, letter)
        fig.tight_layout()
        written.extend(_save_panel(fig, out_dir, stem, dpi=dpi, formats=formats))

    if "doublet_score" in frame.columns:
        fig, ax = plt.subplots(figsize=(3.2, 3.3))
        _plot_doublet_scores(ax, frame)
        _panel_label(ax, "D")
        fig.tight_layout()
        written.extend(
            _save_panel(fig, out_dir, "qc_panel_D_doublet_scores", dpi=dpi, formats=formats)
        )

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    _plot_doublet_umap(ax, adata, frame)
    _panel_label(ax, "E")
    fig.tight_layout()
    written.extend(_save_panel(fig, out_dir, "qc_panel_E_qc_umap", dpi=dpi, formats=formats))

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    _plot_scree(ax, adata)
    _panel_label(ax, "F")
    fig.tight_layout()
    written.extend(_save_panel(fig, out_dir, "qc_panel_F_pca_scree", dpi=dpi, formats=formats))

    for letter, metric, title, ylabel, stem in [
        (
            "G",
            "n_genes_by_counts",
            "Genes detected",
            "Detected genes per cell",
            "qc_panel_G_detected_genes",
        ),
        ("H", "total_counts", "UMI counts", "Total UMI counts", "qc_panel_H_umi_counts"),
    ]:
        if metric not in frame.columns:
            continue
        fig, ax = plt.subplots(figsize=(4.2, 3.3))
        _plot_metric(
            ax,
            frame,
            metric,
            thresholds=threshold_map.get(metric),
            title=title,
            ylabel=ylabel,
            normal_label=normal_label,
            disease_label=disease_label,
        )
        _panel_label(ax, letter)
        fig.tight_layout()
        written.extend(_save_panel(fig, out_dir, stem, dpi=dpi, formats=formats))

    mad_specs = _mad_specs(threshold_table)
    if mad_specs:
        fig = _plot_mad_thresholds(frame, mad_specs)
        written.extend(
            _save_panel(fig, out_dir, "qc_panel_I_mad_thresholds", dpi=dpi, formats=formats)
        )

    written.extend(
        _plot_condition_publication_violins(
            frame,
            out_dir,
            normal_label=normal_label,
            disease_label=disease_label,
            dpi=dpi,
            formats=formats,
        )
    )
    written.extend(
        _plot_cells_per_sample(
            frame,
            out_dir,
            normal_label=normal_label,
            disease_label=disease_label,
            dpi=dpi,
            formats=formats,
        )
    )
    written.extend(
        _plot_umi_detected_gene_gradient(
            frame,
            out_dir,
            normal_label=normal_label,
            disease_label=disease_label,
            dpi=dpi,
            formats=formats,
        )
    )

    return written


def _render_combined_qa(
    adata: ad.AnnData,
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    threshold_map: dict[str, list[float]],
    normal_label: str,
    disease_label: str,
    dpi: int,
    formats: Sequence[str] = ("png", "pdf", "svg"),
) -> list[Path]:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.8, 6.8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.08], wspace=0.48, hspace=0.62)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3]),
    ]
    metric_panels = [
        (axes[0], "pct_counts_mito", "Mitochondrial content", "Mitochondrial reads (%)"),
        (axes[1], "pct_counts_ribo", "Ribosomal content", "Ribosomal reads (%)"),
        (axes[2], "pct_counts_hemoglobin", "Hemoglobin content", "Hemoglobin reads (%)"),
    ]
    for ax, metric, title, ylabel in metric_panels:
        _plot_metric(
            ax,
            frame,
            metric,
            thresholds=threshold_map.get(metric),
            title=title,
            ylabel=ylabel,
            normal_label=normal_label,
            disease_label=disease_label,
        )
    axes[2].set_yscale("symlog", linthresh=0.01)
    max_hb = max(
        float(frame["pct_counts_hemoglobin"].max()),
        *(threshold_map.get("pct_counts_hemoglobin") or [0.0]),
        0.01,
    )
    axes[2].set_ylim(0, max_hb * 1.18)
    _plot_doublet_scores(axes[3], frame)
    _plot_metric(
        axes[4],
        frame,
        "n_genes_by_counts",
        thresholds=threshold_map.get("n_genes_by_counts"),
        title="Genes detected",
        ylabel="Detected genes per cell",
        normal_label=normal_label,
        disease_label=disease_label,
    )
    _plot_doublet_umap(axes[5], adata, frame)
    _plot_scree(axes[6], adata)
    for letter, ax in zip("ABCDEFG", axes, strict=False):
        _panel_label(ax, letter)
    n_final = int(
        frame.get("cellquorum_qc_keep", pd.Series(True, index=frame.index))
        .fillna(True)
        .astype(bool)
        .sum()
    )
    fig.suptitle(
        f"Single-cell quality control: {len(frame):,} input cells to {n_final:,} retained cells",
        y=0.995,
        fontsize=9,
        fontweight="bold",
    )
    return _save_panel(fig, out_dir, "supp_figure_qc_visual_qa_sheet", dpi=dpi, formats=formats)


__all__ = ["QCPublicationFigureError", "write_publication_qc_figures"]
