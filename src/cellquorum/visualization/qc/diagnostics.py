"""QC diagnostic visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.visualization.figstyle import (
    CELLQUORUM_BLUE,
    CELLQUORUM_FIGSIZE_SMALL,
    CELLQUORUM_GRAY,
    CELLQUORUM_RED,
    SEQUENTIAL_CMAP,
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    get_group_palette,
    save_cellquorum_figure,
)

# Import QCThresholdResult only for type checking to avoid circular import.
if TYPE_CHECKING:
    from cellquorum.stages.qc.thresholds import QCThresholdResult


class QCVisualizationError(CellQuorumDataError):
    """Report QC visualization failures."""


@dataclass(frozen=True)
class QCVisualizationResult:
    """
    QC visualization result.

    Contains paths to generated figures and any warnings encountered during
    visualization.
    """

    figure_paths: list[Path]
    warnings: list[str]


def write_qc_figures(
    adata: AnnData,
    output_dir: Path,
    *,
    dpi: int = 300,
    figure_format: str = "png",
    overwrite: bool = False,
    thresholds: QCThresholdResult | None = None,
    group_key: str | None = None,
) -> QCVisualizationResult:
    """
    Write QC diagnostic figures.

    Creates publication-quality diagnostic plots for QC metrics:
    - Total counts histogram
    - Number of genes histogram
    - Gene-class percentage histograms (mitochondrial, ribosomal, hemoglobin —
      one per ``pct_counts_*`` metric present)
    - Total counts vs number of genes scatter
    - Gene detection histogram
    - Keep/fail barplot (if QC decisions exist)
    - Grouped violin plots per metric (if group_key provided)
    - Colored scatter plots (mito percentage, keep/fail)
    - Doublet score distribution (if doublet scores present)
    - Threshold overlays on histograms and violins (if thresholds provided)

    Args:
        adata: AnnData with QC metrics in .obs.
        output_dir: Output directory for figures.
        dpi: Figure resolution (default: 300 for publication).
        figure_format: Figure format (png, pdf, svg).
        overwrite: Whether to overwrite existing figures.
        thresholds: Optional QC threshold result for drawing threshold lines.
        group_key: Optional obs column for grouping violin plots.

    Returns:
        QCVisualizationResult with figure paths and warnings.

    Raises:
        QCVisualizationError: If visualization fails.
    """
    # Apply CellQuorum theme
    apply_cellquorum_theme()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: list[Path] = []
    warnings: list[str] = []

    # Figure 1: Total counts histogram
    if "total_counts" in adata.obs.columns:
        fig_path = output_dir / f"qc_total_counts_histogram.{figure_format}"
        if overwrite or not fig_path.exists():
            try:
                _plot_total_counts_histogram(
                    adata, fig_path, dpi, bounds=_threshold_bounds(thresholds, "total_counts")
                )
                figure_paths.append(fig_path)
            except Exception as e:
                warnings.append(f"Failed to create total counts histogram: {e}")
        else:
            figure_paths.append(fig_path)

    # Figure 2: Number of genes histogram
    if "n_genes_by_counts" in adata.obs.columns:
        fig_path = output_dir / f"qc_n_genes_by_counts_histogram.{figure_format}"
        if overwrite or not fig_path.exists():
            try:
                _plot_n_genes_histogram(
                    adata, fig_path, dpi, bounds=_threshold_bounds(thresholds, "n_genes_by_counts")
                )
                figure_paths.append(fig_path)
            except Exception as e:
                warnings.append(f"Failed to create n_genes histogram: {e}")
        else:
            figure_paths.append(fig_path)

    # Figure 3: Gene-class percentage histograms (mitochondrial, ribosomal,
    # hemoglobin). Each is plotted only when its pct_counts_* metric is present,
    # so ambient/QC configs that skip a class simply skip its figure.
    for family, family_label in (
        ("mito", "Mitochondrial"),
        ("ribo", "Ribosomal"),
        ("hemoglobin", "Hemoglobin"),
    ):
        column = f"pct_counts_{family}"
        if column in adata.obs.columns:
            fig_path = output_dir / f"qc_{column}_histogram.{figure_format}"
            if overwrite or not fig_path.exists():
                try:
                    _plot_pct_counts_histogram(
                        adata,
                        column,
                        family_label,
                        fig_path,
                        dpi,
                        bounds=_threshold_bounds(thresholds, column),
                    )
                    figure_paths.append(fig_path)
                except Exception as e:
                    warnings.append(
                        f"Failed to create {family_label.lower()} percentage histogram: {e}"
                    )
            else:
                figure_paths.append(fig_path)
        else:
            warnings.append(f"{column} not found, skipping {family_label.lower()} histogram")

    # Figure 4: Total counts vs number of genes scatter
    if "total_counts" in adata.obs.columns and "n_genes_by_counts" in adata.obs.columns:
        fig_path = output_dir / f"qc_total_counts_vs_n_genes.{figure_format}"
        if overwrite or not fig_path.exists():
            try:
                _plot_counts_vs_genes_scatter(adata, fig_path, dpi)
                figure_paths.append(fig_path)
            except Exception as e:
                warnings.append(f"Failed to create counts vs genes scatter: {e}")
        else:
            figure_paths.append(fig_path)

    # Figure 5: Gene detection histogram
    if "n_cells_by_counts" in adata.var.columns:
        fig_path = output_dir / f"qc_gene_detection_histogram.{figure_format}"
        if overwrite or not fig_path.exists():
            try:
                _plot_gene_detection_histogram(adata, fig_path, dpi)
                figure_paths.append(fig_path)
            except Exception as e:
                warnings.append(f"Failed to create gene detection histogram: {e}")
        else:
            figure_paths.append(fig_path)

    # Figure 6: Keep/fail barplot (optional)
    if "cellquorum_qc_keep" in adata.obs.columns:
        fig_path = output_dir / f"qc_keep_fail_barplot.{figure_format}"
        if overwrite or not fig_path.exists():
            try:
                _plot_keep_fail_barplot(adata, fig_path, dpi)
                figure_paths.append(fig_path)
            except Exception as e:
                warnings.append(f"Failed to create keep/fail barplot: {e}")
        else:
            figure_paths.append(fig_path)
    else:
        warnings.append("cellquorum_qc_keep not found, skipping keep/fail barplot")

    # Figure 7+: Grouped violin plots per metric (when group_key provided or when
    # thresholds exist). These plots show the distribution of QC metrics grouped by
    # a metadata column (e.g., sample, batch, condition).
    violin_metrics = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mito",
        "pct_counts_ribo",
        "pct_counts_hemoglobin",
    ]
    for metric in violin_metrics:
        if metric in adata.obs.columns:
            fig_path = output_dir / f"qc_violin_{metric}.{figure_format}"
            if overwrite or not fig_path.exists():
                try:
                    bounds = _threshold_bounds(thresholds, metric)
                    _plot_qc_violin(adata, metric, group_key, bounds, fig_path, dpi)
                    figure_paths.append(fig_path)
                except Exception as e:
                    warnings.append(f"Failed to create violin plot for {metric}: {e}")
            else:
                figure_paths.append(fig_path)

    # Figure 8: Colored scatter plots (counts vs genes colored by mito or keep/fail).
    if "total_counts" in adata.obs.columns and "n_genes_by_counts" in adata.obs.columns:
        # Mito-colored variant
        if "pct_counts_mito" in adata.obs.columns:
            fig_path = output_dir / f"qc_counts_vs_genes_mito.{figure_format}"
            if overwrite or not fig_path.exists():
                try:
                    _plot_counts_vs_genes_colored(
                        adata, "pct_counts_mito", "continuous", fig_path, dpi
                    )
                    figure_paths.append(fig_path)
                except Exception as e:
                    warnings.append(f"Failed to create mito-colored scatter: {e}")
            else:
                figure_paths.append(fig_path)

        # Keep/fail-colored variant
        if "cellquorum_qc_keep" in adata.obs.columns:
            fig_path = output_dir / f"qc_counts_vs_genes_keepfail.{figure_format}"
            if overwrite or not fig_path.exists():
                try:
                    _plot_counts_vs_genes_colored(
                        adata, "cellquorum_qc_keep", "categorical", fig_path, dpi
                    )
                    figure_paths.append(fig_path)
                except Exception as e:
                    warnings.append(f"Failed to create keep/fail-colored scatter: {e}")
            else:
                figure_paths.append(fig_path)

    # Figure 9: Doublet score distribution (when doublet scores present).
    doublet_col = _resolve_doublet_score_column(adata)
    if doublet_col:
        fig_path = output_dir / f"qc_doublet_score_distribution.{figure_format}"
        if overwrite or not fig_path.exists():
            try:
                _plot_doublet_distribution(adata, doublet_col, fig_path, dpi)
                figure_paths.append(fig_path)
            except Exception as e:
                warnings.append(f"Failed to create doublet distribution: {e}")
        else:
            figure_paths.append(fig_path)
    else:
        warnings.append("No doublet_score column found, skipping doublet distribution")

    return QCVisualizationResult(figure_paths=figure_paths, warnings=warnings)


def _threshold_bounds(
    thresholds: QCThresholdResult | None, metric: str
) -> tuple[float | None, float | None]:
    """
    Extract threshold bounds for a metric.

    Searches the threshold result for the given metric and returns its lower and
    upper bounds. Returns (None, None) when thresholds is None or no matching
    threshold is found.

    Args:
        thresholds: Optional QC threshold result.
        metric: Metric name to search for.

    Returns:
        Tuple of (lower_bound, upper_bound). Both can be None.
    """
    # Return no bounds when thresholds is None.
    if thresholds is None:
        return (None, None)

    # Search for a threshold matching this metric (duck-typed to avoid circular import).
    lower, upper = None, None
    for threshold in thresholds.thresholds:
        if threshold.metric == metric:
            # Accumulate lower and upper bounds (there may be multiple thresholds for
            # the same metric, e.g., fixed + MAD).
            if threshold.lower is not None:
                lower = max(lower, threshold.lower) if lower is not None else threshold.lower
            if threshold.upper is not None:
                upper = min(upper, threshold.upper) if upper is not None else threshold.upper

    return (lower, upper)


def _resolve_doublet_score_column(adata: AnnData) -> str | None:
    """
    Resolve the doublet score column name.

    Prefers 'doublet_score', otherwise returns the first column matching
    'doublet_score_*'. Returns None if no doublet score column is found.

    Args:
        adata: AnnData object.

    Returns:
        Doublet score column name, or None if not found.
    """
    # Prefer the canonical doublet_score column.
    if "doublet_score" in adata.obs.columns:
        return "doublet_score"

    # Otherwise, return the first doublet_score_* column.
    for col in adata.obs.columns:
        if col.startswith("doublet_score_"):
            return col

    return None


def _plot_total_counts_histogram(
    adata: AnnData,
    output_path: Path,
    dpi: int,
    *,
    bounds: tuple[float | None, float | None] = (None, None),
) -> None:
    """Plot total counts per cell histogram."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)
    try:
        total_counts = adata.obs["total_counts"].values

        # Use seaborn for histogram
        sns.histplot(total_counts, bins=50, color=CELLQUORUM_BLUE, alpha=0.7, ax=ax)

        ax.set_xlabel("Total Counts per Cell")
        ax.set_ylabel("Number of Cells")
        ax.set_title("QC: Total Counts Distribution")

        # Add median line
        median_val = np.median(total_counts)
        ax.axvline(
            median_val,
            color=CELLQUORUM_RED,
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {median_val:.0f}",
        )

        # Add threshold lines when bounds are provided
        lower, upper = bounds
        if lower is not None:
            ax.axvline(
                lower,
                color=CELLQUORUM_RED,
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=f"cutoff: {lower:.3g}",
            )
        if upper is not None:
            ax.axvline(
                upper,
                color=CELLQUORUM_RED,
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=f"cutoff: {upper:.3g}",
            )

        ax.legend()

        apply_cellquorum_axis_style(ax)
        save_cellquorum_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)


def _plot_n_genes_histogram(
    adata: AnnData,
    output_path: Path,
    dpi: int,
    *,
    bounds: tuple[float | None, float | None] = (None, None),
) -> None:
    """Plot number of genes per cell histogram."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)
    try:
        n_genes = adata.obs["n_genes_by_counts"].values

        # Use seaborn for histogram
        sns.histplot(n_genes, bins=50, color=CELLQUORUM_BLUE, alpha=0.7, ax=ax)

        ax.set_xlabel("Number of Genes per Cell")
        ax.set_ylabel("Number of Cells")
        ax.set_title("QC: Gene Detection Distribution")

        # Add median line
        median_val = np.median(n_genes)
        ax.axvline(
            median_val,
            color=CELLQUORUM_RED,
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {median_val:.0f}",
        )

        # Add threshold lines when bounds are provided
        lower, upper = bounds
        if lower is not None:
            ax.axvline(
                lower,
                color=CELLQUORUM_RED,
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=f"cutoff: {lower:.3g}",
            )
        if upper is not None:
            ax.axvline(
                upper,
                color=CELLQUORUM_RED,
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=f"cutoff: {upper:.3g}",
            )

        ax.legend()

        apply_cellquorum_axis_style(ax)
        save_cellquorum_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)


def _plot_pct_counts_histogram(
    adata: AnnData,
    column: str,
    family_label: str,
    output_path: Path,
    dpi: int,
    *,
    bounds: tuple[float | None, float | None] = (None, None),
) -> None:
    """Plot a gene-class percentage histogram (mito / ribo / hemoglobin).

    Shared by all ``pct_counts_*`` gene-class metrics so a new class is plotted
    automatically once its metric exists. ``family_label`` is the human-readable
    class name (e.g. "Mitochondrial", "Ribosomal", "Hemoglobin").
    """
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)
    try:
        pct_values = adata.obs[column].values

        # Use seaborn for histogram
        sns.histplot(pct_values, bins=50, color=CELLQUORUM_BLUE, alpha=0.7, ax=ax)

        ax.set_xlabel(f"{family_label} Percentage (%)")
        ax.set_ylabel("Number of Cells")
        ax.set_title(f"QC: {family_label} Content Distribution")

        # Add median line
        median_val = np.median(pct_values)
        ax.axvline(
            median_val,
            color=CELLQUORUM_RED,
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {median_val:.1f}%",
        )

        # Add threshold lines when bounds are provided
        lower, upper = bounds
        if lower is not None:
            ax.axvline(
                lower,
                color=CELLQUORUM_RED,
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=f"cutoff: {lower:.3g}",
            )
        if upper is not None:
            ax.axvline(
                upper,
                color=CELLQUORUM_RED,
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=f"cutoff: {upper:.3g}",
            )

        ax.legend()

        apply_cellquorum_axis_style(ax)
        save_cellquorum_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)


def _plot_counts_vs_genes_scatter(adata: AnnData, output_path: Path, dpi: int) -> None:
    """Plot total counts vs number of genes scatter."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)
    try:
        total_counts = adata.obs["total_counts"].values
        n_genes = adata.obs["n_genes_by_counts"].values

        # Sample if too many cells (for performance)
        max_cells = 5000
        if len(total_counts) > max_cells:
            rng = np.random.default_rng(0)
            indices = rng.choice(len(total_counts), max_cells, replace=False)
            total_counts = total_counts[indices]
            n_genes = n_genes[indices]

        # Use seaborn for scatter with transparency
        sns.scatterplot(
            x=total_counts,
            y=n_genes,
            color=CELLQUORUM_BLUE,
            alpha=0.3,
            s=10,
            edgecolor="none",
            ax=ax,
        )

        ax.set_xlabel("Total Counts per Cell")
        ax.set_ylabel("Number of Genes per Cell")
        ax.set_title("QC: Counts vs Genes Detection")

        apply_cellquorum_axis_style(ax)
        save_cellquorum_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)


def _plot_gene_detection_histogram(adata: AnnData, output_path: Path, dpi: int) -> None:
    """Plot gene detection (cells per gene) histogram."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)
    try:
        n_cells = adata.var["n_cells_by_counts"].values

        # Use seaborn for histogram
        sns.histplot(n_cells, bins=50, color=CELLQUORUM_BLUE, alpha=0.7, ax=ax)

        ax.set_xlabel("Number of Cells Detecting Gene")
        ax.set_ylabel("Number of Genes")
        ax.set_title("QC: Gene Detection Across Cells")

        # Add median line
        median_val = np.median(n_cells)
        ax.axvline(
            median_val,
            color=CELLQUORUM_RED,
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {median_val:.0f}",
        )
        ax.legend()

        apply_cellquorum_axis_style(ax)
        save_cellquorum_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)


def _plot_keep_fail_barplot(adata: AnnData, output_path: Path, dpi: int) -> None:
    """Plot keep vs fail cell counts."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)
    try:
        qc_keep = adata.obs["cellquorum_qc_keep"].values
        keep_count = qc_keep.sum()
        fail_count = len(qc_keep) - keep_count

        categories = ["Pass QC", "Fail QC"]
        counts = [keep_count, fail_count]
        colors = [CELLQUORUM_BLUE, CELLQUORUM_RED]

        # Use matplotlib bar plot for simplicity
        bars = ax.bar(categories, counts, color=colors, alpha=0.7)

        # Add count labels on bars
        for bar, count in zip(bars, counts, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_ylabel("Number of Cells")
        ax.set_title("QC: Cell Filtering Summary")

        # Add percentage text
        total = keep_count + fail_count
        pass_pct = 100 * keep_count / total if total > 0 else 0
        ax.text(
            0.95,
            0.95,
            f"Pass Rate: {pass_pct:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": CELLQUORUM_GRAY,
                "linewidth": 0.5,
            },
        )

        apply_cellquorum_axis_style(ax)
        save_cellquorum_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)


def _plot_qc_violin(
    adata: AnnData,
    metric: str,
    group_key: str | None,
    bounds: tuple[float | None, float | None],
    output_path: Path,
    dpi: int,
) -> None:
    """
    Plot a grouped violin for a QC metric.

    Draws a violin plot showing the distribution of a QC metric, optionally grouped
    by a metadata column. Threshold lines are drawn when bounds are provided.

    Args:
        adata: AnnData object.
        metric: Metric column to plot.
        group_key: Optional obs column for grouping.
        bounds: Tuple of (lower_bound, upper_bound) for threshold lines.
        output_path: Output file path.
        dpi: Figure resolution.
    """
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)
    try:
        # Build a DataFrame for seaborn.
        metric_values = adata.obs[metric].values
        if group_key is not None and group_key in adata.obs.columns:
            group_values = adata.obs[group_key].values
            plot_df = pd.DataFrame({metric: metric_values, "group": group_values})
            # Get group palette.
            unique_groups = [str(g) for g in plot_df["group"].unique()]
            if len(unique_groups) > 30:
                # A high-cardinality grouping column can create absurdly large
                # rendered figures. Fall back to a single distribution rather
                # than failing the QC plot set.
                plot_df = pd.DataFrame(
                    {metric: metric_values, "group": ["All"] * len(metric_values)}
                )
                sns.violinplot(
                    data=plot_df,
                    x="group",
                    y=metric,
                    color=CELLQUORUM_BLUE,
                    ax=ax,
                    inner="box",
                )
                ax.set_xlabel("")
            else:
                palette = get_group_palette(unique_groups)
                # Draw grouped violin (use hue to assign palette correctly).
                sns.violinplot(
                    data=plot_df,
                    x="group",
                    y=metric,
                    hue="group",
                    palette=palette,
                    ax=ax,
                    inner="box",
                    legend=False,
                )
                ax.set_xlabel(group_key)

            # Publication touch: annotate a two-group comparison with a Mann-Whitney
            # p-value (matches the lekc qc_by_condition_publication style). Only
            # drawn for exactly two plotted groups; skipped silently if the test
            # can't be computed.
            if len(unique_groups) == 2:
                try:
                    from scipy.stats import mannwhitneyu

                    g0, g1 = unique_groups
                    v0 = plot_df.loc[plot_df["group"].astype(str) == g0, metric].to_numpy()
                    v1 = plot_df.loc[plot_df["group"].astype(str) == g1, metric].to_numpy()
                    v0 = v0[np.isfinite(v0)]
                    v1 = v1[np.isfinite(v1)]
                    if v0.size and v1.size:
                        _, p_val = mannwhitneyu(v0, v1, alternative="two-sided")
                        y_top = float(np.nanmax(metric_values))
                        ax.text(
                            0.5,
                            y_top * 1.02 if y_top > 0 else y_top,
                            f"Mann–Whitney p = {p_val:.2e}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            transform=ax.get_xaxis_transform(),
                        )
                except Exception:
                    # Stats are a nice-to-have annotation; never fail the figure
                    # over them.
                    pass
        else:
            # Single-group violin (no grouping).
            plot_df = pd.DataFrame({metric: metric_values, "group": ["All"] * len(metric_values)})
            sns.violinplot(
                data=plot_df, x="group", y=metric, color=CELLQUORUM_BLUE, ax=ax, inner="box"
            )
            ax.set_xlabel("")

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"QC: {metric.replace('_', ' ').title()} Distribution")

        # Draw threshold lines when bounds are provided.
        lower, upper = bounds
        if lower is not None:
            ax.axhline(lower, color=CELLQUORUM_RED, linestyle="--", linewidth=1.5, alpha=0.7)
        if upper is not None:
            ax.axhline(upper, color=CELLQUORUM_RED, linestyle="--", linewidth=1.5, alpha=0.7)

        apply_cellquorum_axis_style(ax)
        save_cellquorum_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)


def _plot_counts_vs_genes_colored(
    adata: AnnData,
    color_col: str,
    color_type: str,
    output_path: Path,
    dpi: int,
) -> None:
    """
    Plot counts vs genes scatter colored by a metric.

    Draws a scatter plot of total_counts vs n_genes_by_counts, colored by a
    continuous metric (e.g., pct_counts_mito) or a categorical metric (e.g.,
    cellquorum_qc_keep).

    Args:
        adata: AnnData object.
        color_col: Column to use for coloring.
        color_type: Either 'continuous' or 'categorical'.
        output_path: Output file path.
        dpi: Figure resolution.
    """
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)
    try:
        total_counts = adata.obs["total_counts"].values
        n_genes = adata.obs["n_genes_by_counts"].values
        color_values = adata.obs[color_col].values

        # Sample if too many cells (for performance).
        max_cells = 5000
        if len(total_counts) > max_cells:
            rng = np.random.default_rng(0)
            indices = rng.choice(len(total_counts), max_cells, replace=False)
            total_counts = total_counts[indices]
            n_genes = n_genes[indices]
            color_values = color_values[indices]

        # Build a DataFrame for seaborn.
        plot_df = pd.DataFrame(
            {"total_counts": total_counts, "n_genes": n_genes, "color": color_values}
        )

        if color_type == "continuous":
            # Continuous color (e.g., mito percentage).
            scatter = ax.scatter(
                plot_df["total_counts"],
                plot_df["n_genes"],
                c=plot_df["color"],
                cmap=SEQUENTIAL_CMAP,
                alpha=0.5,
                s=10,
                edgecolor="none",
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_col.replace("_", " ").title())
        else:
            # Categorical color (e.g., keep/fail).
            unique_vals = sorted(plot_df["color"].unique())
            palette = {True: CELLQUORUM_BLUE, False: CELLQUORUM_RED}
            for val in unique_vals:
                subset = plot_df[plot_df["color"] == val]
                ax.scatter(
                    subset["total_counts"],
                    subset["n_genes"],
                    c=palette.get(val, CELLQUORUM_GRAY),
                    alpha=0.5,
                    s=10,
                    edgecolor="none",
                    label=f"{'Pass' if val else 'Fail'}",
                )
            ax.legend()

        ax.set_xlabel("Total Counts per Cell")
        ax.set_ylabel("Number of Genes per Cell")
        ax.set_title(f"QC: Counts vs Genes ({color_col.replace('_', ' ').title()})")

        apply_cellquorum_axis_style(ax)
        save_cellquorum_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)


def _plot_doublet_distribution(
    adata: AnnData,
    score_col: str,
    output_path: Path,
    dpi: int,
) -> None:
    """
    Plot doublet score distribution.

    Draws a histogram of doublet scores, optionally split by predicted_doublet
    when that column is present.

    Args:
        adata: AnnData object.
        score_col: Doublet score column name.
        output_path: Output file path.
        dpi: Figure resolution.
    """
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)
    try:
        scores = adata.obs[score_col].values

        # Check if predicted_doublet is present for splitting the distribution.
        if "predicted_doublet" in adata.obs.columns:
            predicted = adata.obs["predicted_doublet"].to_numpy().astype(bool)
            # Draw separate histograms for singlets and doublets.
            singlets = scores[~predicted]
            doublets = scores[predicted]
            ax.hist(
                singlets,
                bins=50,
                color=CELLQUORUM_BLUE,
                alpha=0.7,
                label="Singlet",
                edgecolor="none",
            )
            ax.hist(
                doublets,
                bins=50,
                color=CELLQUORUM_RED,
                alpha=0.7,
                label="Doublet",
                edgecolor="none",
            )
            ax.legend()
        else:
            # Draw a single histogram.
            ax.hist(scores, bins=50, color=CELLQUORUM_BLUE, alpha=0.7, edgecolor="none")

        ax.set_xlabel("Doublet Score")
        ax.set_ylabel("Number of Cells")
        ax.set_title("QC: Doublet Score Distribution")

        # Add median line.
        median_val = np.median(scores)
        ax.axvline(
            median_val,
            color=CELLQUORUM_GRAY,
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {median_val:.3f}",
        )

        apply_cellquorum_axis_style(ax)
        save_cellquorum_figure(fig, output_path, dpi=dpi)
    finally:
        plt.close(fig)


__all__ = [
    "QCVisualizationError",
    "QCVisualizationResult",
    "write_qc_figures",
]
