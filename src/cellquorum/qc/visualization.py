"""QC diagnostic visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.visualization.style import (
    CELLQUORUM_BLUE,
    CELLQUORUM_FIGSIZE_SMALL,
    CELLQUORUM_GRAY,
    CELLQUORUM_RED,
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    save_cellquorum_figure,
)


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
) -> QCVisualizationResult:
    """
    Write QC diagnostic figures.

    Creates publication-quality diagnostic plots for QC metrics:
    - Total counts histogram
    - Number of genes histogram
    - Mitochondrial percentage histogram (if available)
    - Total counts vs number of genes scatter
    - Gene detection histogram
    - Keep/fail barplot (if QC decisions exist)

    Args:
        adata: AnnData with QC metrics in .obs.
        output_dir: Output directory for figures.
        dpi: Figure resolution (default: 300 for publication).
        figure_format: Figure format (png, pdf, svg).
        overwrite: Whether to overwrite existing figures.

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
                _plot_total_counts_histogram(adata, fig_path, dpi)
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
                _plot_n_genes_histogram(adata, fig_path, dpi)
                figure_paths.append(fig_path)
            except Exception as e:
                warnings.append(f"Failed to create n_genes histogram: {e}")
        else:
            figure_paths.append(fig_path)

    # Figure 3: Mitochondrial percentage histogram (optional)
    if "pct_counts_mito" in adata.obs.columns:
        fig_path = output_dir / f"qc_pct_counts_mito_histogram.{figure_format}"
        if overwrite or not fig_path.exists():
            try:
                _plot_mito_histogram(adata, fig_path, dpi)
                figure_paths.append(fig_path)
            except Exception as e:
                warnings.append(f"Failed to create mito percentage histogram: {e}")
        else:
            figure_paths.append(fig_path)
    else:
        warnings.append("pct_counts_mito not found, skipping mitochondrial histogram")

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

    return QCVisualizationResult(figure_paths=figure_paths, warnings=warnings)


def _plot_total_counts_histogram(adata: AnnData, output_path: Path, dpi: int) -> None:
    """Plot total counts per cell histogram."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)

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
    ax.legend()

    apply_cellquorum_axis_style(ax)
    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _plot_n_genes_histogram(adata: AnnData, output_path: Path, dpi: int) -> None:
    """Plot number of genes per cell histogram."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)

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
    ax.legend()

    apply_cellquorum_axis_style(ax)
    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _plot_mito_histogram(adata: AnnData, output_path: Path, dpi: int) -> None:
    """Plot mitochondrial percentage histogram."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)

    pct_mito = adata.obs["pct_counts_mito"].values

    # Use seaborn for histogram
    sns.histplot(pct_mito, bins=50, color=CELLQUORUM_BLUE, alpha=0.7, ax=ax)

    ax.set_xlabel("Mitochondrial Percentage (%)")
    ax.set_ylabel("Number of Cells")
    ax.set_title("QC: Mitochondrial Content Distribution")

    # Add median line
    median_val = np.median(pct_mito)
    ax.axvline(
        median_val,
        color=CELLQUORUM_RED,
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_val:.1f}%",
    )
    ax.legend()

    apply_cellquorum_axis_style(ax)
    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _plot_counts_vs_genes_scatter(adata: AnnData, output_path: Path, dpi: int) -> None:
    """Plot total counts vs number of genes scatter."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)

    total_counts = adata.obs["total_counts"].values
    n_genes = adata.obs["n_genes_by_counts"].values

    # Sample if too many cells (for performance)
    max_cells = 5000
    if len(total_counts) > max_cells:
        indices = np.random.choice(len(total_counts), max_cells, replace=False)
        total_counts = total_counts[indices]
        n_genes = n_genes[indices]

    # Use seaborn for scatter with transparency
    sns.scatterplot(
        x=total_counts, y=n_genes, color=CELLQUORUM_BLUE, alpha=0.3, s=10, edgecolor="none", ax=ax
    )

    ax.set_xlabel("Total Counts per Cell")
    ax.set_ylabel("Number of Genes per Cell")
    ax.set_title("QC: Counts vs Genes Detection")

    apply_cellquorum_axis_style(ax)
    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _plot_gene_detection_histogram(adata: AnnData, output_path: Path, dpi: int) -> None:
    """Plot gene detection (cells per gene) histogram."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)

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
    plt.close(fig)


def _plot_keep_fail_barplot(adata: AnnData, output_path: Path, dpi: int) -> None:
    """Plot keep vs fail cell counts."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)

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
    plt.close(fig)


__all__ = [
    "QCVisualizationError",
    "QCVisualizationResult",
    "write_qc_figures",
]
