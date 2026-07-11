"""Preprocessing diagnostic visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
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


class PreprocessingVisualizationError(CellQuorumDataError):
    """Report preprocessing visualization failures."""


@dataclass(frozen=True)
class PreprocessingVisualizationResult:
    """
    Preprocessing visualization result.

    Contains paths to generated figures and any warnings encountered during
    visualization.
    """

    figure_paths: list[Path]
    warnings: list[str]


def write_normalization_figures(
    adata: AnnData,
    output_dir: Path,
    *,
    counts_layer: str = "counts",
    normalized_layer: str = "cellquorum_normalized",
    dpi: int = 300,
    figure_format: str = "png",
    overwrite: bool = False,
) -> PreprocessingVisualizationResult:
    """
    Write normalization diagnostic figures.

    Creates publication-quality diagnostic plots for normalization:
    - Total counts before/after histogram
    - Expression distribution before/after histogram
    - Depth correlation before/after scatter
    - Gene mean-variance relationship

    Handles sparse and dense matrices efficiently without forcing full densification.

    Args:
        adata: AnnData with counts and normalized layers.
        output_dir: Output directory for figures.
        counts_layer: Layer name for raw counts.
        normalized_layer: Layer name for normalized expression.
        dpi: Figure resolution (default: 300 for publication).
        figure_format: Figure format (png, pdf, svg).
        overwrite: Whether to overwrite existing figures.

    Returns:
        PreprocessingVisualizationResult with figure paths and warnings.

    Raises:
        PreprocessingVisualizationError: If visualization fails.
    """
    # Apply CellQuorum theme
    apply_cellquorum_theme()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: list[Path] = []
    warnings: list[str] = []

    # Check that required layers exist
    if counts_layer not in adata.layers:
        warnings.append(f"Counts layer '{counts_layer}' not found, skipping normalization figures")
        return PreprocessingVisualizationResult(figure_paths=figure_paths, warnings=warnings)

    if normalized_layer not in adata.layers:
        warnings.append(
            f"Normalized layer '{normalized_layer}' not found, skipping normalization figures"
        )
        return PreprocessingVisualizationResult(figure_paths=figure_paths, warnings=warnings)

    # Figure 1: Total counts before/after
    fig_path = output_dir / f"normalization_total_counts_before_after.{figure_format}"
    if overwrite or not fig_path.exists():
        try:
            _plot_total_counts_before_after(adata, counts_layer, normalized_layer, fig_path, dpi)
            figure_paths.append(fig_path)
        except Exception as e:
            warnings.append(f"Failed to create total counts before/after plot: {e}")
    else:
        figure_paths.append(fig_path)

    # Figure 2: Expression distribution before/after
    fig_path = output_dir / f"normalization_expression_distribution_before_after.{figure_format}"
    if overwrite or not fig_path.exists():
        try:
            _plot_expression_distribution_before_after(
                adata, counts_layer, normalized_layer, fig_path, dpi
            )
            figure_paths.append(fig_path)
        except Exception as e:
            warnings.append(f"Failed to create expression distribution plot: {e}")
    else:
        figure_paths.append(fig_path)

    # Figure 3: Depth correlation before/after
    fig_path = output_dir / f"normalization_depth_correlation_before_after.{figure_format}"
    if overwrite or not fig_path.exists():
        try:
            _plot_depth_correlation_before_after(
                adata, counts_layer, normalized_layer, fig_path, dpi
            )
            figure_paths.append(fig_path)
        except Exception as e:
            warnings.append(f"Failed to create depth correlation plot: {e}")
    else:
        figure_paths.append(fig_path)

    # Figure 4: Gene mean-variance
    fig_path = output_dir / f"normalization_gene_mean_variance.{figure_format}"
    if overwrite or not fig_path.exists():
        try:
            _plot_gene_mean_variance(adata, normalized_layer, fig_path, dpi)
            figure_paths.append(fig_path)
        except Exception as e:
            warnings.append(f"Failed to create gene mean-variance plot: {e}")
    else:
        figure_paths.append(fig_path)

    return PreprocessingVisualizationResult(figure_paths=figure_paths, warnings=warnings)


def _plot_total_counts_before_after(
    adata: AnnData, counts_layer: str, normalized_layer: str, output_path: Path, dpi: int
) -> None:
    """Plot total counts per cell before and after normalization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Before: counts
    counts_matrix = adata.layers[counts_layer]
    if sp.issparse(counts_matrix):
        counts_total = np.asarray(counts_matrix.sum(axis=1)).flatten()
    else:
        counts_total = counts_matrix.sum(axis=1)

    sns.histplot(counts_total, bins=50, color=CELLQUORUM_GRAY, alpha=0.7, ax=ax1)
    ax1.set_xlabel("Total Counts per Cell")
    ax1.set_ylabel("Number of Cells")
    ax1.set_title("Before Normalization")
    median_before = np.median(counts_total)
    ax1.axvline(
        median_before,
        color=CELLQUORUM_RED,
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_before:.0f}",
    )
    ax1.legend()
    apply_cellquorum_axis_style(ax1)

    # After: normalized
    norm_matrix = adata.layers[normalized_layer]
    if sp.issparse(norm_matrix):
        norm_total = np.asarray(norm_matrix.sum(axis=1)).flatten()
    else:
        norm_total = norm_matrix.sum(axis=1)

    sns.histplot(norm_total, bins=50, color=CELLQUORUM_BLUE, alpha=0.7, ax=ax2)
    ax2.set_xlabel("Total Expression per Cell")
    ax2.set_ylabel("Number of Cells")
    ax2.set_title("After Normalization")
    median_after = np.median(norm_total)
    ax2.axvline(
        median_after,
        color=CELLQUORUM_RED,
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_after:.1f}",
    )
    ax2.legend()
    apply_cellquorum_axis_style(ax2)

    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _plot_expression_distribution_before_after(
    adata: AnnData, counts_layer: str, normalized_layer: str, output_path: Path, dpi: int
) -> None:
    """Plot expression value distribution before and after normalization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Sample values for histogram (avoid plotting millions of points)
    max_values = 100000

    # Before: counts (nonzero only for sparse)
    counts_matrix = adata.layers[counts_layer]
    if sp.issparse(counts_matrix):
        # Sample from nonzero values
        counts_data = counts_matrix.data
        if len(counts_data) > max_values:
            indices = np.random.choice(len(counts_data), max_values, replace=False)
            counts_sample = counts_data[indices]
        else:
            counts_sample = counts_data
    else:
        counts_flat = counts_matrix.flatten()
        if len(counts_flat) > max_values:
            indices = np.random.choice(len(counts_flat), max_values, replace=False)
            counts_sample = counts_flat[indices]
        else:
            counts_sample = counts_flat

    # Plot with log scale if appropriate
    if counts_sample.max() > 100:
        sns.histplot(
            counts_sample, bins=50, color=CELLQUORUM_GRAY, alpha=0.7, ax=ax1, log_scale=True
        )
        ax1.set_xlabel("Count Value (log scale)")
    else:
        sns.histplot(counts_sample, bins=50, color=CELLQUORUM_GRAY, alpha=0.7, ax=ax1)
        ax1.set_xlabel("Count Value")

    ax1.set_ylabel("Frequency")
    ax1.set_title("Raw Counts Distribution")
    apply_cellquorum_axis_style(ax1)

    # After: normalized
    norm_matrix = adata.layers[normalized_layer]
    if sp.issparse(norm_matrix):
        # Sample from nonzero values
        norm_data = norm_matrix.data
        if len(norm_data) > max_values:
            indices = np.random.choice(len(norm_data), max_values, replace=False)
            norm_sample = norm_data[indices]
        else:
            norm_sample = norm_data
    else:
        norm_flat = norm_matrix.flatten()
        if len(norm_flat) > max_values:
            indices = np.random.choice(len(norm_flat), max_values, replace=False)
            norm_sample = norm_flat[indices]
        else:
            norm_sample = norm_flat

    sns.histplot(norm_sample, bins=50, color=CELLQUORUM_BLUE, alpha=0.7, ax=ax2)
    ax2.set_xlabel("Normalized Expression")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Normalized Expression Distribution")
    apply_cellquorum_axis_style(ax2)

    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _plot_depth_correlation_before_after(
    adata: AnnData, counts_layer: str, normalized_layer: str, output_path: Path, dpi: int
) -> None:
    """Plot gene expression vs depth correlation before and after normalization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Sample cells if too many
    max_cells = 5000
    n_cells = adata.n_obs
    if n_cells > max_cells:
        cell_indices = np.random.choice(n_cells, max_cells, replace=False)
    else:
        cell_indices = np.arange(n_cells)

    # Calculate cell depths
    counts_matrix = adata.layers[counts_layer]
    if sp.issparse(counts_matrix):
        counts_depth = np.asarray(counts_matrix.sum(axis=1)).flatten()
    else:
        counts_depth = counts_matrix.sum(axis=1)

    # Sample cell depths
    counts_depth_sample = counts_depth[cell_indices]

    # Before: mean expression vs depth
    if sp.issparse(counts_matrix):
        counts_mean = np.asarray(counts_matrix[cell_indices, :].mean(axis=1)).flatten()
    else:
        counts_mean = counts_matrix[cell_indices, :].mean(axis=1)

    sns.scatterplot(
        x=counts_depth_sample,
        y=counts_mean,
        color=CELLQUORUM_GRAY,
        alpha=0.3,
        s=10,
        edgecolor="none",
        ax=ax1,
    )
    ax1.set_xlabel("Total Counts (Depth)")
    ax1.set_ylabel("Mean Expression")
    ax1.set_title("Before: Depth vs Expression")
    apply_cellquorum_axis_style(ax1)

    # After: mean expression vs original depth
    norm_matrix = adata.layers[normalized_layer]
    if sp.issparse(norm_matrix):
        norm_mean = np.asarray(norm_matrix[cell_indices, :].mean(axis=1)).flatten()
    else:
        norm_mean = norm_matrix[cell_indices, :].mean(axis=1)

    sns.scatterplot(
        x=counts_depth_sample,
        y=norm_mean,
        color=CELLQUORUM_BLUE,
        alpha=0.3,
        s=10,
        edgecolor="none",
        ax=ax2,
    )
    ax2.set_xlabel("Original Total Counts (Depth)")
    ax2.set_ylabel("Mean Normalized Expression")
    ax2.set_title("After: Independence from Depth")
    apply_cellquorum_axis_style(ax2)

    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _plot_gene_mean_variance(
    adata: AnnData, normalized_layer: str, output_path: Path, dpi: int
) -> None:
    """Plot gene mean-variance relationship in normalized data."""
    fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SMALL)

    # Calculate gene means and variances
    norm_matrix = adata.layers[normalized_layer]
    if sp.issparse(norm_matrix):
        gene_means = np.asarray(norm_matrix.mean(axis=0)).flatten()
        gene_vars = np.asarray(norm_matrix.power(2).mean(axis=0)).flatten() - gene_means**2
    else:
        gene_means = norm_matrix.mean(axis=0)
        gene_vars = norm_matrix.var(axis=0)

    # Remove zero-variance genes for log scale
    nonzero_mask = (gene_means > 0) & (gene_vars > 0)
    gene_means_nz = gene_means[nonzero_mask]
    gene_vars_nz = gene_vars[nonzero_mask]

    # Plot on log-log scale
    sns.scatterplot(
        x=gene_means_nz,
        y=gene_vars_nz,
        color=CELLQUORUM_BLUE,
        alpha=0.3,
        s=10,
        edgecolor="none",
        ax=ax,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean Normalized Expression (log)")
    ax.set_ylabel("Variance (log)")
    ax.set_title("Gene Mean-Variance Relationship")

    # Add reference line (variance = mean for Poisson)
    xlim = ax.get_xlim()
    x_ref = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 100)
    ax.plot(x_ref, x_ref, color=CELLQUORUM_GRAY, linestyle="--", linewidth=1, label="Var = Mean")
    ax.legend()

    apply_cellquorum_axis_style(ax)
    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


__all__ = [
    "PreprocessingVisualizationError",
    "PreprocessingVisualizationResult",
    "write_normalization_figures",
]
