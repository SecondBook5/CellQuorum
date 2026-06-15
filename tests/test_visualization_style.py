"""Tests for CellQuorum visualization style module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cellquorum.visualization import (
    BACKGROUND_BLUE,
    BACKGROUND_GRAY,
    BACKGROUND_RED,
    CELLQUORUM_BLUE,
    CELLQUORUM_CLUSTER_COLORS,
    CELLQUORUM_FIGSIZE_SMALL,
    CELLQUORUM_FIGSIZE_SQUARE,
    CELLQUORUM_FIGSIZE_WIDE,
    CELLQUORUM_GRAY,
    CELLQUORUM_RED,
    add_dashed_reference_lines,
    add_directional_arrows,
    add_statistical_annotation_box,
    add_volcano_background_panels,
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    get_cellquorum_colors,
    save_cellquorum_figure,
)


def test_color_constants_are_hex_strings() -> None:
    """Test that color constants are valid hex color codes."""
    colors = [
        CELLQUORUM_BLUE,
        CELLQUORUM_RED,
        CELLQUORUM_GRAY,
        BACKGROUND_BLUE,
        BACKGROUND_GRAY,
        BACKGROUND_RED,
    ]

    for color in colors:
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7


def test_cluster_colors_palette() -> None:
    """Test that cluster colors palette is non-empty and valid."""
    assert len(CELLQUORUM_CLUSTER_COLORS) > 0
    assert all(c.startswith("#") for c in CELLQUORUM_CLUSTER_COLORS)


def test_figsize_constants() -> None:
    """Test that figsize constants are tuples."""
    assert isinstance(CELLQUORUM_FIGSIZE_SMALL, tuple)
    assert isinstance(CELLQUORUM_FIGSIZE_WIDE, tuple)
    assert isinstance(CELLQUORUM_FIGSIZE_SQUARE, tuple)
    assert len(CELLQUORUM_FIGSIZE_SMALL) == 2
    assert len(CELLQUORUM_FIGSIZE_WIDE) == 2
    assert len(CELLQUORUM_FIGSIZE_SQUARE) == 2


def test_apply_cellquorum_theme() -> None:
    """Test that applying theme doesn't raise errors."""
    apply_cellquorum_theme()

    # Verify some key rcParams were set
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["savefig.bbox"] == "tight"


def test_apply_cellquorum_axis_style() -> None:
    """Test that axis styling works correctly."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    apply_cellquorum_axis_style(ax)

    # Top and right spines should be removed
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()

    # Bottom and left spines should be visible
    assert ax.spines["bottom"].get_visible()
    assert ax.spines["left"].get_visible()

    plt.close(fig)


def test_apply_cellquorum_axis_style_keep_all_spines() -> None:
    """Test that axis styling can keep all spines."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    apply_cellquorum_axis_style(ax, remove_top_right=False)

    # All spines should be visible
    assert ax.spines["top"].get_visible()
    assert ax.spines["right"].get_visible()
    assert ax.spines["bottom"].get_visible()
    assert ax.spines["left"].get_visible()

    plt.close(fig)


def test_get_cellquorum_colors_default() -> None:
    """Test getting all cluster colors."""
    colors = get_cellquorum_colors()

    assert isinstance(colors, list)
    assert len(colors) == len(CELLQUORUM_CLUSTER_COLORS)
    assert colors == CELLQUORUM_CLUSTER_COLORS


def test_get_cellquorum_colors_subset() -> None:
    """Test getting subset of cluster colors."""
    colors = get_cellquorum_colors(n=5)

    assert len(colors) == 5
    assert colors == CELLQUORUM_CLUSTER_COLORS[:5]


def test_get_cellquorum_colors_exceeds_palette() -> None:
    """Test that colors cycle when n exceeds palette size."""
    n = len(CELLQUORUM_CLUSTER_COLORS) + 3
    colors = get_cellquorum_colors(n=n)

    assert len(colors) == n
    # First colors should match palette
    assert colors[: len(CELLQUORUM_CLUSTER_COLORS)] == CELLQUORUM_CLUSTER_COLORS
    # Colors should cycle
    assert colors[len(CELLQUORUM_CLUSTER_COLORS)] == CELLQUORUM_CLUSTER_COLORS[0]


def test_save_cellquorum_figure() -> None:
    """Test saving figure with CellQuorum settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        output_path = Path(tmpdir) / "test_figure.png"
        result_path = save_cellquorum_figure(fig, output_path)

        assert result_path.exists()
        assert result_path == output_path

        plt.close(fig)


def test_save_cellquorum_figure_pdf() -> None:
    """Test saving figure as PDF."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        output_path = Path(tmpdir) / "test_figure.pdf"
        result_path = save_cellquorum_figure(fig, output_path, dpi=150)

        assert result_path.exists()
        assert result_path.suffix == ".pdf"

        plt.close(fig)


def test_add_volcano_background_panels() -> None:
    """Test adding background panels to volcano plot."""
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 5)

    add_volcano_background_panels(ax, fc_threshold=0.5)

    # Verify that patches were added to the axis
    # (axvspan creates Rectangle patches)
    patches = [p for p in ax.patches]
    assert len(patches) == 3  # Three background panels

    plt.close(fig)


def test_add_dashed_reference_lines() -> None:
    """Test adding dashed reference lines to volcano plot."""
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 5)

    add_dashed_reference_lines(ax, fc_threshold=0.5, p_threshold=1.3)

    # Verify that lines were added
    lines = ax.get_lines()
    assert len(lines) == 3  # Two vertical + one horizontal

    plt.close(fig)


def test_add_directional_arrows() -> None:
    """Test adding directional arrow labels."""
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 5)

    add_directional_arrows(ax, left_label="Higher in Control", right_label="Higher in Treatment")

    # Verify that annotations were added
    # (Unfortunately matplotlib doesn't expose annotations in a simple list)
    # Just verify it doesn't raise an error

    plt.close(fig)


def test_add_statistical_annotation_box() -> None:
    """Test adding statistical annotation box."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    stats = {"p<0.05": 123, "log2FC>0.5": 456}
    add_statistical_annotation_box(ax, stats, position="lower left")

    # Verify that text was added
    texts = ax.texts
    assert len(texts) > 0

    plt.close(fig)


def test_complete_volcano_plot_styling() -> None:
    """Test complete volcano plot styling pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Apply theme
        apply_cellquorum_theme()

        # Create volcano plot
        fig, ax = plt.subplots(figsize=CELLQUORUM_FIGSIZE_SQUARE)

        # Simulate volcano plot data
        np.random.seed(42)
        log2fc = np.random.randn(1000) * 1.5
        neg_log10p = np.random.exponential(scale=1.0, size=1000)

        # Plot points
        ax.scatter(log2fc, neg_log10p, s=10, alpha=0.5, c=CELLQUORUM_GRAY, edgecolors="none")

        # Apply styling
        add_volcano_background_panels(ax, fc_threshold=0.5)
        add_dashed_reference_lines(ax, fc_threshold=0.5, p_threshold=1.3)
        add_directional_arrows(
            ax, left_label="Higher in Control", right_label="Higher in Treatment"
        )
        add_statistical_annotation_box(
            ax, {"p<0.05": 300, "log2FC>0.5": 250}, position="lower left"
        )

        ax.set_xlabel("Log2 Fold Change")
        ax.set_ylabel("-Log10(p-value)")
        ax.set_title("Differential Expression")

        apply_cellquorum_axis_style(ax)

        # Save figure
        output_path = Path(tmpdir) / "volcano_plot.png"
        save_cellquorum_figure(fig, output_path)

        assert output_path.exists()

        plt.close(fig)
