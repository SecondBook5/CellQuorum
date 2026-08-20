"""Publication-grade summary figure for multicellular programs (MCPs).

Produces a 1×2 panel: (left) program × cell-type participation heatmap showing
gene counts or mean loadings per cell type; (right) program score distributions
with donor-support flags annotated.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cellquorum.visualization import figstyle


def plot_mcp_summary(
    programs: pd.DataFrame,
    scores: pd.DataFrame,
    donor_support: pd.DataFrame,
    *,
    cell_type_col_values: list[str],
    out_dir: Path,
    name: str = "mcp_summary",
) -> Path:
    """Plot MCP summary figure with program participation and score distributions.

    Parameters
    ----------
    programs : pd.DataFrame
        DIALOGUE program gene loadings with columns: program, cell_type, gene,
        loading, direction.
    scores : pd.DataFrame
        Per-cell program scores with columns: cell_id, sample, cell_type,
        program, score.
    donor_support : pd.DataFrame
        Donor support per program with columns: program, n_donors,
        donor_fraction, supported.
    cell_type_col_values : list[str]
        Ordered cell-type labels for stable color assignment.
    out_dir : Path
        Output directory for the figure.
    name : str, optional
        Figure filename stem (default: "mcp_summary").

    Returns
    -------
    Path
        Path to the saved .png file.
    """
    figstyle.set_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_heatmap, ax_scores = axes

    # Left panel: program × cell-type participation heatmap (gene counts).
    participation = programs.groupby(["program", "cell_type"]).size().reset_index(name="gene_count")
    heatmap_data = participation.pivot(
        index="program", columns="cell_type", values="gene_count"
    ).fillna(0)

    # Ensure consistent cell-type order.
    heatmap_data = heatmap_data.reindex(
        columns=[ct for ct in cell_type_col_values if ct in heatmap_data.columns],
        fill_value=0,
    )

    sns.heatmap(
        heatmap_data,
        ax=ax_heatmap,
        cmap="YlOrRd",
        cbar_kws={"label": "Gene count"},
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
    )
    ax_heatmap.set_xlabel("Cell type")
    ax_heatmap.set_ylabel("Program")
    ax_heatmap.set_title("Program participation")

    # Right panel: program score distributions with donor-support annotations.
    program_order = sorted(scores["program"].unique())
    program_palette = figstyle.categorical_palette(program_order)

    for i, program in enumerate(program_order):
        program_scores = scores[scores["program"] == program]["score"]
        parts = ax_scores.violinplot(
            [program_scores],
            positions=[i],
            widths=0.7,
            showmeans=True,
            showextrema=True,
        )
        # Color each violin with the categorical palette.
        for pc in parts["bodies"]:
            pc.set_facecolor(program_palette[program])
            pc.set_alpha(0.7)

    # Annotate donor support.
    support_map = donor_support.set_index("program")["supported"].to_dict()
    for i, program in enumerate(program_order):
        supported = support_map.get(program, False)
        marker = "✓" if supported else "✗"
        color = figstyle.NORMAL_COLOR if supported else figstyle.QC_FAIL_COLOR
        ax_scores.text(
            i,
            ax_scores.get_ylim()[1] * 0.95,
            marker,
            ha="center",
            va="top",
            fontsize=10,
            color=color,
            weight="bold",
        )

    ax_scores.set_xticks(range(len(program_order)))
    ax_scores.set_xticklabels(program_order, rotation=45, ha="right")
    ax_scores.set_xlabel("Program")
    ax_scores.set_ylabel("Score")
    ax_scores.set_title("Program score distributions")
    ax_scores.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    # Save figure and return the .png path (save_figure returns list[Path]).
    paths = figstyle.save_figure(fig, out_dir, name)
    png_path = next(p for p in paths if str(p).endswith(".png"))
    return png_path
