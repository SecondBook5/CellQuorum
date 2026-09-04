"""Diagnostics plots for reference mapping (loss curves + uncertainty)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from cellquorum.visualization.figstyle import save_cellquorum_figure

matplotlib.use("Agg")


def plot_loss_curves(loss_history: dict[str, dict[str, list[float]]], out_path: Path) -> None:
    """
    Plot scVI/scANVI/query loss curves (elbo, reconstruction, train_loss).

    Args:
        loss_history: {phase: {metric: values}} from model.history serialization.
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.
    """
    phases = ["scvi", "scanvi", "query_surgery"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for i, phase in enumerate(phases):
        ax = axes[i]
        if phase not in loss_history:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(phase)
            continue

        phase_data = loss_history[phase]
        for metric, values in phase_data.items():
            if values:
                ax.plot(values, label=metric, linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.set_title(phase, fontsize=10, weight="bold")
        ax.tick_params(labelsize=8)
        if phase_data:
            ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3, linewidth=0.5)

    # The shared writer supplies the atomic rename and the vector twin; these two
    # were bare savefig calls, so the diagnostics shipped as PNG only.
    save_cellquorum_figure(fig, out_path, dpi=150)
    plt.close(fig)


def plot_uncertainty(obs: pd.DataFrame, key_added: str, out_path: Path) -> None:
    """
    Plot kNN entropy, agreement, and consensus_frac histograms.

    Args:
        obs: Query obs DataFrame carrying uncertainty columns.
        key_added: Base name for uncertainty columns.
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    knn_entropy_col = f"{key_added}_knn_entropy"
    knn_agreement_col = f"{key_added}_knn_agreement"
    consensus_frac_col = f"{key_added}_consensus_frac"

    if knn_entropy_col in obs.columns:
        axes[0].hist(obs[knn_entropy_col], bins=30, color="#4A90E2", alpha=0.7)
        axes[0].set_xlabel("kNN entropy", fontsize=9)
        axes[0].set_ylabel("Cells", fontsize=9)
        axes[0].set_title("Label uncertainty", fontsize=10, weight="bold")
        axes[0].tick_params(labelsize=8)
        axes[0].grid(alpha=0.3, linewidth=0.5)
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes)

    if knn_agreement_col in obs.columns:
        axes[1].hist(obs[knn_agreement_col], bins=30, color="#50C878", alpha=0.7)
        axes[1].set_xlabel("kNN agreement", fontsize=9)
        axes[1].set_ylabel("Cells", fontsize=9)
        axes[1].set_title("Label agreement", fontsize=10, weight="bold")
        axes[1].tick_params(labelsize=8)
        axes[1].grid(alpha=0.3, linewidth=0.5)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1].transAxes)

    if consensus_frac_col in obs.columns:
        axes[2].hist(obs[consensus_frac_col], bins=20, color="#FF6B6B", alpha=0.7)
        axes[2].set_xlabel("Consensus fraction", fontsize=9)
        axes[2].set_ylabel("Cells", fontsize=9)
        axes[2].set_title("Multi-seed consensus", fontsize=10, weight="bold")
        axes[2].tick_params(labelsize=8)
        axes[2].grid(alpha=0.3, linewidth=0.5)
    else:
        axes[2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[2].transAxes)

    # The shared writer supplies the atomic rename and the vector twin; these two
    # were bare savefig calls, so the diagnostics shipped as PNG only.
    save_cellquorum_figure(fig, out_path, dpi=150)
    plt.close(fig)


__all__ = ["plot_loss_curves", "plot_uncertainty"]
