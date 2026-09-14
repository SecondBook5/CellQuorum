"""Figure for cellquorum.stats.rare_population_triage's output.

Plots only the "candidate" rows -- real, donor-recurrent, non-artifact
populations -- ranked by condition enrichment, so the most disease-
associated small populations are the ones that stand out.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cellquorum.stages.integration.embeddings.plots import apply_theme
from cellquorum.visualization.figstyle import save_figure

if TYPE_CHECKING:
    import pandas as pd


def plot_condition_enrichment_bar(
    triage: pd.DataFrame,
    *,
    out_path: Path,
) -> Path | None:
    """
    Plot condition enrichment for every "candidate" population, most depleted to most enriched.

    Args:
        triage: Output of :func:`cellquorum.stats.rare_population_triage.triage_populations`
            (must carry ``label``, ``condition_enrichment``, ``category``).
        out_path: Path to write the figure PNG. A ``.pdf`` is written beside it.

    Returns:
        The PNG path written, or ``None`` when no row has ``category == "candidate"``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    candidates = triage[triage["category"] == "candidate"].sort_values("condition_enrichment")
    if candidates.empty:
        return None

    apply_theme()
    out_path = Path(out_path)

    values = candidates["condition_enrichment"].to_numpy()
    limit = float(np.abs(values).max()) or 1.0
    colors = plt.get_cmap("RdBu_r")((values / limit / 2) + 0.5)

    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.32 * len(candidates) + 1.5)))
    ax.barh(range(len(candidates)), values, color=colors, alpha=0.9)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_yticks(range(len(candidates)))
    ax.set_yticklabels(candidates["label"], fontsize=8)
    ax.set_xlabel("Condition enrichment (this label's share vs. cohort baseline)", fontsize=9)
    ax.set_title("Rare-population condition association", fontsize=11, weight="bold")
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)

    paths = save_figure(fig, out_path.parent, out_path.stem)
    return next(p for p in paths if p.suffix == ".png")


__all__ = ["plot_condition_enrichment_bar"]
