"""PCA dimensionality-reduction method with scree artifact and auto n_pcs.

PCAMethod is an AnalysisMethod strategy: it computes PCA on the active matrix,
selects the component count (fixed or via the variance-ratio knee), truncates the
embedding, and emits a house-style scree/elbow figure so the choice is auditable.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.dimensionality.knee import select_n_pcs
from cellquorum.methods.base import AnalysisMethod


def write_scree_plot(variance_ratio: np.ndarray, chosen_n: int, output_path: Path) -> None:
    """
    Render a house-style scree/elbow plot to ``output_path`` (PNG).

    Bars show per-PC variance %; a red line shows cumulative variance % on a
    secondary axis; an 80% dashed reference line and a vertical marker at the
    chosen component count are drawn.

    Args:
        variance_ratio: Per-PC explained-variance ratios (descending).
        chosen_n: Selected component count (for the vertical marker).
        output_path: Destination PNG path.
    """

    # Use a non-interactive backend so this is safe in headless runs.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Show at most the leading 30 PCs for legibility.
    n_show = int(min(30, len(variance_ratio)))
    x = np.arange(1, n_show + 1)
    pct = 100.0 * np.asarray(variance_ratio[:n_show], dtype=float)
    cumulative = np.cumsum(pct)

    # Build the figure: slate bars (per-PC) + red cumulative line on twin axis.
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(x, pct, width=0.66, color="#334155", alpha=0.94)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")

    # Twin axis for cumulative variance.
    ax2 = ax.twinx()
    ax2.plot(x, cumulative, color="#C44E52", linewidth=0.95, marker="o", markersize=3)
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_ylim(0, 100)

    # 80% reference line and the chosen-n marker.
    ax2.axhline(80, linewidth=0.4, linestyle="--", color="#6B7280")
    ax.axvline(chosen_n, linewidth=0.8, linestyle=":", color="#6B7280")
    ax.set_title("PCA scree")

    # Persist and close.
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


class PCAMethod(AnalysisMethod):
    """PCA reduction strategy (fixed or auto component count)."""

    # Registry identity.
    name = "pca"
    stage_category = "dimensionality"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """PCA operates on the active matrix; no structural precondition."""

        # No required layers/obs — an empty contract validates any object.
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """
        Compute PCA, select components, truncate, and emit a scree artifact.

        Args:
            adata: Input AnnData (uses ``.X``).
            config: Resolved dimensionality config sub-block.
            context: Pipeline context (for the figures output directory).

        Returns:
            StageResult with the PCA embedding, scree artifact, and metrics.
        """

        # Resolve settings with defaults matching DimensionalityConfig.
        n_pcs = config.get("n_pcs", "auto")
        max_pcs = int(config.get("max_pcs", 50))
        random_state = int(config.get("random_state", 0))
        use_hvg = bool(config.get("use_highly_variable", False))

        # Compute the full (capped) PCA once so we have the variance-ratio curve.
        # scanpy >=1.10 deprecated use_highly_variable in favor of mask_var:
        # "highly_variable" restricts PCA to the HVG var column; None uses all genes.
        n_comps = int(min(max_pcs, adata.n_obs - 1, adata.n_vars - 1))
        mask_var = "highly_variable" if use_hvg else None
        sc.pp.pca(
            adata,
            n_comps=n_comps,
            random_state=random_state,
            mask_var=mask_var,
        )
        variance_ratio = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=float)

        # Resolve the component count: knee for "auto", else the fixed int.
        if isinstance(n_pcs, str) and n_pcs == "auto":
            chosen = select_n_pcs(variance_ratio, max_pcs=max_pcs)
            mode = "auto"
        else:
            chosen = int(min(int(n_pcs), n_comps))
            mode = "fixed"

        # Truncate the embedding to the chosen number of components.
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :chosen]

        # Emit the scree artifact into the figures directory.
        figures_dir = Path(getattr(getattr(context, "paths", None), "figures", "."))
        scree_path = figures_dir / "dimensionality_scree.png"
        write_scree_plot(variance_ratio, chosen, scree_path)

        # Record the choice for provenance.
        return StageResult(
            adata=adata,
            artifacts=[
                StageArtifact(
                    name="dimensionality_scree",
                    path=scree_path,
                    kind="figure",
                    description="PCA scree/elbow plot with chosen-component marker.",
                )
            ],
            metrics={"n_pcs": int(chosen), "n_pcs_mode": mode, "n_comps_computed": int(n_comps)},
            notes=[f"Selected {chosen} PCs ({mode})."],
        )


__all__ = ["PCAMethod", "write_scree_plot"]
