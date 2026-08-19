"""PCA dimensionality-reduction method with scree artifact and auto n_pcs.

PCAMethod is an AnalysisMethod strategy: it computes PCA on the active matrix,
selects the component count (fixed or via the variance-ratio knee), truncates the
embedding, and emits a house-style scree/elbow figure so the choice is auditable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.preprocessing.dimensionality.knee import select_n_pcs

logger = logging.getLogger(__name__)


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
        """PCA operates on the configured normalized layer."""

        # Read the input layer from config (default to cellquorum_normalized).
        input_layer = config.get("input_layer", "cellquorum_normalized")

        # Require the layer to exist and be tagged as lognorm.
        return DataContract(
            required_layers=[input_layer],
            expression_layer=input_layer,
            expected_kind="lognorm",
        )

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
        input_layer = config.get("input_layer", "cellquorum_normalized")
        n_pcs = config.get("n_pcs", "auto")
        max_pcs = int(config.get("max_pcs", 50))
        random_state = int(config.get("random_state", 0))
        use_hvg = bool(config.get("use_highly_variable", False))

        # Compute the full (capped) PCA once so we have the variance-ratio curve.
        # scanpy >=1.10 deprecated use_highly_variable in favor of mask_var:
        # "highly_variable" restricts PCA to the HVG var column; None uses all genes.
        n_comps = int(min(max_pcs, adata.n_obs - 1, adata.n_vars - 1))
        mask_var = "highly_variable" if use_hvg else None

        # Route by normalization method: a scclr-normalized layer carries a
        # per-cell row_center, so PCA runs through scclr's implicit-centered
        # sparse path (no densify). Any other layer uses the standard scanpy PCA.
        row_center_col = f"{input_layer}_row_center"
        if row_center_col in adata.obs.columns:
            self._run_scclr_pca(
                adata,
                input_layer=input_layer,
                row_center_col=row_center_col,
                n_comps=n_comps,
                random_state=random_state,
                context=context,
            )
            compute_used = "scclr"
            gpu_fallback_note = None
        else:
            compute_used, gpu_fallback_note = self._run_scanpy_pca(
                adata,
                input_layer=input_layer,
                n_comps=n_comps,
                mask_var=mask_var,
                random_state=random_state,
                context=context,
            )

        variance_ratio = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=float)

        # Resolve the component count: knee for "auto", else the fixed int.
        if isinstance(n_pcs, str) and n_pcs == "auto":
            chosen = select_n_pcs(variance_ratio, max_pcs=max_pcs)
            mode = "auto"
        else:
            chosen = int(min(int(n_pcs), n_comps))
            mode = "fixed"

        # Cumulative variance captured by the chosen components (for provenance
        # and the under-selection guard).
        cumulative_variance = float(np.sum(variance_ratio[:chosen]))

        # Record the choice for provenance.
        notes = [f"Selected {chosen} PCs ({mode})."]

        # No-silent-decisions guard: `n_pcs=auto` picks the kneedle elbow, which
        # is KNOWN to under-select on scRNA variance curves (steep-then-flat
        # curves put max curvature very low). Log the decision where it acts and
        # warn loudly when the auto elbow lands far below the cap, so a silent
        # under-selection (e.g. 8/50 PCs) is visible rather than buried.
        if mode == "auto":
            logger.info(
                "n_pcs=auto selected %d of %d computed PCs (%.1f%% cumulative variance).",
                chosen,
                n_comps,
                100.0 * cumulative_variance,
            )
            under_select_floor = max(10, max_pcs // 2)
            if chosen < under_select_floor:
                under_msg = (
                    f"n_pcs=auto selected only {chosen} of up to {max_pcs} PCs "
                    f"({100.0 * cumulative_variance:.1f}% cumulative variance). The kneedle "
                    "elbow is known to under-select on scRNA variance curves; set an "
                    "explicit n_pcs (e.g. 50) if downstream steps use the PCA embedding."
                )
                logger.warning(under_msg)
                notes.append(under_msg)

        # Truncate the embedding to the chosen number of components.
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :chosen]

        # Emit the scree artifact into the figures directory.
        figures_dir = Path(getattr(getattr(context, "paths", None), "figures", "."))
        scree_path = figures_dir / "dimensionality_scree.png"
        write_scree_plot(variance_ratio, chosen, scree_path)

        if gpu_fallback_note is not None:
            notes.append(gpu_fallback_note)

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
            metrics={
                "n_pcs": int(chosen),
                "n_pcs_mode": mode,
                "n_comps_computed": int(n_comps),
                "n_pcs_cumulative_variance": cumulative_variance,
                "compute": compute_used,
            },
            notes=notes,
        )

    def _run_scanpy_pca(
        self,
        adata: ad.AnnData,
        *,
        input_layer: str,
        n_comps: int,
        mask_var: str | None,
        random_state: int,
        context: object,
    ) -> tuple[str, str | None]:
        """Run standard scanpy/rapids PCA on a dense/standard lognorm layer.

        Returns (compute_used, gpu_fallback_note).
        """

        # Decide GPU vs CPU once via the shared router.
        from cellquorum.compute.router import resolve_compute

        routing = resolve_compute(context)
        compute_used = "cpu"
        gpu_fallback_note = None

        if routing["use_gpu"]:
            try:
                import rapids_singlecell as rsc

                # Move to GPU, run rapids PCA, move back — same output key X_pca.
                rsc.get.anndata_to_GPU(adata)
                rsc.pp.pca(
                    adata,
                    n_comps=n_comps,
                    mask_var=mask_var,
                    random_state=random_state,
                    layer=input_layer,
                )
                rsc.get.anndata_to_CPU(adata)
                compute_used = "gpu"
            except Exception as exc:  # noqa: BLE001
                # GPU path failed; fall back to CPU when permitted.
                if not routing["fallback_to_cpu"]:
                    raise
                # Ensure any partial GPU state is returned to CPU before retrying.
                try:
                    import rapids_singlecell as rsc

                    rsc.get.anndata_to_CPU(adata)
                except Exception:
                    pass
                gpu_fallback_note = (
                    f"GPU PCA failed ({type(exc).__name__}: {str(exc)[:80]}); fell back to CPU."
                )
                sc.pp.pca(
                    adata,
                    n_comps=n_comps,
                    random_state=random_state,
                    mask_var=mask_var,
                    layer=input_layer,
                )
        else:
            sc.pp.pca(
                adata,
                n_comps=n_comps,
                random_state=random_state,
                mask_var=mask_var,
                layer=input_layer,
            )

        return compute_used, gpu_fallback_note

    def _run_scclr_pca(
        self,
        adata: ad.AnnData,
        *,
        input_layer: str,
        row_center_col: str,
        n_comps: int,
        random_state: int,
        context: object,
    ) -> None:
        """Run scclr's implicit-centered sparse PCA on a scclr-normalized layer.

        The scclr-normalized layer is sparse PFlog values; combined with the
        per-cell ``row_center`` it represents ``layer - row_center[:, None]``
        without densifying. This routes that sparse+center pair through the scclr
        backend's PCA helper and writes ``obsm["X_pca"]`` +
        ``uns["pca"]["variance_ratio"]`` so the shared knee/scree/truncate logic
        runs unchanged.

        Raises:
            CellQuorumStageError: If the scclr backend is unavailable.
        """

        import json
        import tempfile

        import scipy.sparse as sp

        from cellquorum.backends.scclr_backend import PFLOG_HELPER, ScclrBackend
        from cellquorum.core.exceptions import CellQuorumStageError

        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("scclr")
            except Exception:
                backend = None
        if not isinstance(backend, ScclrBackend) or not backend.status().available:
            raise CellQuorumStageError(
                "dimensionality",
                "scclr-normalized layer requires the scclr backend for sparse PCA, "
                "which is unavailable. Build the isolated scclr environment.",
            )

        layer = adata.layers[input_layer]
        sparse = layer.tocsr() if sp.issparse(layer) else sp.csr_matrix(np.asarray(layer))
        row_center = np.asarray(adata.obs[row_center_col].to_numpy(), dtype=float)

        scratch = Path(getattr(getattr(context, "paths", None), "scratch", tempfile.gettempdir()))
        scratch.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            tmp_path = Path(tmp)
            matrix_in = tmp_path / "pflog.npz"
            center_in = tmp_path / "center.npy"
            pca_out = tmp_path / "pca.npz"
            pca_meta = tmp_path / "pca_meta.json"
            sp.save_npz(matrix_in, sparse)
            np.save(center_in, row_center)

            result = backend.run_helper(
                PFLOG_HELPER,
                [
                    "pca",
                    str(matrix_in),
                    str(center_in),
                    str(pca_out),
                    str(pca_meta),
                    "--n-components",
                    str(n_comps),
                    "--seed",
                    str(random_state),
                ],
            )
            if result.returncode != 0 or not pca_out.is_file():
                raise CellQuorumStageError(
                    "dimensionality",
                    "scclr sparse PCA failed: " f"{result.stderr.strip()[:500] or 'no stderr'}",
                )

            with np.load(pca_out) as data:
                scores = np.asarray(data["scores"], dtype=float)
                variance_ratio = np.asarray(data["explained_variance_ratio"], dtype=float)
                explained_variance = np.asarray(data["explained_variance"], dtype=float)
            json.loads(pca_meta.read_text())

        # Write results in the scanpy-shaped keys the shared logic expects.
        adata.obsm["X_pca"] = scores
        adata.uns["pca"] = {
            "variance_ratio": variance_ratio,
            "variance": explained_variance,
        }


__all__ = ["PCAMethod", "write_scree_plot"]
