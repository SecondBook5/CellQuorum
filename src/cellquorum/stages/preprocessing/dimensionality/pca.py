"""PCA dimensionality-reduction method with scree artifact and auto n_pcs.

PCAMethod is an AnalysisMethod strategy: it computes PCA on the active matrix,
selects the component count (fixed or via the variance-ratio knee), truncates the
embedding, and emits a house-style scree/elbow figure so the choice is auditable.

PCA loadings are a cohort-derived quantity, so the basis is fitted on the cells QC
permits to fit and every cell is then projected onto it. See
:func:`project_onto_fitted_basis` for why that is exact rather than an approximation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from kneed import KneeLocator

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.stages.qc.eligibility import fitting_cells

logger = logging.getLogger(__name__)


def project_onto_fitted_basis(
    matrix: np.ndarray | sp.spmatrix,
    loadings: np.ndarray,
    fit_gene_means: np.ndarray,
) -> np.ndarray:
    """Project cells onto a PCA basis fitted on a different set of cells.

    PCA has an exact out-of-sample transform, which is what makes the core-fit /
    project-everyone split honest here: a borderline cell receives a real coordinate in the
    core cells' manifold without having influenced where that manifold is.

    The arithmetic is the textbook ``(X - mean) @ PCs``, rearranged:

        (X - 1·meanᵀ) @ PCs  ==  X @ PCs - (meanᵀ @ PCs)

    Algebraically identical, but the left form densifies a sparse matrix to subtract the
    mean, and the right form does not. That is not a micro-optimisation: on the validation
    cohort the matrix is 201,923 x 36,601, where densifying is tens of gigabytes.

    Two details make this exact rather than approximate:

    * ``scanpy`` writes ``varm["PCs"]`` at full gene length, zero-filled outside
      ``mask_var``, so passing the whole matrix is correct — non-HVG genes contribute
      ``(x - mean) * 0``. Their means are therefore irrelevant too.
    * ``sc.pp.pca`` centres by default, so the mean must be the **fit population's** gene
      means. Using every cell's means would leak the excluded cells straight back into the
      embedding, which is the whole thing being prevented.

    Args:
        matrix: Cells x genes expression for the cells being projected.
        loadings: ``varm["PCs"]``, genes x components.
        fit_gene_means: Per-gene means over the fit population only.

    Returns:
        A dense cells x components embedding.
    """
    projected = matrix @ loadings
    projected = np.asarray(projected.todense() if sp.issparse(projected) else projected)
    return projected - (fit_gene_means @ loadings)


def write_scree_plot(variance_ratio: np.ndarray, chosen_n: int, output_path: Path) -> None:
    """
    Render a house-style scree/elbow plot to ``output_path``, plus a vector twin.

    Bars show per-PC variance %; a red line shows cumulative variance % on a
    secondary axis; an 80% dashed reference line and a vertical marker at the
    chosen component count are drawn.

    Args:
        variance_ratio: Per-PC explained-variance ratios (descending).
        chosen_n: Selected component count (for the vertical marker).
        output_path: Destination PNG path. A ``.pdf`` is written beside it, because
            this used to be a bare ``fig.savefig`` and the scree plot was one of only
            three figures in a run that had no vector form.
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

    # Persist and close. The shared writer supplies the atomic rename and the
    # vector twin; closing stays here, since save_cellquorum_figure does not close.
    from cellquorum.visualization.figstyle import save_cellquorum_figure

    save_cellquorum_figure(fig, output_path, dpi=150)
    plt.close(fig)


def select_n_pcs(variance_ratio: np.ndarray, *, max_pcs: int) -> int:
    """
    Return the elbow component count from a descending variance-ratio curve.

    NOTE: the kneedle elbow of a scRNA-seq variance curve is known to UNDER-select
    (steep-then-flat curves put max-curvature very low). Hypothesis configs
    therefore set an explicit ``n_pcs`` (typically 50, matching field practice)
    rather than relying on ``auto``. A more principled ``auto`` (Marchenko-Pastur
    noise threshold / parallel analysis) is a possible future replacement.

    Args:
        variance_ratio: Per-PC explained-variance ratios, descending.
        max_pcs: Upper bound on the returned count.

    Returns:
        A component count in [1, min(len(variance_ratio), max_pcs)].
    """

    # Bound the search to the available components and the configured cap.
    n_available = int(len(variance_ratio))
    cap = max(1, min(n_available, int(max_pcs)))

    # A knee needs at least three points; below that, use the cap.
    if n_available < 3:
        return cap

    # x is 1-based component index; y is the (capped) variance-ratio curve.
    x = np.arange(1, cap + 1)
    y = np.asarray(variance_ratio[:cap], dtype=float)

    # Locate the elbow of the convex, decreasing curve.
    locator = KneeLocator(x, y, curve="convex", direction="decreasing")
    knee = locator.knee

    # Fall back to the cap when no knee is detected.
    if knee is None:
        return cap

    # Clamp into [1, cap] and return as an int count.
    return int(min(max(1, int(knee)), cap))


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
        # PCA loadings are a cohort-derived quantity used to transform biological data, so
        # the basis is fitted only on the cells QC permits to fit and everyone else is
        # projected onto it. This stage declares fit_scope=CORE at registration; the branch
        # below is what honours it.
        #
        # Component count comes from the FIT population, not the full object: asking for
        # more components than there are fitting cells is what turns a small core into an
        # error deep inside the SVD.
        fitting = fitting_cells(adata.obs)
        n_fit_cells = adata.n_obs if fitting is None else int(fitting.sum())
        n_comps = int(min(max_pcs, n_fit_cells - 1, adata.n_vars - 1))
        mask_var = "highly_variable" if use_hvg else None
        scope_notes: list[str] = []

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
            if fitting is not None:
                # The scclr path centres implicitly per cell and exposes no separable
                # basis, so there is nothing to project onto. Stated rather than silently
                # ignored: a reader must be able to see that this route fitted on
                # everything.
                scope_notes.append(
                    "scclr PCA fitted on all cells: its implicit per-cell centering has no "
                    "out-of-sample transform, so the QC fit population could not be honoured. "
                    "Use a standard lognorm layer for a core-only manifold."
                )
        elif fitting is None:
            compute_used, gpu_fallback_note = self._run_scanpy_pca(
                adata,
                input_layer=input_layer,
                n_comps=n_comps,
                mask_var=mask_var,
                random_state=random_state,
                context=context,
            )
        else:
            compute_used, gpu_fallback_note, scope_note = self._fit_on_core_then_project(
                adata,
                fitting,
                input_layer=input_layer,
                n_comps=n_comps,
                mask_var=mask_var,
                random_state=random_state,
                context=context,
            )
            scope_notes.append(scope_note)

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
        notes.extend(scope_notes)

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

    def _fit_on_core_then_project(
        self,
        adata: ad.AnnData,
        fitting: pd.Series,
        *,
        input_layer: str,
        n_comps: int,
        mask_var: str | None,
        random_state: int,
        context: object,
    ) -> tuple[str, str | None, str]:
        """Fit the PCA basis on the QC fit population, then project every cell onto it.

        The distinction this preserves is the one the eligibility masks exist for: a
        borderline cell is *projected into* the manifold, never *joined to* the fit that
        defined it. It still receives a usable coordinate, so it can be clustered,
        annotated, inspected in a UMAP and considered for rescue — it simply had no say in
        where the axes point.

        Args:
            adata: The full object. Mutated in place with the basis and the embedding.
            fitting: Boolean per-cell mask of the cells permitted to fit.
            input_layer: Expression layer PCA runs on.
            n_comps: Components to compute, already capped by the fit population size.
            mask_var: ``var`` column restricting PCA to a gene subset, or None.
            random_state: Seed.
            context: Pipeline context, for GPU routing.

        Returns:
            ``(compute_used, gpu_fallback_note, scope_note)``.
        """
        fit_adata = adata[fitting].copy()
        compute_used, gpu_fallback_note = self._run_scanpy_pca(
            fit_adata,
            input_layer=input_layer,
            n_comps=n_comps,
            mask_var=mask_var,
            random_state=random_state,
            context=context,
        )

        # No basis means nothing to project onto. Only reachable if a backend stops writing
        # varm["PCs"], and a wrong embedding would be far worse than a slower correct one,
        # so fall back to fitting on everything and say so.
        if "PCs" not in fit_adata.varm:
            compute_used, gpu_fallback_note = self._run_scanpy_pca(
                adata,
                input_layer=input_layer,
                n_comps=n_comps,
                mask_var=mask_var,
                random_state=random_state,
                context=context,
            )
            return (
                compute_used,
                gpu_fallback_note,
                "PCA fitted on all cells: the backend wrote no varm['PCs'], so the core-only "
                "basis could not be projected. The embedding includes non-core cells.",
            )

        # The basis and its variance spectrum describe the fit population and are what
        # later stages read, so they transfer to the full object unchanged.
        loadings = np.asarray(fit_adata.varm["PCs"], dtype=np.float64)
        adata.varm["PCs"] = loadings
        adata.uns["pca"] = dict(fit_adata.uns["pca"])

        # The matrix PCA actually ran on, layer or X, for each object.
        fit_matrix = fit_adata.layers.get(input_layer, fit_adata.X)
        full_matrix = adata.layers.get(input_layer, adata.X)

        # Centering must use the fit population's means. Taking them over every cell would
        # readmit the excluded cells into the embedding through the back door.
        fit_gene_means = np.asarray(fit_matrix.mean(axis=0), dtype=np.float64).ravel()
        adata.obsm["X_pca"] = project_onto_fitted_basis(full_matrix, loadings, fit_gene_means)

        n_projected = int(adata.n_obs - len(fit_adata))
        scope_note = (
            f"PCA basis fitted on {len(fit_adata)} QC-permitted cells; {n_projected} further "
            f"cells projected onto it without influencing it."
        )
        logger.info(scope_note)
        return compute_used, gpu_fallback_note, scope_note

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
        from cellquorum.backends.compute import resolve_compute

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
