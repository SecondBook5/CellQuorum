"""Highly-variable-gene selection method (flavor-dispatched, flag-only).

seurat_v3 / pearson_residuals operate on RAW COUNTS; seurat (v1) operates on the
log-normalized layer. Feeding PFlog1pPF (centered) to a count flavor is a silent
correctness bug, so the input contract asserts the expected layer kind.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import scanpy as sc

from cellquorum.core.anndata_access import obs_frame, var_frame
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.stages.preprocessing.feature_selection.visualization import write_hvg_figure
from cellquorum.stages.qc.eligibility import fitting_cells

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

# Flavors that require raw counts vs. the log-normalized layer.
_COUNT_FLAVORS = {"seurat_v3", "pearson_residuals"}

#: Minimum number of expressing cells tried, in order, when the seurat_v3 trend fit is singular.
_LOESS_RETRY_MIN_CELLS = (10, 20, 50)


def _is_loess_failure(error: Exception) -> bool:
    """Whether scanpy's seurat_v3 mean-variance trend fit failed numerically."""
    text = str(error).lower()
    return "condition number" in text or "singularit" in text


def _highly_variable_with_retry(
    fit_adata: ad.AnnData,
    *,
    method: str,
    n_top: int,
    layer: str,
    batch_key: str | None,
) -> int | None:
    """Run scanpy's HVG selection; on a singular seurat_v3 fit, retry on well-expressed genes.

    seurat_v3 fits a loess trend of variance on mean, and with a batch key it does so per batch.
    A batch with few cells leaves many genes expressed in a handful of cells, the design matrix goes
    singular and scanpy raises (a random 4,000-cell subset of four donors did this, while the full
    data did not). The first attempt is unchanged, so a run that works keeps its exact result. Only
    after a failure is the fit repeated on genes expressed in at least ``_LOESS_RETRY_MIN_CELLS``
    cells, smallest first; genes below the bar are not eligible and are flagged false.

    Returns:
        None when the first attempt worked, otherwise the minimum cell count that made it work.

    Raises:
        ValueError: The original error, when the method is not seurat_v3, the failure is not a
            singular fit, or no cell-count bar made it work.
    """
    try:
        sc.pp.highly_variable_genes(
            fit_adata, flavor=method, n_top_genes=n_top, layer=layer, batch_key=batch_key
        )
    except ValueError as error:
        if method != "seurat_v3" or not _is_loess_failure(error):
            raise
        original = error
    else:
        return None

    counts = fit_adata.layers[layer]
    n_expressing = np.asarray((counts > 0).sum(axis=0)).ravel()
    for min_cells in _LOESS_RETRY_MIN_CELLS:
        eligible = n_expressing >= min_cells
        if int(eligible.sum()) <= n_top:
            break
        subset = fit_adata[:, eligible].copy()
        try:
            sc.pp.highly_variable_genes(
                subset, flavor=method, n_top_genes=n_top, layer=layer, batch_key=batch_key
            )
        except ValueError as error:
            if not _is_loess_failure(error):
                raise
            continue
        for column in subset.var.columns:
            fit_adata.var[column] = subset.var[column].reindex(fit_adata.var_names)
        fit_adata.var["highly_variable"] = (
            fit_adata.var["highly_variable"].fillna(False).astype(bool)
        )
        return min_cells
    raise original


class HVGMethod(AnalysisMethod):
    """Scanpy HVG selection; flavor + layer chosen by config.

    Writes var['highly_variable'] but never subsets the object. PCA and scVI
    consume the flag automatically once it exists.
    """

    name = "seurat"
    stage_category = "feature_selection"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """Count flavors require the counts layer; seurat requires lognorm."""

        # Fallback matches FeatureSelectionConfig.method; see the comment in `_run`.
        method = config.get("method", "seurat_v3")
        if method in _COUNT_FLAVORS:
            layer = config.get("counts_layer", "counts")
            return DataContract(
                required_layers=[layer],
                expression_layer=layer,
                expected_kind="counts",
            )
        layer = config.get("lognorm_layer", "cellquorum_normalized")
        return DataContract(
            required_layers=[layer],
            expression_layer=layer,
            expected_kind="lognorm",
        )

    def _run(self, adata: ad.AnnData, config: dict, context: PipelineContext) -> StageResult:
        """Compute HVGs, strip excluded patterns, flag var['highly_variable']."""

        # Fallbacks match FeatureSelectionConfig; that model is the source of truth, these
        # only cover a bare dict passed straight to `_run` (e.g. in tests).
        method = config.get("method", "seurat_v3")
        n_top = int(config.get("n_top_genes", 3000))
        batch_key = config.get("batch_key", None)
        exclude = config.get("exclude_gene_patterns", []) or []

        # Choose the layer that matches the flavor's expected input space.
        if method in _COUNT_FLAVORS:
            layer = config.get("counts_layer", "counts")
        else:
            layer = config.get("lognorm_layer", "cellquorum_normalized")

        # Fit on the cells QC permits, apply to every gene.
        #
        # HVG means, variances and dispersions are cohort statistics, so a damaged cell
        # carrying stress, mitochondrial and immediate-early genes pulls the gene set toward
        # its own signal. That happens HERE, upstream of PCA, so excluding those cells from
        # PCA later does not undo it — the manifold is already defined on the wrong genes.
        # This stage declares fit_scope=CORE at registration; honouring it is this block.
        #
        # Fitting on a cell subset and applying to all cells is exactly the fit/transform
        # split, and it is safe because the result lands in `var`, which is gene-level.
        fitting = fitting_cells(obs_frame(adata))
        fit_adata = adata if fitting is None else adata[fitting].copy()

        retried_at: int | None = None

        # scanpy writes var['highly_variable'] (+ ranks/means) in place.
        # pearson_residuals HVG lives ONLY in sc.experimental; all other flavors
        # (seurat, seurat_v3) are in sc.pp.
        if method == "pearson_residuals":
            sc.experimental.pp.highly_variable_genes(
                fit_adata,
                flavor="pearson_residuals",
                n_top_genes=n_top,
                layer=layer,
                batch_key=batch_key,
            )
        else:
            retried_at = _highly_variable_with_retry(
                fit_adata, method=method, n_top=n_top, layer=layer, batch_key=batch_key
            )

        # Copy the gene-level result back onto the full object when HVG was fitted on a
        # subset. var is indexed by gene, so this is a straight transfer.
        if fit_adata is not adata:
            for column in fit_adata.var.columns:
                if column not in adata.var.columns or column.startswith("highly_variable"):
                    adata.var[column] = fit_adata.var[column].reindex(adata.var_names)

        # Exclude unwanted gene families from the HVG set (do not drop them).
        n_excluded = 0
        if exclude:
            patterns = [re.compile(p) for p in exclude]
            hv = adata.var["highly_variable"].to_numpy().copy()
            for i, gene in enumerate(adata.var_names):
                if hv[i] and any(p.search(str(gene)) for p in patterns):
                    hv[i] = False
                    n_excluded += 1
            adata.var["highly_variable"] = hv

        n_hvg = int(adata.var["highly_variable"].sum())

        artifacts = []
        figures_dir = getattr(getattr(context, "paths", None), "figures", None)
        if figures_dir is not None and config.get("write_figures", True):
            figure_path = write_hvg_figure(
                var_frame(adata), Path(figures_dir) / "feature_selection_hvg.png", method=method
            )
            if figure_path is not None:
                artifacts.append(
                    StageArtifact(
                        name="feature_selection_hvg",
                        path=figure_path,
                        kind="figure",
                        description="Mean-vs-dispersion HVG selection diagnostic.",
                    )
                )

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            metrics={
                "method": method,
                "n_hvg": n_hvg,
                "n_top_genes": n_top,
                "n_excluded": n_excluded,
                "layer": layer,
                "retry_min_cells": retried_at,
            },
            notes=[f"HVG ({method}) flagged {n_hvg} genes (excluded {n_excluded})."],
            warnings=(
                [
                    f"HVG ({method}): the trend fit was singular on the full gene set, so it was "
                    f"repeated on genes expressed in at least {retried_at} cells; genes expressed "
                    "in fewer cells are not eligible. This happens when a batch has few cells."
                ]
                if retried_at is not None
                else []
            ),
        )


__all__ = ["HVGMethod"]
