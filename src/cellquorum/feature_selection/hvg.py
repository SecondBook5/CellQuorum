"""Highly-variable-gene selection method (flavor-dispatched, flag-only).

seurat_v3 / pearson_residuals operate on RAW COUNTS; seurat (v1) operates on the
log-normalized layer. Feeding PFlog1pPF (centered) to a count flavor is a silent
correctness bug, so the input contract asserts the expected layer kind.
"""

from __future__ import annotations

import re

import anndata as ad
import scanpy as sc

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod

# Flavors that require raw counts vs. the log-normalized layer.
_COUNT_FLAVORS = {"seurat_v3", "pearson_residuals"}


class HVGMethod(AnalysisMethod):
    """Scanpy HVG selection; flavor + layer chosen by config.

    Writes var['highly_variable'] but never subsets the object. To consume
    the HVGs, also set dimensionality.use_highly_variable: true.
    """

    name = "seurat"
    stage_category = "feature_selection"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """Count flavors require the counts layer; seurat requires lognorm."""

        method = config.get("method", "seurat")
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

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """Compute HVGs, strip excluded patterns, flag var['highly_variable']."""

        method = config.get("method", "seurat")
        n_top = int(config.get("n_top_genes", 2000))
        batch_key = config.get("batch_key", None)
        exclude = config.get("exclude_gene_patterns", []) or []

        # Choose the layer that matches the flavor's expected input space.
        if method in _COUNT_FLAVORS:
            layer = config.get("counts_layer", "counts")
        else:
            layer = config.get("lognorm_layer", "cellquorum_normalized")

        # scanpy writes var['highly_variable'] (+ ranks/means) in place.
        # pearson_residuals HVG lives ONLY in sc.experimental; all other flavors
        # (seurat, seurat_v3) are in sc.pp.
        if method == "pearson_residuals":
            sc.experimental.pp.highly_variable_genes(
                adata,
                flavor="pearson_residuals",
                n_top_genes=n_top,
                layer=layer,
                batch_key=batch_key,
            )
        else:
            sc.pp.highly_variable_genes(
                adata,
                flavor=method,
                n_top_genes=n_top,
                layer=layer,
                batch_key=batch_key,
            )

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
        return StageResult(
            adata=adata,
            metrics={
                "method": method,
                "n_hvg": n_hvg,
                "n_top_genes": n_top,
                "n_excluded": n_excluded,
                "layer": layer,
            },
            notes=[f"HVG ({method}) flagged {n_hvg} genes (excluded {n_excluded})."],
        )


__all__ = ["HVGMethod"]
