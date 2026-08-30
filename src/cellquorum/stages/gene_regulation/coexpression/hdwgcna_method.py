"""hdWGCNA co-expression network analysis method."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.backends.hdwgcna_backend import HDWGCNA_R
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.stages.gene_regulation.coexpression.module_umap_plot import (
    plot_module_umap,
)
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class HdwgcnaMethod(AnalysisMethod):
    """Co-expression network analysis via hdWGCNA in an isolated R environment.

    hdWGCNA (hierarchical Weighted Gene Co-expression Network Analysis) identifies
    gene modules from scRNA-seq data. This method runs hdWGCNA in an isolated
    micromamba environment, produces module assignments, eigengenes, and a
    publication-grade module UMAP visualization.
    """

    name = "hdwgcna"
    stage_category = "coexpression"
    backend = "hdwgcna_r"

    def input_contract(self, config: dict) -> DataContract:
        """Require the counts layer for co-expression analysis.

        ``group_by`` is deliberately NOT a hard obs requirement: when the
        configured column is absent the R script falls back to a single "all"
        group (see hdwgcna.R), so requiring it here would hard-fail the
        contract instead of allowing that graceful fallback.
        """
        layer = config.get("layer", "counts")

        return DataContract(
            required_layers=[layer],
            required_obs=[],
            expression_layer=layer,
            expected_kind="counts",
        )

    def requires_obs(self, config: dict) -> list[str]:
        """Return obs columns required for co-expression analysis.

        Returns empty list to avoid over-constraining since group_by can fall back
        to obs-present columns or R "all" fallback.
        """
        # Don't hard-require columns that may be absent - the R script handles fallbacks
        return []

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        """Run hdWGCNA co-expression analysis."""

        # 1. Resolve config fields
        layer = config.get("layer", "counts")
        group_by_cfg = config.get("group_by")
        condition_col = config.get("condition_col")
        n_hvg = int(config.get("n_hvg", 3000))
        k = int(config.get("k", 25))
        min_cells = int(config.get("min_cells", 50))
        min_cells_total = int(config.get("min_cells_total", 100))
        soft_power = config.get("soft_power")
        seed = int(config.get("seed", 0))
        launcher = config.get("launcher", "micromamba")
        timeout_seconds = int(config.get("timeout_seconds", 3600))

        # 2. Resolve group_by: config OR first-present of (cell_type, leiden) OR "cell_type"
        if group_by_cfg:
            group_by = group_by_cfg
        elif "cell_type" in adata.obs.columns:
            group_by = "cell_type"
        elif "leiden" in adata.obs.columns:
            group_by = "leiden"
        else:
            group_by = "cell_type"

        # 3. Eligibility guards (each → MethodSkip)
        # a. Check minimum cell count
        if adata.n_obs < min_cells_total:
            return self._skip(
                f"too few cells ({adata.n_obs} < {min_cells_total})",
                n_obs=int(adata.n_obs),
                min_cells_total=min_cells_total,
            )

        # b. Check launcher availability
        if shutil.which(launcher) is None:
            return self._skip(f"launcher '{launcher}' not found on PATH", launcher=launcher)

        # c. Resolve backend from context.backend_registry
        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("hdwgcna_r")
            except Exception:
                backend = None
        if backend is None:
            return self._skip("hdwgcna_r backend unavailable")

        # d. Check hdWGCNA R package availability
        if not backend._r_package_available("hdWGCNA"):
            return self._skip("hdWGCNA R package unavailable")

        # 4. Write adata to scratch h5ad
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        h5ad = scratch / "coexpr_input.h5ad"

        # If layer is set and != "X" and present in layers, write a shallow copy with X = layer
        if layer and layer != "X" and layer in adata.layers:
            a2 = adata.copy()
            a2.X = a2.layers[layer]
            a2.write_h5ad(h5ad)
        else:
            adata.write_h5ad(h5ad)

        # 5. Create output directory
        out_dir = Path(getattr(context.paths, "results", ".")) / "coexpression"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 6. Build args for R script
        args = [
            str(h5ad),
            str(out_dir),
            group_by,
            condition_col or "condition",
            str(n_hvg),
            str(k),
            str(min_cells),
            "NA" if soft_power is None else str(soft_power),
            str(seed),
        ]

        # 7. Run the hdWGCNA R script
        try:
            proc = backend.run_script(HDWGCNA_R, args, timeout=timeout_seconds)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            return self._skip("R script execution failed or timed out", error=str(exc)[:500])

        if proc.returncode != 0:
            return self._skip("hdWGCNA script failed", stderr=proc.stderr.strip()[:500])

        # 8. Check sentinel files and module results
        modules_csv = out_dir / "modules.csv"
        if not modules_csv.exists():
            return self._skip("modules.csv not found")

        skip_file = out_dir / "hdwgcna_SKIPPED.txt"
        if skip_file.exists():
            skip_text = skip_file.read_text().strip()
            return self._skip(skip_text, skip_file_content=skip_text)

        # Read modules and check for results
        try:
            modules_df = pd.read_csv(modules_csv)
        except Exception as exc:
            return self._skip("could not read modules.csv", error=str(exc)[:500])

        if len(modules_df) == 0:
            return self._skip("no modules detected")

        # Guard against missing 'module' column
        if "module" not in modules_df.columns:
            return self._skip("modules.csv missing 'module' column")

        # 9. Render module UMAP figure if data available
        figs = []
        notes = []
        module_umap_csv = out_dir / "module_umap.csv"
        if module_umap_csv.exists():
            try:
                figs = plot_module_umap(module_umap_csv, out_dir)
            except Exception as exc:
                notes.append(f"Module UMAP figure rendering failed: {str(exc)[:200]}")
                figs = []

        # 10. Build artifacts list (only for files that exist)
        artifacts = []

        # Always add modules.csv
        artifacts.append(
            StageArtifact(
                name="modules",
                path=modules_csv,
                kind="csv",
                description="Gene-to-module assignments from hdWGCNA",
            )
        )

        # Optional artifacts
        eigengenes_csv = out_dir / "eigengenes.csv"
        if eigengenes_csv.exists():
            artifacts.append(
                StageArtifact(
                    name="eigengenes",
                    path=eigengenes_csv,
                    kind="csv",
                    description="Module eigengene expression values",
                )
            )

        if module_umap_csv.exists():
            artifacts.append(
                StageArtifact(
                    name="module_umap_data",
                    path=module_umap_csv,
                    kind="csv",
                    description="Module UMAP coordinates and hub gene annotations",
                )
            )

        module_condition_corr_csv = out_dir / "module_condition_corr.csv"
        if module_condition_corr_csv.exists():
            artifacts.append(
                StageArtifact(
                    name="module_condition_corr",
                    path=module_condition_corr_csv,
                    kind="csv",
                    description="Module-condition correlation results",
                )
            )

        # Add figure artifacts
        for fig_path in figs:
            artifacts.append(
                StageArtifact(
                    name=f"module_umap_figure_{fig_path.suffix[1:]}",
                    path=fig_path,
                    kind="figure",
                    description="Module UMAP visualization",
                )
            )

        # 11. Build metrics
        # Exclude the unassigned "grey" bin (WGCNA/hdWGCNA label for genes not
        # placed in any real module) so counts reflect true co-expression modules.
        real_modules = modules_df[modules_df["module"].astype(str).str.lower() != "grey"]
        n_modules = real_modules["module"].nunique()
        n_genes_assigned = len(real_modules)

        metrics = {
            "n_modules": int(n_modules),
            "n_genes_assigned": int(n_genes_assigned),
            "group_by": group_by,
            "soft_power": soft_power,
            "n_obs": int(adata.n_obs),
        }

        # 12. Return StageResult
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics=metrics,
            backend="hdwgcna_r",
        )


__all__ = ["HdwgcnaMethod"]
