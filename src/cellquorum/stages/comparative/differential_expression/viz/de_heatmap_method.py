"""Advanced DE heatmap (ComplexHeatmap, R): genes x cluster x condition.

Complements volcano_viz rather than replacing it -- volcano_viz answers "what
changed in this one comparison," this answers "how do the changed genes from
EVERY subcluster compare side by side." Reads the by-subcluster DE CSV
directly (not the pooled one volcano_viz uses), pulls real expression from
the object, and renders through ComplexHeatmap, which has no real Python
equivalent for this: real dendrograms, real multi-row annotation bars, a
render that degrades gracefully from a handful of genes to several thousand.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.sparse as sp

from cellquorum.backends.script_paths import r_script_path
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod

_DE_HEATMAP_R = r_script_path("de_heatmap.R")
_BY_CLUSTER_CSV = "de_pseudobulk_edger_by_subcluster.csv"


def select_top_genes(
    df: pd.DataFrame,
    *,
    group_col: str,
    logfc_col: str,
    fdr_col: str,
    gene_col: str,
    fdr_cut: float,
    top_n: int,
) -> list[str]:
    """Top-``top_n`` up + top-``top_n`` down genes per group among significant rows.

    Deliberately per-group (not global top-N), so a group with a real but
    modest signature isn't crowded out by a group with a huge one -- every
    cluster gets a say in which genes make the heatmap. Deduplicated and
    sorted for a deterministic row order into the matrix build.
    """
    sig = df[df[fdr_col] < fdr_cut]
    genes: set[str] = set()
    for _group, sub in sig.groupby(group_col):
        ordered = sub.sort_values(logfc_col, ascending=False)
        genes.update(ordered.head(top_n)[gene_col].tolist())
        genes.update(ordered.tail(top_n)[gene_col].tolist())
    return sorted(genes)


class DeHeatmapMethod(RAnalysisMethod):
    """Render the cross-cluster DE heatmap via ComplexHeatmap."""

    name = "de_heatmap"
    stage_category = "de_viz"
    r_package = "ComplexHeatmap"

    def input_contract(self, config: dict) -> DataContract:
        layer = config.get("layer", "cellquorum_normalized")
        return DataContract(required_layers=[layer], required_obs=[], expected_kind="normalized")

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        backend, skip = self._resolve_rscript_backend(context, config)
        if skip is not None:
            return skip

        results_dir = Path(context.paths.results)
        de_csv = results_dir / _BY_CLUSTER_CSV
        if not de_csv.exists():
            return self._skip(f"no {_BY_CLUSTER_CSV} in results")
        try:
            df = pd.read_csv(de_csv)
        except Exception as exc:  # noqa: BLE001
            return self._skip("could not read DE CSV", error=str(exc)[:300])

        cell_type_col = config.get("cell_type_col", "cell_type")
        condition_col = config.get("condition_col", "condition")
        layer = config.get("layer", "cellquorum_normalized")
        top_n = int(config.get("top_n", 15))
        fdr_cut = float(config.get("fdr_cut", 0.05))

        genes = select_top_genes(
            df,
            group_col="cell_type",
            logfc_col="logFC",
            fdr_col="FDR",
            gene_col="gene",
            fdr_cut=fdr_cut,
            top_n=top_n,
        )
        genes = [g for g in genes if g in adata.var_names]
        if not genes:
            return self._skip(f"no significant genes at FDR<{fdr_cut}")

        X = adata[:, genes].layers[layer]
        if sp.issparse(X):
            X = X.toarray()
        expr = pd.DataFrame(X, columns=genes, index=adata.obs_names)
        expr["cluster"] = adata.obs[cell_type_col].astype(str).to_numpy()
        expr["condition"] = adata.obs[condition_col].astype(str).to_numpy()
        expr["group"] = expr["cluster"] + "_" + expr["condition"]
        matrix = expr.groupby("group")[genes].mean().T

        scratch_dir = Path(context.paths.scratch)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        matrix_csv = scratch_dir / "de_heatmap_matrix.csv"
        matrix.to_csv(matrix_csv)

        figures_dir = Path(context.paths.figures) / "differential_expression"
        figures_dir.mkdir(parents=True, exist_ok=True)
        out_png = figures_dir / "de_heatmap.png"

        timeout_seconds = int(config.get("timeout_seconds", 600))
        try:
            proc = backend.run_script(
                _DE_HEATMAP_R, [str(matrix_csv), str(out_png)], timeout=timeout_seconds
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            return self._skip("de_heatmap.R execution failed or timed out", error=str(exc)[:500])

        if proc.returncode != 0:
            rc = proc.returncode
            if rc < 0:
                try:
                    sig_name = signal.Signals(-rc).name
                except ValueError:
                    sig_name = f"signal {-rc}"
                reason = f"de_heatmap.R killed by {sig_name} (signal {-rc}) -- likely OOM"
            else:
                reason = "de_heatmap.R failed"
            return self._skip(
                reason, returncode=rc, stderr=str(getattr(proc, "stderr", "")).strip()[:500]
            )

        if not out_png.exists():
            return self._skip("de_heatmap.R reported success but wrote no output")

        artifacts = [
            StageArtifact(
                name="de_heatmap",
                path=out_png,
                kind="figure",
                description=(
                    f"Cross-cluster DE heatmap: {len(genes)} significant genes "
                    f"(top {top_n} up/down per cluster, FDR<{fdr_cut}), z-scored, "
                    "split by cluster and condition."
                ),
            )
        ]
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"de_heatmap rendered {len(genes)} genes across {matrix.shape[1]} groups."],
            warnings=[],
            metrics={"method": self.name, "n_genes": len(genes)},
            backend="rscript",
        )


__all__ = ["DeHeatmapMethod", "select_top_genes"]
