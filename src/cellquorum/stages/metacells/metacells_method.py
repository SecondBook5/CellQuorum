"""MetaCell2 partitioning method."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.backends.metacells_backend import RUN_METACELL2_PY
from cellquorum.core.contracts.data_contract import DataContract
from cellquorum.core.h5ad_io import write_h5ad
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.metacells.aggregate import dominant_group, outlier_summary
from cellquorum.stages.metacells.viz.composition import plot_metacell_composition_bar
from cellquorum.stages.metacells.viz.heatmap import plot_metacell_gene_heatmap
from cellquorum.stages.metacells.viz.marker_traces import plot_metacell_marker_traces
from cellquorum.stages.metacells.viz.network import plot_metacell_network
from cellquorum.stages.metacells.viz.qc_scatter import (
    plot_metacell_qc_size_comparison,
    plot_metacell_qc_unclassified_vs_size,
)
from cellquorum.visualization.figstyle import render_figure


def _top_k_positive_edges(corr: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Build an edge list from each node's top-k strongest positive correlations.

    Drops non-positive and NaN correlations (a negative or undefined
    correlation is not a "similarity" edge) and keeps at most `k` edges per
    node, rather than every pair -- an unthresholded full pairwise graph is
    O(n^2) edges, which is both meaningless (includes anti-correlated pairs)
    and unusable to render once metacell counts reach the hundreds to
    thousands. Fully vectorized over the correlation matrix (no Python-level
    loop over nodes or pairs).

    Args:
        corr: Square, symmetric correlation matrix (metacell_id x metacell_id).
        k: Max edges kept per node before undirected deduplication.

    Returns:
        DataFrame with columns source, target, weight -- weight always > 0,
        at most k edges contributed per node before dedup (so at most
        roughly k * n rows total, never n*(n-1)/2).
    """
    if corr.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])
    labels = corr.columns.to_numpy()
    n = len(labels)
    k_eff = min(k, n - 1) if n > 1 else 0
    if k_eff <= 0:
        return pd.DataFrame(columns=["source", "target", "weight"])

    values = corr.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    values = np.where(values > 0, values, np.nan)

    # Top-k column indices per row, descending by value (NaNs sort last).
    sort_key = np.where(np.isnan(values), -np.inf, values)
    order = np.argsort(sort_key, axis=1)[:, ::-1][:, :k_eff]
    row_idx = np.repeat(np.arange(n), k_eff)
    col_idx = order.ravel()
    weights = values[row_idx, col_idx]

    valid = ~np.isnan(weights)
    row_idx, col_idx, weights = row_idx[valid], col_idx[valid], weights[valid]
    if row_idx.size == 0:
        return pd.DataFrame(columns=["source", "target", "weight"])

    edges = pd.DataFrame({"source": labels[row_idx], "target": labels[col_idx], "weight": weights})
    # Dedupe undirected pairs -- a top-k edge often gets contributed by both
    # of its endpoints; keep one row per unordered pair.
    pair = np.sort(np.stack([row_idx, col_idx], axis=1), axis=1)
    edges["_pair"] = list(map(tuple, pair))
    edges = (
        edges.sort_values("weight", ascending=False)
        .drop_duplicates(subset="_pair")
        .drop(columns="_pair")
        .reset_index(drop=True)
    )
    return edges


class MetacellsMethod(AnalysisMethod):
    """Partition cells into near-homogeneous metacells via MetaCell2 in an isolated env."""

    name = "metacells"
    stage_category = "metacells"
    backend = "metacells"

    def input_contract(self, config: dict) -> DataContract:
        layer = config.get("layer", "counts")
        return DataContract(
            required_layers=[layer],
            required_obs=[],
            expression_layer=layer,
            expected_kind="counts",
        )

    def requires_obs(self, config: dict) -> list[str]:
        return []

    def _run(self, adata, config, context) -> StageResult | MethodSkip:  # noqa: ANN001
        # 1. Resolve config
        layer = config.get("layer", "counts")
        rep_key = config.get("rep_key", "X_pca_harmony")
        group_by = config.get("group_by", "ref_cell_type_granular_coarse")
        target_metacell_size = int(config.get("target_metacell_size", 96))
        min_cells = int(config.get("min_cells", 5000))
        seed = int(config.get("seed", 0))
        launcher = config.get("launcher", "micromamba")
        timeout_seconds = int(config.get("timeout_seconds", 3600))

        # 2. Guards -> MethodSkip
        if adata.n_obs < min_cells:
            return self._skip(
                f"too few cells ({adata.n_obs} < {min_cells})", n_obs=int(adata.n_obs)
            )
        if group_by not in adata.obs.columns:
            return self._skip(f"group_by column '{group_by}' absent", group_by=group_by)
        if rep_key not in adata.obsm:
            return self._skip(f"rep_key '{rep_key}' absent from obsm", rep_key=rep_key)
        if shutil.which(launcher) is None:
            return self._skip(f"launcher '{launcher}' not found on PATH", launcher=launcher)
        registry = getattr(context, "backend_registry", None)
        backend = None
        if registry is not None:
            try:
                backend = registry.get("metacells")
            except Exception:
                backend = None
        if backend is None:
            return self._skip("metacells backend unavailable")
        try:
            module_ok = backend._py_module_available("metacells")
        except Exception:
            module_ok = False
        if not module_ok:
            return self._skip("metacells module unavailable in env")

        # 3. Write scratch h5ad (raw counts + rep + group_by only)
        scratch = Path(getattr(context.paths, "scratch", "."))
        scratch.mkdir(parents=True, exist_ok=True)
        h5ad = scratch / "metacells_input.h5ad"
        write_h5ad(
            ad.AnnData(
                X=adata.layers[layer] if layer != "X" else adata.X,
                obs=adata.obs[[group_by]].copy(),
                var=adata.var[[]].copy(),
                obsm={rep_key: adata.obsm[rep_key]},
            ),
            h5ad,
        )

        out_dir = Path(getattr(context.paths, "results", ".")) / "metacells"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 4. Run the in-env script
        args = [
            str(h5ad),
            str(out_dir),
            rep_key,
            "X",  # the scratch h5ad already carries counts as X
            group_by,
            str(target_metacell_size),
            str(min_cells),
            str(seed),
        ]
        try:
            proc = backend.run_script(RUN_METACELL2_PY, args, timeout=timeout_seconds)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            return self._skip(
                "metacells script execution failed or timed out", error=str(exc)[:500]
            )
        if proc.returncode != 0:
            return self._skip("metacells script failed", stderr=proc.stderr.strip()[:500])

        assignments_csv = out_dir / "metacell_assignments.csv"
        skip_marker = out_dir / "metacells_SKIPPED.txt"
        if skip_marker.exists() or not assignments_csv.exists():
            reason = (
                skip_marker.read_text().strip()
                if skip_marker.exists()
                else "no assignments produced"
            )
            return self._skip(reason)
        try:
            # Explicit string dtypes: letting pandas infer types lets numeric-
            # looking barcodes (e.g. "1000", "1001", ...) come back as int64,
            # which then silently fails to match `adata.obs_names` (always
            # str) in the `.map()` join below -- every cell would fall
            # through to "Outliers" with no error and no warning.
            assignments = pd.read_csv(assignments_csv, dtype={"cell_id": str, "metacell_id": str})
        except Exception as exc:
            return self._skip("could not read assignments CSV", error=str(exc)[:500])
        if len(assignments) == 0:
            return self._skip("no cell assignments produced")

        # 5. Write metacell_id onto the main object
        id_by_cell = dict(zip(assignments["cell_id"], assignments["metacell_id"], strict=False))
        matched = set(adata.obs_names) & set(assignments["cell_id"])
        has_real_assignments = (assignments["metacell_id"] != "Outliers").any()
        if has_real_assignments and len(matched) < max(1, int(0.01 * adata.n_obs)):
            # The assignments table has real (non-Outliers) metacells, but
            # almost none of its cell_id values actually match adata's
            # obs_names -- the join below would silently fail and write a
            # plausible-looking obs column that is 100% "Outliers". That is
            # a wrong answer, not a legitimate result, so skip loudly rather
            # than proceed.
            return self._skip(
                "assignments cell_id values do not match adata.obs_names "
                f"({len(matched)}/{adata.n_obs} matched) -- join would silently fail",
                n_matched=len(matched),
                n_obs=int(adata.n_obs),
            )
        adata.obs["metacell_id"] = (
            adata.obs_names.to_series().map(id_by_cell).fillna("Outliers").astype(str).to_numpy()
        )

        mc_adata_path = out_dir / "metacell_adata.h5ad"

        # 5b. Build the small per-metacell "nodes" table every figure shares
        real_assignments = assignments[assignments["metacell_id"] != "Outliers"]
        nodes = pd.DataFrame(
            {
                "metacell_id": sorted(real_assignments["metacell_id"].unique()),
            }
        )
        dom = dominant_group(assignments)
        counts = real_assignments.groupby("metacell_id").size()
        nodes["dominant_group"] = nodes["metacell_id"].map(dom)
        nodes["n_cells"] = nodes["metacell_id"].map(counts).fillna(0).astype(int)

        figures_dir = Path(getattr(context.paths, "figures", ".")) / "metacells"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figs: list[Path] = []
        fig_warnings: list[str] = []

        expr = None
        marker_dir = None
        if mc_adata_path.exists():
            try:
                mc_adata = ad.read_h5ad(mc_adata_path)
                mc_x = mc_adata.X.toarray() if sp.issparse(mc_adata.X) else mc_adata.X
                expr = pd.DataFrame(
                    mc_x.T if hasattr(mc_x, "T") else mc_x,
                    index=mc_adata.var_names,
                    columns=mc_adata.obs_names,
                )
            except Exception as exc:  # noqa: BLE001
                fig_warnings.append(f"could not read metacell_adata for figures: {str(exc)[:150]}")

        # Metacell-to-metacell similarity from shared nearest hub genes -- a
        # simple correlation graph over the pseudobulk expression, thresholded
        # to the top-k strongest positive-correlation edges per node (k=8),
        # since MetaCell2's own umap/graph object is not persisted by the
        # in-env script (only the collected pseudobulk is). An unthresholded
        # full pairwise graph is O(n^2) edges -- unreadable and unusably slow
        # to render once metacell counts reach the hundreds to thousands.
        edges = pd.DataFrame(columns=["source", "target", "weight"])
        if expr is not None and expr.shape[1] > 1:
            edges = _top_k_positive_edges(expr.corr(), k=8)

        render_figure(
            "metacell network",
            lambda: plot_metacell_network(edges, nodes, figures_dir),
            figures=figs,
            warnings=fig_warnings,
        )
        if expr is not None:
            render_figure(
                "metacell gene heatmap",
                lambda: plot_metacell_gene_heatmap(expr, nodes, figures_dir),
                figures=figs,
                warnings=fig_warnings,
            )
        render_figure(
            "metacell composition bar",
            lambda: plot_metacell_composition_bar(assignments, nodes, figures_dir),
            figures=figs,
            warnings=fig_warnings,
        )
        render_figure(
            "metacell qc unclassified vs size",
            lambda: plot_metacell_qc_unclassified_vs_size(assignments, nodes, figures_dir),
            figures=figs,
            warnings=fig_warnings,
        )
        render_figure(
            "metacell qc size comparison",
            lambda: plot_metacell_qc_size_comparison(assignments, nodes, figures_dir),
            figures=figs,
            warnings=fig_warnings,
        )

        marker_genes = config.get("marker_genes") or []
        if marker_genes and expr is not None:
            marker_dir = figures_dir / "metacell_marker_traces"
            render_figure(
                "metacell marker traces",
                lambda: plot_metacell_marker_traces(expr, nodes, marker_dir, marker_genes),
                figures=figs,
                warnings=fig_warnings,
            )

        # 6. Artifacts
        artifacts: list[StageArtifact] = [
            StageArtifact(
                name="metacell_assignments",
                path=assignments_csv,
                kind="csv",
                description="Per-cell metacell assignment (including the Outliers bucket)",
            ),
        ]
        if mc_adata_path.exists():
            artifacts.append(
                StageArtifact(
                    name="metacell_adata",
                    path=mc_adata_path,
                    kind="h5ad",
                    description="Aggregated (pseudobulk) metacell AnnData",
                )
            )
        for fig_path in figs:
            artifacts.append(
                StageArtifact(
                    name=f"figure_{fig_path.stem}",
                    path=fig_path,
                    kind="figure",
                    description="metacells stage figure",
                )
            )

        # 7. Metrics
        summary = outlier_summary(assignments)
        metrics = {
            "n_metacells": summary["n_metacells"],
            "n_outlier_cells": summary["n_outlier_cells"],
            "pct_outlier": summary["pct_outlier"],
            "group_by": group_by,
            "n_obs": int(adata.n_obs),
        }

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            warnings=fig_warnings,
            metrics=metrics,
            backend="metacells",
        )


__all__ = ["MetacellsMethod"]
