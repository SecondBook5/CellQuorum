"""Lineage-driver figures (per-lineage diverging bar + combined heatmap)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory.viz import plots
from cellquorum.trajectory.viz.save import apply_theme, figure_artifacts, save_figure


class DriverVizMethod(AnalysisMethod):
    name = "driver_viz"
    stage_category = "trajectory_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        if "cellrank_lineage_drivers" not in adata.varm:
            return self._skip("no cellrank_lineage_drivers in varm")
        mat = np.asarray(adata.varm["cellrank_lineage_drivers"], dtype="float64")

        # Defensive uns-guard: handle non-dict uns["trajectory"]
        traj = adata.uns.get("trajectory", {})
        cr = traj.get("cellrank", {}) if isinstance(traj, dict) else {}
        fate_names = cr.get("fate_names") if isinstance(cr, dict) else None
        driver_columns = cr.get("driver_columns") if isinstance(cr, dict) else None

        # The cellrank producer stores the FULL compute_lineage_drivers frame in
        # varm: per lineage, five columns <lineage>_corr/_pval/_qval/_ci_low/_ci_high
        # (column names recorded in uns[...]['cellrank']['driver_columns']). Plot
        # ONLY the signed-correlation (_corr) columns; the lineage label is the
        # column name minus that suffix (lineage names may contain underscores, so
        # strip the fixed suffix rather than splitting). When driver_columns is
        # present and identifies _corr columns, select those; otherwise fall back
        # to treating each column as its own lineage (older runs / plain matrices).
        suffix = "_corr"
        corr_cols: list[tuple[int, str]] = []
        if driver_columns and len(driver_columns) == mat.shape[1]:
            corr_cols = [
                (i, str(c)[: -len(suffix)])
                for i, c in enumerate(driver_columns)
                if str(c).endswith(suffix)
            ]
        if corr_cols:
            mat = mat[:, [i for i, _ in corr_cols]]
            names = [nm for _, nm in corr_cols]
        elif fate_names and len(fate_names) == mat.shape[1]:
            names = [str(x) for x in fate_names]
        else:
            names = [f"lineage_{i}" for i in range(mat.shape[1])]

        top_k = int(config.get("top_k", 15))
        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        genes = np.asarray(adata.var_names)

        apply_theme()
        artifacts, warnings, n = [], [], 0
        top_gene_set: list[str] = []
        for col, name in enumerate(names):
            try:
                vals = mat[:, col]
                order = np.argsort(-np.abs(vals), kind="stable")[:top_k]
                sel_genes = [str(genes[i]) for i in order]
                for gname in sel_genes:
                    if gname not in top_gene_set:
                        top_gene_set.append(gname)
                fig = plots.signed_diverging_bar(sel_genes, vals[order], title=f"{name} drivers")
                paths = save_figure(
                    fig, figures_dir, f"drivers_bar_{name}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths, name="trajectory_figure", description=f"Top-{top_k} drivers for {name}."
                )
                n += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"driver_viz: bar {name} failed: {str(exc)[:200]}")

        try:
            if top_gene_set:
                idx = {str(g): i for i, g in enumerate(genes)}
                rows = [mat[idx[g], :] for g in top_gene_set]
                fig = plots.matrix_heatmap(
                    np.vstack(rows),
                    top_gene_set,
                    names,
                    title="Lineage drivers",
                    cbar_label="correlation",
                )
                paths = save_figure(fig, figures_dir, "drivers_heatmap", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="trajectory_figure", description="Driver correlation heatmap."
                )
                n += 1
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"driver_viz: heatmap failed: {str(exc)[:200]}")

        if n == 0:
            return self._skip("nothing rendered", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"driver_viz rendered {n} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n},
            backend="python",
        )


__all__ = ["DriverVizMethod"]
