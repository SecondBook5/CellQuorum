"""Per-subcluster volcano: one file per cluster, highlighting its own markers.

Complements volcano_viz (one pooled comparison) and de_heatmap (top genes
across all clusters, compressed to a heatmap). This answers a different
question: how much does THIS cluster's transcriptome actually shift between
conditions, and are its own identity markers among the genes that move.
Pooling across clusters can hide real heterogeneity -- one subcluster might
show almost no response while another shows hundreds of significant genes --
which is exactly what per-cluster volcanoes surface and a single pooled plot
cannot.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.clustering.subclustering.diagnostics import select_marker_genes
from cellquorum.stages.comparative.differential_expression.viz import plots
from cellquorum.visualization import figstyle

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

_BY_CLUSTER_CSV = "de_pseudobulk_edger_by_subcluster.csv"


def _safe_slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "group"


class VolcanoBySubclusterMethod(AnalysisMethod):
    """Render one volcano per subcluster, highlighting that cluster's markers."""

    name = "volcano_by_subcluster"
    stage_category = "de_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        layer = config.get("layer", "cellquorum_normalized")
        return DataContract(required_layers=[layer], required_obs=[], expected_kind="normalized")

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        de_csv = results_dir / _BY_CLUSTER_CSV
        if not de_csv.exists():
            return self._skip(f"no {_BY_CLUSTER_CSV} in results")
        try:
            df = pd.read_csv(de_csv)
        except Exception as exc:  # noqa: BLE001
            return self._skip("could not read DE CSV", error=str(exc)[:300])

        cell_type_col = config.get("cell_type_col", "cell_type")
        if cell_type_col not in adata.obs.columns:
            return self._skip(f"obs column '{cell_type_col}' not found on adata")

        csv_groups = set(df["cell_type"].astype(str).unique())
        adata_groups = set(adata.obs[cell_type_col].astype(str).unique())
        if csv_groups.isdisjoint(adata_groups):
            return self._skip(
                f"cell_type values in {_BY_CLUSTER_CSV} don't match any value "
                f"in obs['{cell_type_col}']"
            )

        layer = config.get("layer", "cellquorum_normalized")
        n_marker_genes = int(config.get("n_marker_genes", 8))
        try:
            markers = select_marker_genes(
                adata, cluster_key=cell_type_col, layer=layer, n_top=n_marker_genes
            )
        except Exception as exc:  # noqa: BLE001
            return self._skip("marker computation failed", error=str(exc)[:300])

        case = config.get("case")
        control = config.get("control")
        case_color = config.get("case_color") or figstyle.LE_RED
        control_color = config.get("control_color") or figstyle.NORMAL_BLUE
        cluster_labels = config.get("cluster_labels") or {}
        fc_cut = float(config.get("fc_cut", 1.0))
        fdr_cut = float(config.get("fdr_cut", 0.05))
        top_n_labels = int(config.get("top_n_labels", 20))
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))

        figures_dir = Path(context.paths.figures) / "differential_expression"
        artifacts: list[StageArtifact] = []
        for group in sorted(csv_groups):
            sub = df[df["cell_type"].astype(str) == group]
            if sub.empty:
                continue
            label = cluster_labels.get(group, group)
            x_label = (
                f"log2 fold change ({case} - {control}) -- {label}"
                if case and control
                else f"log2 fold change (case - control) -- {label}"
            )
            try:
                fig = plots.volcano(
                    sub,
                    fc_cut=fc_cut,
                    fdr_cut=fdr_cut,
                    case_color=case_color,
                    control_color=control_color,
                    x_label=x_label,
                    top_n_labels=top_n_labels,
                    highlight_genes=set(markers.get(group, [])),
                )
            except Exception as exc:  # noqa: BLE001
                return self._skip(f"render failed for group '{group}'", error=str(exc)[:300])
            paths = figstyle.save_figure(
                fig,
                figures_dir,
                f"volcano_subcluster_{_safe_slug(str(label))}",
                formats=formats,
                dpi=dpi,
            )
            artifacts.extend(
                StageArtifact(
                    name="de_figure",
                    path=p,
                    kind="figure",
                    description=f"Per-subcluster DE volcano ({label}), markers highlighted.",
                )
                for p in paths
            )

        if not artifacts:
            return self._skip("no subcluster groups produced output")

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"volcano_by_subcluster rendered {len(csv_groups)} subcluster(s)."],
            warnings=[],
            metrics={"method": self.name, "n_groups": len(csv_groups)},
            backend="python",
        )


__all__ = ["VolcanoBySubclusterMethod"]
