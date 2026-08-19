"""CellRank GAM gene-trend figures along pseudotime (opt-in via config.genes)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory.viz import inputs
from cellquorum.trajectory.viz.save import apply_theme, figure_artifacts, save_figure


class GeneTrendVizMethod(AnalysisMethod):
    name = "gene_trend_viz"
    stage_category = "trajectory_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        genes = config.get("genes")
        if not genes:  # no defaulted biology
            return self._skip("no genes configured")
        try:
            import cellrank as cr
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"cellrank unavailable ({exc})")
        fate_path = inputs.results_file(context, "cellrank", "fate_mapping.h5ad")
        if not Path(fate_path).exists():
            return self._skip("no fate_mapping.h5ad")
        try:
            fate = ad.read_h5ad(fate_path)
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"read failed ({exc})")

        present = sorted(g for g in genes if g in set(map(str, fate.var_names)))
        if not present:
            return self._skip("no requested gene in var_names", requested=list(genes))

        import matplotlib.pyplot as plt

        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        apply_theme()
        warnings = []
        try:
            model = cr.models.GAM(fate)
            cr.pl.gene_trends(fate, model=model, genes=present, show=False)
            paths = save_figure(plt.gcf(), figures_dir, "gene_trends", formats=formats, dpi=dpi)
            artifacts = figure_artifacts(
                paths, name="trajectory_figure", description=f"GAM gene trends: {present}."
            )
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"gene_trends failed ({str(exc)[:200]})")
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"gene_trend_viz rendered trends for {present}."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": 1, "genes": present},
            backend="python",
        )


__all__ = ["GeneTrendVizMethod"]
