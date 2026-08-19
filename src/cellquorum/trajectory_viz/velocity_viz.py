"""scVelo velocity-stream figures per velocity group."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory_viz import inputs
from cellquorum.trajectory_viz.save import apply_theme, figure_artifacts, save_figure


class VelocityVizMethod(AnalysisMethod):
    name = "velocity_viz"
    stage_category = "trajectory_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        try:
            import scvelo as scv
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"scvelo unavailable ({exc})")
        vel_dir = inputs.results_file(context, "velocity")
        files = sorted(Path(vel_dir).glob("*.h5ad")) if Path(vel_dir).exists() else []
        if not files:
            return self._skip("no velocity h5ads")

        import matplotlib.pyplot as plt

        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        apply_theme()
        artifacts, warnings, n = [], [], 0
        for f in files:
            try:
                sub = ad.read_h5ad(f)
                scv.pl.velocity_embedding_stream(sub, show=False)
                paths = save_figure(
                    plt.gcf(), figures_dir, f"velocity_stream_{f.stem}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths, name="trajectory_figure", description=f"Velocity stream for {f.stem}."
                )
                n += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"velocity_viz: {f.stem} failed: {str(exc)[:200]}")

        if n == 0:
            return self._skip("nothing rendered", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"velocity_viz rendered {n} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n},
            backend="python",
        )


__all__ = ["VelocityVizMethod"]
