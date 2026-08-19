"""CellRank estimator-native figures (macrostates, coarse transition matrix)."""

from __future__ import annotations

import inspect
from pathlib import Path

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory_viz import inputs
from cellquorum.trajectory_viz.save import apply_theme, figure_artifacts, save_figure


class MacrostateVizMethod(AnalysisMethod):
    name = "macrostate_viz"
    stage_category = "trajectory_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        try:
            import cellrank as cr
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"cellrank unavailable ({exc})")

        pkl = inputs.results_file(context, "cellrank", "gpcca_estimator.pickle")
        if not Path(pkl).exists():
            return self._skip("no gpcca_estimator.pickle")
        try:
            estimator = cr.estimators.GPCCA.read(str(pkl))
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"estimator read failed ({exc})")

        est_adata = getattr(estimator, "adata", None)
        basis = (
            inputs.resolve_basis(est_adata, config.get("embedding_basis"))
            if est_adata is not None
            else None
        )

        import matplotlib.pyplot as plt

        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))

        apply_theme()
        artifacts, warnings, n = [], [], 0

        # plot_macrostates needs an embedding basis.
        if basis is not None:
            try:
                sig = inspect.signature(estimator.plot_macrostates)
                kwargs = {"which": "all", "basis": basis}
                if "show" in sig.parameters:
                    kwargs["show"] = False
                estimator.plot_macrostates(**kwargs)
                paths = save_figure(plt.gcf(), figures_dir, "macrostates", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths,
                    name="trajectory_figure",
                    description="CellRank macrostates on embedding.",
                )
                n += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"macrostate_viz: plot_macrostates failed: {str(exc)[:200]}")
        else:
            warnings.append("macrostate_viz: no embedding basis; plot_macrostates skipped")

        # coarse transition matrix (no show= kwarg).
        try:
            estimator.plot_coarse_T()
            paths = save_figure(
                plt.gcf(), figures_dir, "coarse_transition", formats=formats, dpi=dpi
            )
            artifacts += figure_artifacts(
                paths, name="trajectory_figure", description="CellRank coarse transition matrix."
            )
            n += 1
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"macrostate_viz: plot_coarse_T failed: {str(exc)[:200]}")

        if n == 0:
            return self._skip("nothing rendered", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"macrostate_viz rendered {n} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n},
            backend="python",
        )


__all__ = ["MacrostateVizMethod"]
