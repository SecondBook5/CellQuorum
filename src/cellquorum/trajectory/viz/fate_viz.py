"""Fate-probability figures (per-lineage embedding + by-cluster violin)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory.viz import inputs, plots
from cellquorum.trajectory.viz.save import apply_theme, figure_artifacts, save_figure

_FATE_KEYS = ("cellrank_fate_probabilities", "palantir_fate_probabilities")


class FateVizMethod(AnalysisMethod):
    name = "fate_viz"
    stage_category = "trajectory_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _lineage_names(
        self, adata: ad.AnnData, n_lin: int, wanted: list[str] | None
    ) -> list[tuple[int, str]]:
        names = None
        traj = adata.uns.get("trajectory", {})
        cr = traj.get("cellrank", {}) if isinstance(traj, dict) else {}
        fate_names = cr.get("fate_names") if isinstance(cr, dict) else None
        if fate_names and len(fate_names) == n_lin:
            names = [str(x) for x in fate_names]
        else:
            names = [f"lineage_{i}" for i in range(n_lin)]
        if wanted is not None:
            keep = set(wanted)
            return [(i, nm) for i, nm in enumerate(names) if nm in keep]
        return list(enumerate(names))

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        fate_key = next((k for k in _FATE_KEYS if k in adata.obsm), None)
        if fate_key is None:
            return self._skip("no *_fate_probabilities in obsm")
        basis = inputs.resolve_basis(adata, config.get("embedding_basis"))
        if basis is None:
            return self._skip("no embedding basis")

        fp = np.asarray(adata.obsm[fate_key], dtype="float64")
        lineages = self._lineage_names(adata, fp.shape[1], config.get("lineages"))
        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        coords = adata.obsm[basis]
        cluster_key = config.get("cluster_key")

        apply_theme()
        artifacts, warnings, n = [], [], 0
        for col, name in lineages:
            try:
                fig = plots.embedding_scatter(
                    coords, fp[:, col], title=name, cbar_label="fate prob"
                )
                paths = save_figure(fig, figures_dir, f"fate_{name}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="trajectory_figure", description=f"Fate probability for {name}."
                )
                n += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"fate_viz: lineage {name} failed: {str(exc)[:200]}")

        if cluster_key and cluster_key in adata.obs:
            try:
                labels = adata.obs[cluster_key].astype(str).to_numpy()
                for col, name in lineages:
                    groups = {g: fp[labels == g, col] for g in sorted(set(labels))}
                    fig = plots.grouped_violin(
                        groups, title=f"{name} by {cluster_key}", ylabel="fate prob"
                    )
                    paths = save_figure(
                        fig, figures_dir, f"fate_violin_{name}", formats=formats, dpi=dpi
                    )
                    artifacts += figure_artifacts(
                        paths,
                        name="trajectory_figure",
                        description=f"{name} fate by {cluster_key}.",
                    )
                    n += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"fate_viz: violin failed: {str(exc)[:200]}")

        if n == 0:
            return self._skip("nothing rendered", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"fate_viz rendered {n} figures from {fate_key}."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n, "source": fate_key},
            backend="python",
        )


__all__ = ["FateVizMethod"]
