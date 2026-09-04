"""CellRank kernel / fate / driver / velocity plot methods.

Combines:
- VelocityVizMethod (scVelo velocity-stream figures)
- MacrostateVizMethod (CellRank macrostates and coarse transition matrix)
- FateVizMethod (fate-probability figures: embedding + violin)
- DriverVizMethod (lineage-driver figures: bar + heatmap)
"""

from __future__ import annotations

import inspect
from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.trajectory.viz import _helpers

_FATE_KEYS = ("cellrank_fate_probabilities", "palantir_fate_probabilities")


# ══════════════════════════════════════════════════════════════════════════════
# VelocityVizMethod (from velocity_viz.py)
# ══════════════════════════════════════════════════════════════════════════════


class VelocityVizMethod(AnalysisMethod):
    name = "velocity_viz"
    stage_category = "trajectory_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        """Render ONE velocity figure for the whole object, plus diagnostics.

        Rewritten twice over. It used to call
        ``scv.pl.velocity_embedding_stream(sub, show=False)`` — no group colouring,
        no labels, no legend, no axes, no title — once per per-cluster h5ad, so it
        emitted a dozen grey unlabelled fragments instead of one readable figure.
        Now it renders the whole object once, coloured by a resolved group key,
        through cellquorum.visualization.velocity.
        """
        from cellquorum.visualization import velocity as velocity_viz

        vel_dir = _helpers.results_file(context, "velocity")
        whole = Path(vel_dir) / "whole_object.h5ad"
        if not whole.exists():
            return self._skip("no whole_object.h5ad velocity object")

        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        velocity_viz.apply_theme()

        try:
            sub = ad.read_h5ad(whole)
        except Exception as exc:  # noqa: BLE001 — skip-not-crash
            return self._skip(f"could not read velocity object: {exc}")

        # Colour by the most specific NAMED grouping available. A velocity figure
        # with no grouping is the grey-blob failure this rewrite exists to fix, so
        # having no usable key is a skip, not a silently ugly figure.
        group_key = velocity_viz.resolve_group_key(
            sub,
            config.get("group_key"),
            ("cell_type_granular", "cell_type", "state", "leiden"),
        )
        if group_key is None:
            return self._skip("no categorical obs column with 2+ levels to colour velocity by")

        basis = str(config.get("basis", "umap"))
        artifacts, warnings, n = [], [], 0
        try:
            fig = velocity_viz.velocity_stream_figure(
                sub,
                group_key=group_key,
                basis=basis,
                title=config.get("title") or f"RNA velocity ({sub.n_obs:,} cells)",
            )
            paths = _helpers.save_figure(
                fig, figures_dir, "velocity_stream", formats=formats, dpi=dpi
            )
            artifacts += _helpers.figure_artifacts(
                paths,
                name="trajectory_figure",
                description=f"RNA velocity stream over '{group_key}'.",
            )
            n += 1
        except velocity_viz.VelocityRenderError as exc:
            warnings.append(f"velocity_viz: stream skipped: {exc}")
        except Exception as exc:  # noqa: BLE001 — skip-not-crash
            warnings.append(f"velocity_viz: stream failed: {str(exc)[:200]}")

        try:
            diagnostics = velocity_viz.velocity_diagnostics_figure(
                sub,
                group_key=group_key,
                basis=basis,
                title="velocity diagnostics (read before trusting arrow directions)",
            )
            if diagnostics is not None:
                paths = _helpers.save_figure(
                    diagnostics,
                    figures_dir,
                    "velocity_diagnostics",
                    formats=formats,
                    dpi=dpi,
                )
                artifacts += _helpers.figure_artifacts(
                    paths,
                    name="trajectory_figure",
                    description="Velocity speed and coherence, per cell and per group.",
                )
                n += 1
        except Exception as exc:  # noqa: BLE001 — skip-not-crash
            warnings.append(f"velocity_viz: diagnostics failed: {str(exc)[:200]}")

        if n == 0:
            return self._skip("nothing rendered", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"velocity_viz rendered {n} figures (grouped by {group_key})."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n},
            backend="python",
        )


# ══════════════════════════════════════════════════════════════════════════════
# MacrostateVizMethod (from macrostate_viz.py)
# ══════════════════════════════════════════════════════════════════════════════


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

        pkl = _helpers.results_file(context, "cellrank", "gpcca_estimator.pickle")
        if not Path(pkl).exists():
            return self._skip("no gpcca_estimator.pickle")
        try:
            estimator = cr.estimators.GPCCA.read(str(pkl))
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"estimator read failed ({exc})")

        est_adata = getattr(estimator, "adata", None)
        basis = (
            _helpers.resolve_basis(est_adata, config.get("embedding_basis"))
            if est_adata is not None
            else None
        )

        import matplotlib.pyplot as plt

        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))

        _helpers.apply_theme()
        artifacts, warnings, n = [], [], 0

        # plot_macrostates needs an embedding basis.
        if basis is not None:
            try:
                sig = inspect.signature(estimator.plot_macrostates)
                kwargs = {"which": "all", "basis": basis}
                if "show" in sig.parameters:
                    kwargs["show"] = False
                estimator.plot_macrostates(**kwargs)
                paths = _helpers.save_figure(
                    plt.gcf(), figures_dir, "macrostates", formats=formats, dpi=dpi
                )
                artifacts += _helpers.figure_artifacts(
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
            paths = _helpers.save_figure(
                plt.gcf(), figures_dir, "coarse_transition", formats=formats, dpi=dpi
            )
            artifacts += _helpers.figure_artifacts(
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


# ══════════════════════════════════════════════════════════════════════════════
# FateVizMethod (from fate_viz.py)
# ══════════════════════════════════════════════════════════════════════════════


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
        basis = _helpers.resolve_basis(adata, config.get("embedding_basis"))
        if basis is None:
            return self._skip("no embedding basis")

        fp = np.asarray(adata.obsm[fate_key], dtype="float64")
        lineages = self._lineage_names(adata, fp.shape[1], config.get("lineages"))
        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        coords = adata.obsm[basis]
        cluster_key = config.get("cluster_key")

        _helpers.apply_theme()
        artifacts, warnings, n = [], [], 0
        for col, name in lineages:
            try:
                fig = _helpers.embedding_scatter(
                    coords, fp[:, col], title=name, cbar_label="fate prob"
                )
                paths = _helpers.save_figure(
                    fig, figures_dir, f"fate_{name}", formats=formats, dpi=dpi
                )
                artifacts += _helpers.figure_artifacts(
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
                    fig = _helpers.grouped_violin(
                        groups, title=f"{name} by {cluster_key}", ylabel="fate prob"
                    )
                    paths = _helpers.save_figure(
                        fig, figures_dir, f"fate_violin_{name}", formats=formats, dpi=dpi
                    )
                    artifacts += _helpers.figure_artifacts(
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


# ══════════════════════════════════════════════════════════════════════════════
# DriverVizMethod (from driver_viz.py)
# ══════════════════════════════════════════════════════════════════════════════


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

        _helpers.apply_theme()
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
                fig = _helpers.signed_diverging_bar(sel_genes, vals[order], title=f"{name} drivers")
                paths = _helpers.save_figure(
                    fig, figures_dir, f"drivers_bar_{name}", formats=formats, dpi=dpi
                )
                artifacts += _helpers.figure_artifacts(
                    paths, name="trajectory_figure", description=f"Top-{top_k} drivers for {name}."
                )
                n += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"driver_viz: bar {name} failed: {str(exc)[:200]}")

        try:
            if top_gene_set:
                idx = {str(g): i for i, g in enumerate(genes)}
                rows = [mat[idx[g], :] for g in top_gene_set]
                fig = _helpers.matrix_heatmap(
                    np.vstack(rows),
                    top_gene_set,
                    names,
                    title="Lineage drivers",
                    cbar_label="correlation",
                )
                paths = _helpers.save_figure(
                    fig, figures_dir, "drivers_heatmap", formats=formats, dpi=dpi
                )
                artifacts += _helpers.figure_artifacts(
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


__all__ = ["VelocityVizMethod", "MacrostateVizMethod", "FateVizMethod", "DriverVizMethod"]
