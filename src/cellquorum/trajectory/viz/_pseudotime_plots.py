"""Pseudotime-ordered plot methods: embedding scatter, GAM gene trends, and heatmaps.

Combines:
- PseudotimeVizMethod (scatter of pseudotime on embedding)
- GeneTrendVizMethod (CellRank GAM gene trends along pseudotime)
- PseudotimeHeatmapVizMethod (condition-split annotated pseudotime heatmap)
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory.viz import _helpers, heatmap
from cellquorum.visualization import figstyle

_SCORE_FALLBACKS = ("G2M_score", "S_score", "phase_score")
_STATE_FALLBACKS = ("cell_type", "leiden")


def _dense(mat: object) -> np.ndarray:
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


# ══════════════════════════════════════════════════════════════════════════════
# PseudotimeVizMethod (from pseudotime_viz.py)
# ══════════════════════════════════════════════════════════════════════════════


class PseudotimeVizMethod(AnalysisMethod):
    name = "pseudotime_viz"
    stage_category = "trajectory_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        basis = _helpers.resolve_basis(adata, config.get("embedding_basis"))
        if basis is None:
            return self._skip("no embedding basis in obsm")
        pts = _helpers.available_pseudotimes(adata, config.get("pseudotime_keys"))
        if not pts:
            return self._skip("no *_pseudotime obs column")

        figures_dir = Path(context.paths.figures) / "trajectory"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        coords = adata.obsm[basis]

        _helpers.apply_theme()
        artifacts, warnings, n = [], [], 0
        for key in pts:
            try:
                values = _helpers.numeric_obs(adata, key)
                fig = _helpers.embedding_scatter(coords, values, title=key, cbar_label=key)
                paths = _helpers.save_figure(
                    fig, figures_dir, f"pseudotime_{key}", formats=formats, dpi=dpi
                )
                artifacts += _helpers.figure_artifacts(
                    paths, name="trajectory_figure", description=f"{key} on {basis}."
                )
                n += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"pseudotime_viz: {key} failed: {str(exc)[:200]}")

        if n == 0:
            return self._skip("no plottable pseudotime", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"pseudotime_viz rendered {n} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n, "basis": basis},
            backend="python",
        )


# ══════════════════════════════════════════════════════════════════════════════
# GeneTrendVizMethod (from gene_trend_viz.py)
# ══════════════════════════════════════════════════════════════════════════════


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
        fate_path = _helpers.results_file(context, "cellrank", "fate_mapping.h5ad")
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
        _helpers.apply_theme()
        warnings = []
        try:
            model = cr.models.GAM(fate)
            cr.pl.gene_trends(fate, model=model, genes=present, show=False)
            paths = _helpers.save_figure(
                plt.gcf(), figures_dir, "gene_trends", formats=formats, dpi=dpi
            )
            artifacts = _helpers.figure_artifacts(
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


# ══════════════════════════════════════════════════════════════════════════════
# PseudotimeHeatmapVizMethod (from pseudotime_heatmap_viz.py)
# ══════════════════════════════════════════════════════════════════════════════


class PseudotimeHeatmapVizMethod(AnalysisMethod):
    name = "pseudotime_heatmap"
    stage_category = "trajectory_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _resolve_key(
        self, adata: ad.AnnData, configured: str | None, fallbacks: tuple[str, ...]
    ) -> str | None:
        if configured and configured in adata.obs:
            return configured
        for key in fallbacks:
            if key in adata.obs:
                return key
        return None

    def _select_genes(self, adata: ad.AnnData, pt: np.ndarray, config: dict) -> list[str]:
        genes = config.get("heatmap_genes") or config.get("genes")
        present = [g for g in (genes or []) if g in adata.var_names]
        if present:
            return present
        # Fallback: DPT-correlation prefilter.
        from scipy.stats import spearmanr

        corr_cut = float(config.get("heatmap_corr_cut", 0.1))
        max_genes = int(config.get("heatmap_max_genes", 60))
        X = _dense(adata.X)
        finite = np.isfinite(pt)
        rhos = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            r, _ = spearmanr(X[finite, j], pt[finite])
            rhos[j] = 0.0 if np.isnan(r) else abs(r)
        idx = np.where(rhos > corr_cut)[0]
        idx = idx[np.argsort(-rhos[idx])][:max_genes]
        return [str(adata.var_names[i]) for i in idx]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        pts = _helpers.available_pseudotimes(adata, config.get("pseudotime_keys"))
        if not pts:
            return self._skip("no *_pseudotime obs column")
        pt_key = pts[0]

        try:
            pt = _helpers.numeric_obs(adata, pt_key)

            genes = self._select_genes(adata, pt, config)
            if not genes:
                return self._skip("no genes resolved")
            gi = [adata.var_names.get_loc(g) for g in genes]
            M = _dense(adata.layers["magic"] if "magic" in adata.layers else adata.X)[:, gi]

            score_key = self._resolve_key(adata, config.get("heatmap_score_key"), _SCORE_FALLBACKS)
            state_key = self._resolve_key(adata, config.get("heatmap_state_key"), _STATE_FALLBACKS)
            cond_key = config.get("condition_col")
            cond_present = bool(cond_key) and cond_key in adata.obs

            n_bins = int(config.get("heatmap_n_bins", 100))
            finite = np.isfinite(pt)

            score = adata.obs[score_key].to_numpy() if score_key else np.full(adata.n_obs, np.nan)
            if state_key:
                state_series = adata.obs[state_key].astype(str)
                state_cats = sorted(state_series.unique())
                code_map = {c: i for i, c in enumerate(state_cats)}
                state_codes = state_series.map(code_map).to_numpy()
                state_colors = [
                    figstyle.CATEGORICAL_PALETTE[i % len(figstyle.CATEGORICAL_PALETTE)]
                    for i in range(len(state_cats))
                ]
            else:
                state_cats, state_colors, state_codes = [], [], np.full(adata.n_obs, -1)

            if cond_present:
                cond = adata.obs[cond_key].astype(str).to_numpy()
                case = config.get("case")
                control = config.get("control")
                ordered = [c for c in (control, case) if c and c in set(cond)]
                if not ordered:
                    ordered = sorted(set(cond))
                condition_order = ordered
            else:
                cond = np.array(["all"] * adata.n_obs)
                condition_order = ["all"]

            profiles, tracks = {}, {}
            for c in condition_order:
                m = (cond == c) & finite
                if m.sum() < 2:
                    continue
                p = heatmap.binned_profile(pt[m], M[m], n_bins)
                p = (p - p.min(0)) / (np.ptp(p, axis=0) + 1e-9)
                profiles[c] = p
                tracks[c] = heatmap.binned_tracks(pt[m], score[m], state_codes[m], n_bins)
            if not profiles:
                return self._skip("no condition had enough cells")
            condition_order = [c for c in condition_order if c in profiles]

            combined = np.mean([profiles[c] for c in condition_order], axis=0)
            gene_order = heatmap.peak_bin_order(combined)

            present_codes = sorted(
                set(int(v) for c in condition_order for v in tracks[c][1] if v >= 0)
            )

            figstyle.set_style()
            figures_dir = Path(context.paths.figures) / "trajectory"
            formats = tuple(config.get("figure_formats", ["pdf", "png"]))
            dpi = int(config.get("dpi", 300))
            fig = heatmap.condition_split_heatmap(
                profiles,
                tracks,
                genes,
                gene_order,
                condition_order=condition_order,
                state_cats=state_cats,
                state_colors=state_colors,
                present_state_codes=present_codes,
                expr_cmap=config.get("heatmap_expr_cmap", figstyle.SEQUENTIAL_CMAP),
                title=f"Expression along {pt_key}",
            )
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"input/render failed ({str(exc)[:200]})")
        paths = figstyle.save_figure(
            fig, figures_dir, "pseudotime_heatmap", formats=formats, dpi=dpi
        )
        artifacts = [
            StageArtifact(
                name="trajectory_figure",
                path=p,
                kind="figure",
                description="Condition-split pseudotime heatmap.",
            )
            for p in paths
        ]
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"pseudotime_heatmap rendered {len(paths)} files over {condition_order}."],
            warnings=[],
            metrics={"method": self.name, "n_genes": len(genes), "conditions": condition_order},
            backend="python",
        )


__all__ = ["PseudotimeVizMethod", "GeneTrendVizMethod", "PseudotimeHeatmapVizMethod"]
