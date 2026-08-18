"""Condition-split annotated pseudotime heatmap method."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory_viz import heatmap, inputs
from cellquorum.visualization import figstyle

_SCORE_FALLBACKS = ("G2M_score", "S_score", "phase_score")
_STATE_FALLBACKS = ("cell_type", "leiden")


def _dense(mat: object) -> np.ndarray:
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


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
        pts = inputs.available_pseudotimes(adata, config.get("pseudotime_keys"))
        if not pts:
            return MethodSkip(
                reason="pseudotime_heatmap skipped: no *_pseudotime obs column",
                details={"method": self.name},
            )
        pt_key = pts[0]

        try:
            pt = inputs.numeric_obs(adata, pt_key)

            genes = self._select_genes(adata, pt, config)
            if not genes:
                return MethodSkip(
                    reason="pseudotime_heatmap skipped: no genes resolved",
                    details={"method": self.name},
                )
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
                return MethodSkip(
                    reason="pseudotime_heatmap skipped: no condition had enough cells",
                    details={"method": self.name},
                )
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
            return MethodSkip(
                reason=f"pseudotime_heatmap skipped: input/render failed ({str(exc)[:200]})",
                details={"method": self.name},
            )
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


__all__ = ["PseudotimeHeatmapVizMethod"]
