"""DptMethod: whole-object diffusion pseudotime (scanpy)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory import _pseudotime
from cellquorum.trajectory.save import write_pseudotime_h5ad


class DptMethod(AnalysisMethod):
    """Diffusion pseudotime on the whole object with a resolved root."""

    name = "dpt"
    stage_category = "trajectory"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract(required_obs=[], required_layers=[])

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        notes: list[str] = []

        rep = _pseudotime.resolve_rep(
            adata, config.get("use_rep"), config.get("use_rep_fallback", ["X_pca"])
        )
        if rep is None:
            return MethodSkip(reason="dpt: no usable representation", details={"method": self.name})

        exclude = bool(config.get("exclude_outliers", False))
        marker_key = config.get("root_marker_score_key")
        root_key = config.get("root_key")
        root_group = config.get("root_group")

        work = adata
        outlier_mask = None
        if exclude:
            outlier_mask = _pseudotime.flag_outliers(
                adata, rep, float(config.get("outlier_mad", 5.0))
            )
            if outlier_mask.any():
                work = adata[~outlier_mask].copy()
                notes.append(f"excluded {int(outlier_mask.sum())} outliers before dpt")

        # Resolve root on the working object.
        try:
            iroot = _pseudotime.resolve_root(
                work,
                rep=rep,
                marker_score_key=marker_key,
                root_key=root_key,
                root_group=root_group,
            )
        except _pseudotime.PseudotimeComputeError as exc:
            return MethodSkip(reason=str(exc), details={"method": self.name})
        root_source = "marker_score" if (marker_key and marker_key in work.obs) else "root_group"

        try:
            res = _pseudotime.compute_dpt(
                work,
                use_rep=rep,
                use_rep_fallback=config.get("use_rep_fallback", ["X_pca"]),
                n_neighbors=int(config.get("n_neighbors", 15)),
                n_comps=int(config.get("n_comps", 15)),
                n_dcs=int(config.get("n_dcs", 10)),
                n_branchings=int(config.get("n_branchings", 0)),
                iroot=iroot,
            )
        except _pseudotime.PseudotimeComputeError as exc:
            return MethodSkip(reason=str(exc), details={"method": self.name, "notes": notes})
        notes.extend(res.get("notes", []))

        oriented = False
        orient_key = config.get("orient_by_score_key")
        if orient_key and orient_key in work.obs:
            oriented = self._maybe_reorient(work, res, orient_key, notes)

        self._writeback(adata, work, res, outlier_mask, notes)

        uns = adata.uns.setdefault("trajectory", {}).setdefault("dpt", {})
        uns.update(
            {
                "root_index": int(iroot),
                "root_source": root_source,
                "n_dcs": res["n_dcs"],
                "n_branchings": int(config.get("n_branchings", 0)),
                "excluded_outliers": bool(
                    exclude and outlier_mask is not None and outlier_mask.any()
                ),
                "oriented": oriented,
            }
        )

        artifacts: list[StageArtifact] = []
        results_dir = Path(context.paths.results) / "trajectory" / "dpt"
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"could not create results dir: {exc}")
        artifact, write_note = write_pseudotime_h5ad(work, results_dir, "dpt")
        notes.append(write_note)
        if artifact is not None:
            artifacts.append(artifact)

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics={
                "method": self.name,
                "root_index": int(iroot),
                "root_source": root_source,
                "excluded_outliers": bool(exclude),
                "oriented": oriented,
                "status": "success",
            },
            backend="python",
        )

    def _maybe_reorient(
        self, work: ad.AnnData, res: dict, orient_key: str, notes: list[str]
    ) -> bool:
        """Sign-check corr(dpt, score); re-root once at argmax(score) if reversed."""
        import scanpy as sc

        pt_vals = np.asarray(res["pseudotime"], dtype="float64")
        score = np.asarray(work.obs[orient_key], dtype="float64")
        finite = np.isfinite(pt_vals) & np.isfinite(score)
        if finite.sum() < 3:
            return False
        corr = float(np.corrcoef(pt_vals[finite], score[finite])[0, 1])
        # A stemness score should DECREASE along pseudotime → negative corr expected.
        if corr > 0:
            work.uns["iroot"] = int(np.argmax(score))
            try:
                sc.tl.dpt(work, n_dcs=int(res["n_dcs"]))
            except Exception as exc:  # noqa: BLE001
                notes.append(f"re-root dpt failed: {exc}")
                return False
            res["pseudotime"] = np.asarray(work.obs["dpt_pseudotime"], dtype="float64")
            notes.append("re-rooted dpt for orientation")
            return True
        return False

    def _writeback(
        self,
        adata: ad.AnnData,
        work: ad.AnnData,
        res: dict,
        outlier_mask: np.ndarray | None,
        notes: list[str],
    ) -> None:
        """Align dpt_pseudotime + X_diffmap back to the full object by obs_name."""
        try:
            if outlier_mask is not None and outlier_mask.any():
                ser = pd.Series(np.asarray(res["pseudotime"]), index=work.obs_names)
                adata.obs["dpt_pseudotime"] = ser.reindex(adata.obs_names)
            else:
                adata.obs["dpt_pseudotime"] = np.asarray(res["pseudotime"])
        except Exception as exc:  # noqa: BLE001
            notes.append(f"dpt obs writeback failed: {exc}")

        try:
            if "X_diffmap" in work.obsm:
                dm = np.asarray(work.obsm["X_diffmap"])
                if outlier_mask is not None and outlier_mask.any():
                    full = np.zeros((adata.n_obs, dm.shape[1]), dtype=dm.dtype)
                    pos = {n: i for i, n in enumerate(adata.obs_names)}
                    rows = [pos[c] for c in work.obs_names if c in pos]
                    full[np.array(rows), :] = dm
                    adata.obsm["X_diffmap"] = full
                else:
                    adata.obsm["X_diffmap"] = dm
        except Exception as exc:  # noqa: BLE001
            notes.append(f"dpt obsm writeback failed: {exc}")


__all__ = ["DptMethod"]
