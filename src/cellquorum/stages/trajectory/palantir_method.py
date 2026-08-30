"""PalantirMethod: whole-object Palantir pseudotime + entropy."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.trajectory import _pseudotime
from cellquorum.stages.trajectory.save import write_pseudotime_h5ad


class PalantirMethod(AnalysisMethod):
    """Palantir pseudotime/entropy on the whole object with a resolved root."""

    name = "palantir"
    stage_category = "trajectory"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract(required_obs=[], required_layers=[])

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        seed = int(config.get("seed", 1337))
        notes: list[str] = []

        rep = _pseudotime.resolve_rep(
            adata, config.get("use_rep"), config.get("use_rep_fallback", ["X_pca"])
        )
        if rep is None:
            return MethodSkip(
                reason="palantir: no usable representation", details={"method": self.name}
            )

        marker_key = config.get("root_marker_score_key")
        root_key = config.get("root_key")
        root_group = config.get("root_group")
        try:
            iroot = _pseudotime.resolve_root(
                adata,
                rep=rep,
                marker_score_key=marker_key,
                root_key=root_key,
                root_group=root_group,
            )
        except _pseudotime.PseudotimeComputeError as exc:
            return MethodSkip(reason=str(exc), details={"method": self.name})
        root_source = "marker_score" if (marker_key and marker_key in adata.obs) else "root_group"

        # Optional seeded subsample (root always retained).
        max_cells = config.get("max_cells")
        subsampled = False
        work = adata
        if max_cells is not None and adata.n_obs > int(max_cells):
            rng = np.random.default_rng(seed)
            idx = rng.choice(adata.n_obs, size=int(max_cells), replace=False)
            if iroot not in idx:
                idx[-1] = iroot
            idx = np.sort(idx)
            work = adata[idx].copy()
            subsampled = True
            notes.append(f"subsampled to {int(max_cells)} cells for palantir")

        early_cell = str(adata.obs_names[iroot])

        try:
            res = _pseudotime.compute_palantir(
                work,
                use_rep=rep,
                use_rep_fallback=config.get("use_rep_fallback", ["X_pca"]),
                n_components=int(config.get("n_components", 10)),
                knn=int(config.get("knn", 30)),
                n_eigs=int(config.get("n_eigs", 10)),
                num_waypoints=int(config.get("num_waypoints", 1200)),
                early_cell=early_cell,
                seed=seed,
            )
        except _pseudotime.PseudotimeComputeError as exc:
            return MethodSkip(reason=str(exc), details={"method": self.name, "notes": notes})
        notes.extend(res.get("notes", []))

        self._writeback(adata, work, res, notes)

        uns = adata.uns.setdefault("trajectory", {}).setdefault("palantir", {})
        uns.update(
            {
                "root_index": int(iroot),
                "root_source": root_source,
                "num_waypoints": min(int(config.get("num_waypoints", 1200)), int(work.n_obs)),
                "n_cells_used": int(work.n_obs),
                "subsampled": subsampled,
                "fate_names": res["fate_names"],
            }
        )

        artifacts: list[StageArtifact] = []
        results_dir = Path(context.paths.results) / "trajectory" / "palantir"
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"could not create results dir: {exc}")
        artifact, write_note = write_pseudotime_h5ad(
            work, results_dir, "palantir", subset=subsampled
        )
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
                "n_cells_used": int(work.n_obs),
                "subsampled": subsampled,
                "n_fates": len(res["fate_names"]),
                "status": "success",
            },
            backend="python",
        )

    def _writeback(self, adata: ad.AnnData, work: ad.AnnData, res: dict, notes: list[str]) -> None:
        """Align palantir pseudotime/entropy/fate-probs back by obs_name (NaN outside)."""
        try:
            pt_ser = pd.Series(np.asarray(res["pseudotime"]), index=list(res["pseudotime"].index))
            adata.obs["palantir_pseudotime"] = pt_ser.reindex(adata.obs_names)
            ent_ser = pd.Series(np.asarray(res["entropy"]), index=list(res["entropy"].index))
            adata.obs["palantir_entropy"] = ent_ser.reindex(adata.obs_names)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"palantir obs writeback failed: {exc}")

        try:
            if res["fate_prob"] is not None and res["fate_names"]:
                n_fate = len(res["fate_names"])
                full = np.full((adata.n_obs, n_fate), np.nan, dtype="float32")
                pos = {n: i for i, n in enumerate(adata.obs_names)}
                rows = [pos[c] for c in work.obs_names if c in pos]
                src_rows = [i for i, c in enumerate(work.obs_names) if c in pos]
                full[np.array(rows), :] = np.asarray(res["fate_prob"])[np.array(src_rows), :]
                adata.obsm["palantir_fate_probabilities"] = full
        except Exception as exc:  # noqa: BLE001
            notes.append(f"palantir obsm writeback failed: {exc}")


__all__ = ["PalantirMethod"]
