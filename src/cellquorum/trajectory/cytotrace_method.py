"""CytoTraceMethod: whole-object CytoTRACE 2 developmental-potency scoring.

CytoTRACE 2 is a PRODUCER: it writes a ``cytotrace2_score`` obs column (plus the
categorical potency) that CellRank's CytoTRACEKernel can consume via the
``cytotrace_key`` seam on ``CellRankConfig``. The heavy, weight-downloading run
is import-guarded and retyped so the stage skips-not-crashes when the optional
``cytotrace2-py`` dependency is absent.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory import _cytotrace
from cellquorum.trajectory.save import write_pseudotime_h5ad


class CytoTraceMethod(AnalysisMethod):
    """CytoTRACE 2 potency scoring on the whole object."""

    name = "cytotrace"
    stage_category = "trajectory"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract(required_obs=[], required_layers=[])

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        notes: list[str] = []

        try:
            counts = _cytotrace.resolve_counts(adata, config.get("counts_layer"))
        except _cytotrace.CytoTraceComputeError as exc:
            return MethodSkip(reason=str(exc), details={"method": self.name})

        workdir = Path(context.paths.results) / "trajectory" / "cytotrace"
        try:
            frame = _cytotrace.run_cytotrace2(
                counts,
                var_names=[str(v) for v in adata.var_names],
                obs_names=[str(o) for o in adata.obs_names],
                species=str(config.get("species", "human")),
                workdir=workdir,
                seed=int(config.get("seed", 14)),
                disable_parallelization=bool(config.get("disable_parallelization", False)),
                batch_size=int(config.get("batch_size", 20000)),
                smooth_batch_size=int(config.get("smooth_batch_size", 1000)),
            )
        except _cytotrace.CytoTraceComputeError as exc:
            return MethodSkip(reason=str(exc), details={"method": self.name, "notes": notes})

        n_scored = self._writeback(adata, frame, notes)
        if n_scored == 0:
            return MethodSkip(
                reason="cytotrace: no cells could be aligned to results",
                details={"method": self.name, "notes": notes},
            )

        uns = adata.uns.setdefault("trajectory", {}).setdefault("cytotrace", {})
        uns.update(
            {
                "species": str(config.get("species", "human")),
                "n_cells_scored": int(n_scored),
                "counts_layer": config.get("counts_layer"),
                "seed": int(config.get("seed", 14)),
            }
        )

        artifacts: list[StageArtifact] = []
        results_dir = Path(context.paths.results) / "trajectory" / "cytotrace"
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001 — skip-not-crash
            notes.append(f"could not create results dir: {exc}")
        artifact, write_note = write_pseudotime_h5ad(adata, results_dir, "cytotrace", subset=False)
        notes.append(write_note)
        if artifact is not None:
            artifacts.append(artifact)

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics={
                "method": self.name,
                "n_cells_scored": int(n_scored),
                "species": str(config.get("species", "human")),
                "status": "success",
            },
            backend="python",
        )

    def _writeback(self, adata: ad.AnnData, frame: pd.DataFrame, notes: list[str]) -> int:
        """Align CytoTRACE 2 score/potency/relative back by obs_name (NaN outside).

        Returns the number of cells that matched a results row.
        """
        try:
            idx = adata.obs_names
            score_col = "CytoTRACE2_Score"
            pot_col = "CytoTRACE2_Potency"
            rel_col = "CytoTRACE2_Relative"
            if score_col in frame.columns:
                ser = pd.Series(np.asarray(frame[score_col], dtype="float64"), index=frame.index)
                adata.obs["cytotrace2_score"] = ser.reindex(idx)
            if pot_col in frame.columns:
                pot = frame[pot_col].astype(str)
                adata.obs["cytotrace2_potency"] = (
                    pd.Series(pot.to_numpy(), index=frame.index).reindex(idx).astype("category")
                )
            if rel_col in frame.columns:
                rel = pd.Series(np.asarray(frame[rel_col], dtype="float64"), index=frame.index)
                adata.obs["cytotrace2_relative"] = rel.reindex(idx)
        except Exception as exc:  # noqa: BLE001 — skip-not-crash
            notes.append(f"cytotrace obs writeback failed: {exc}")
            return 0
        matched = int(adata.obs_names.isin(frame.index).sum())
        return matched


__all__ = ["CytoTraceMethod"]
