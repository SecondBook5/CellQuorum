"""VelocityMethod: loom I/O + scVelo velocity per cell-lineage group."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.trajectory import compute
from cellquorum.trajectory._loom_io import reconcile_looms
from cellquorum.trajectory._velocyto import ensure_loom
from cellquorum.trajectory.config import VelocityGenerationConfig
from cellquorum.trajectory.save import write_velocity_h5ad


class VelocityMethod(AnalysisMethod):
    """RNA velocity via scVelo, computed once per group and re-projected."""

    name = "velocity"
    stage_category = "trajectory"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        # spliced/unspliced are attached at runtime from looms, so no X/layer
        # precondition on the incoming atlas.
        return DataContract(required_obs=[], required_layers=[])

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        sample_col = config.get("sample_col", "sample_id")
        loom_path_col = config.get("loom_path_col", "loom_path")
        grouping_col = config.get("grouping_col", "cell_type")

        # 1. Resolve the manifest (skip-not-crash if absent).
        try:
            manifest = context.require_manifest()
        except Exception:
            manifest = None

        notes: list[str] = []

        # 2. Optionally generate missing looms, then attach spliced/unspliced.
        if manifest is not None:
            manifest = self._maybe_generate(manifest, config, sample_col, loom_path_col, notes)
            velo_adata, io_notes = reconcile_looms(
                adata, manifest, sample_col=sample_col, loom_path_col=loom_path_col
            )
            notes.extend(io_notes)
        else:
            velo_adata = None
            notes.append("no manifest available")

        if velo_adata is None:
            return MethodSkip(
                reason="velocity requires spliced/unspliced counts",
                details={"method": self.name, "notes": notes},
            )

        # 3. Per-group velocity.
        if grouping_col not in velo_adata.obs:
            return MethodSkip(
                reason=f"grouping_col '{grouping_col}' not in obs",
                details={"method": self.name},
            )

        configured_groups = config.get("groups")
        levels = (
            sorted(str(g) for g in configured_groups)
            if configured_groups
            else sorted(velo_adata.obs[grouping_col].astype(str).unique())
        )

        results_dir = Path(context.paths.results) / "trajectory" / "velocity"
        results_dir.mkdir(parents=True, exist_ok=True)

        per_group: list[dict] = []
        artifacts: list[StageArtifact] = []
        uns = adata.uns.setdefault("trajectory", {}).setdefault("velocity", {})

        for group in levels:
            record = self._run_group(velo_adata, group, grouping_col, config, results_dir)
            per_group.append(record["metrics"])
            notes.extend(record["notes"])
            if record["artifact"] is not None:
                artifacts.append(record["artifact"])
            uns[group] = record["metrics"]

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics={
                "method": self.name,
                "n_groups": len(levels),
                "per_group": per_group,
            },
            backend="python",
        )

    def _maybe_generate(
        self,
        manifest: pd.DataFrame,
        config: dict,
        sample_col: str,
        loom_path_col: str,
        notes: list[str],
    ) -> pd.DataFrame:
        """Fill missing loom_path entries via the generation harness when gated."""
        gen_dict = config.get("generation", {}) or {}
        gen = (
            VelocityGenerationConfig(**gen_dict)
            if not isinstance(gen_dict, VelocityGenerationConfig)
            else gen_dict
        )
        if not gen.generate_missing or gen.bam_dir is None:
            return manifest
        manifest = manifest.copy()
        if loom_path_col not in manifest.columns:
            manifest[loom_path_col] = None
        for idx in manifest.index:
            existing = manifest.at[idx, loom_path_col]
            if existing is not None and str(existing) and Path(str(existing)).exists():
                continue
            sample_id = str(manifest.at[idx, sample_col])
            sample_dir = Path(gen.bam_dir) / sample_id
            loom, reason = ensure_loom(sample_id, sample_dir, gen)
            notes.append(f"{sample_id}: {reason}")
            if loom is not None:
                manifest.at[idx, loom_path_col] = str(loom)
        return manifest

    def _run_group(
        self,
        velo_adata: ad.AnnData,
        group: str,
        grouping_col: str,
        config: dict,
        results_dir: Path,
    ) -> dict:
        """Compute velocity for one group; never raises (skip-not-crash)."""
        notes: list[str] = []
        mask = (velo_adata.obs[grouping_col].astype(str) == group).to_numpy()
        sub = velo_adata[mask].copy()
        min_cells = int(config.get("min_cells", 30))
        if sub.n_obs < min_cells:
            return {
                "artifact": None,
                "notes": [f"{group}: {sub.n_obs} cells < min_cells"],
                "metrics": {
                    "group": group,
                    "n_cells": int(sub.n_obs),
                    "status": "skipped",
                    "skip_reason": "too few cells",
                    "rep": None,
                },
            }

        rep = compute.resolve_use_rep(
            sub, config.get("use_rep"), config.get("use_rep_fallback", ["X_pca"])
        )
        if rep is None:
            return {
                "artifact": None,
                "notes": [f"{group}: no usable representation"],
                "metrics": {
                    "group": group,
                    "n_cells": int(sub.n_obs),
                    "status": "skipped",
                    "skip_reason": "no representation",
                    "rep": None,
                },
            }

        try:
            compute.compute_velocity(
                sub,
                mode=config.get("mode", "dynamical"),
                use_rep=rep,
                min_shared_counts=int(config.get("min_shared_counts", 20)),
                n_top_genes=int(config.get("n_top_genes", 2000)),
                n_pcs=int(config.get("n_pcs", 30)),
                n_neighbors=int(config.get("n_neighbors", 30)),
                n_jobs=int(config.get("n_jobs", 1)),
                seed=int(config.get("seed", 1337)),
            )
        except compute.TrajectoryComputeError as exc:
            return {
                "artifact": None,
                "notes": [f"{group}: {exc}"],
                "metrics": {
                    "group": group,
                    "n_cells": int(sub.n_obs),
                    "status": "skipped",
                    "skip_reason": str(exc),
                    "rep": rep,
                },
            }

        bases = compute.reproject_velocity(sub, bases=compute.embedding_bases(sub))
        artifact, write_note = write_velocity_h5ad(sub, results_dir, group)
        notes.append(write_note)
        return {
            "artifact": artifact,
            "notes": notes,
            "metrics": {
                "group": group,
                "n_cells": int(sub.n_obs),
                "status": "success",
                "skip_reason": None,
                "rep": rep,
                "mode": config.get("mode", "dynamical"),
                "reprojected": bases,
            },
        }


__all__ = ["VelocityMethod"]
