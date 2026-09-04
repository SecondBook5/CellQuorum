"""VelocityMethod: loom I/O + scVelo velocity per cell-lineage group."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.core.context import resolve_n_jobs
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.trajectory import compute
from cellquorum.stages.trajectory._loom_io import reconcile_looms
from cellquorum.stages.trajectory._velocyto import ensure_loom
from cellquorum.stages.trajectory.config import VelocityGenerationConfig
from cellquorum.stages.trajectory.save import record_write, write_velocity_h5ad


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
            if velo_adata is not None:
                carried = self._carry_embeddings(adata, velo_adata)
                notes.append(
                    f"carried {len(carried)} embedding(s) onto the velocity object: "
                    f"{', '.join(carried)}"
                    if carried
                    else "no 2-D embeddings available to carry onto the velocity object"
                )
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

        # Resolved once, here, because it needs the context: the per-group and
        # whole-object paths below take only a config dict, and the stage config
        # leaves n_jobs unset by default so `compute.n_jobs` governs.
        n_jobs = resolve_n_jobs(context, config.get("n_jobs"))
        if n_jobs > 1:
            notes.append(f"velocity using {n_jobs} workers")

        per_group: list[dict] = []
        warnings: list[str] = []
        artifacts: list[StageArtifact] = []
        uns = adata.uns.setdefault("trajectory", {}).setdefault("velocity", {})

        # Track used stems to disambiguate colliding safe_names.
        used_stems: dict[str, int] = {}

        for group in levels:
            from cellquorum.stages.trajectory.save import safe_name

            base_stem = safe_name(group)
            if base_stem in used_stems:
                used_stems[base_stem] += 1
                stem = f"{base_stem}_{used_stems[base_stem]}"
            else:
                used_stems[base_stem] = 0
                stem = base_stem

            record = self._run_group(
                velo_adata, group, grouping_col, config, results_dir, stem, n_jobs
            )
            per_group.append(record["metrics"])
            notes.extend(record["notes"])
            warnings.extend(record["warnings"])
            if record["artifact"] is not None:
                artifacts.append(record["artifact"])
            uns[group] = record["metrics"]

            # Writeback per-group velocity obs columns onto the working adata.
            if record["writeback"] is not None:
                try:
                    for col_name, series in record["writeback"].items():
                        adata.obs[col_name] = series
                except Exception as exc:  # noqa: BLE001 — skip-not-crash
                    warnings.append(f"{group}: obs writeback failed: {exc}")

        # Optional whole-object velocity for CellRank's VelocityKernel. Runs on a
        # COPY so compute.compute_velocity's in-place HVG var-subset never touches
        # the working atlas. Skip-not-crash: any failure is a note, never a raise.
        if config.get("whole_object"):
            whole_artifact = self._run_whole_object(
                velo_adata, config, results_dir, notes, warnings, n_jobs
            )
            if whole_artifact is not None:
                artifacts.append(whole_artifact)

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            warnings=warnings,
            metrics={
                "method": self.name,
                "n_groups": len(levels),
                "per_group": per_group,
            },
            backend="python",
        )

    @staticmethod
    def _carry_embeddings(source: ad.AnnData, target: ad.AnnData) -> list[str]:
        """Copy 2-D embeddings from `source` onto `target` for shared cells.

        `reconcile_looms` builds a FRESH object from the loom files, so it carries
        obs but no obsm. Without this the object has no embedding at all:
        `embedding_bases()` returns [], `reproject_velocity()` is a silent no-op,
        and every downstream velocity figure has no basis to draw on — which is
        exactly why the rendered streams were unusable. Returns the basis names
        copied.
        """
        import numpy as np

        if source is None or target is None:
            return []
        shared = target.obs_names.intersection(source.obs_names)
        if len(shared) == 0:
            return []
        # Align by name, not position: the loom reconciliation may drop or reorder
        # cells, and a positional copy would silently scramble the embedding.
        source_rows = source.obs_names.get_indexer(target.obs_names)
        present = source_rows >= 0
        copied = []
        for key in list(source.obsm.keys()):
            if not key.startswith("X_") or key == "X_pca":
                continue
            values = np.asarray(source.obsm[key])
            if values.ndim != 2 or values.shape[1] < 2:
                continue
            carried = np.full((target.n_obs, values.shape[1]), np.nan, dtype=float)
            carried[present] = values[source_rows[present]]
            target.obsm[key] = carried
            copied.append(key)
        return copied

    def _run_whole_object(
        self,
        velo_adata: ad.AnnData,
        config: dict,
        results_dir: Path,
        notes: list[str],
        warnings: list[str],
        n_jobs: int,
    ) -> StageArtifact | None:
        """Compute velocity once on the WHOLE object; write whole_object.h5ad.

        Runs on a copy (compute.compute_velocity subsets vars in place, so the
        caller's atlas must not be the target). Never raises (skip-not-crash);
        every failure path returns None and WARNS rather than notes, because this
        one file is what CellRank's VelocityKernel consumes — without it the whole
        trajectory result quietly changes meaning.
        """
        from cellquorum.stages.trajectory.save import write_whole_object_velocity_h5ad

        whole = velo_adata.copy()
        rep = compute.resolve_use_rep(
            whole, config.get("use_rep"), config.get("use_rep_fallback", ["X_pca"])
        )
        if rep is None:
            warnings.append("whole-object velocity skipped: no usable representation")
            return None

        try:
            compute.compute_velocity(
                whole,
                mode=config.get("mode", "dynamical"),
                use_rep=rep,
                min_shared_counts=int(config.get("min_shared_counts", 20)),
                n_top_genes=int(config.get("n_top_genes", 2000)),
                n_pcs=int(config.get("n_pcs", 30)),
                n_neighbors=int(config.get("n_neighbors", 30)),
                n_jobs=n_jobs,
                seed=int(config.get("seed", 1337)),
                warnings=warnings,
            )
        except compute.TrajectoryComputeError as exc:
            warnings.append(f"whole-object velocity skipped: {exc}")
            return None

        compute.reproject_velocity(whole, bases=compute.embedding_bases(whole))
        return record_write(
            write_whole_object_velocity_h5ad(whole, results_dir),
            notes=notes,
            warnings=warnings,
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
        stem: str,
        n_jobs: int,
    ) -> dict:
        """Compute velocity for one group; never raises (skip-not-crash).

        Args:
            stem: Collision-free filename stem for the h5ad output.
            n_jobs: Resolved worker count (see `resolve_n_jobs`).

        Returns:
            ``{artifact, notes, warnings, writeback, metrics}``. The split between
            notes and warnings is the point: a group below ``min_cells`` is the
            gate working as configured, while a failed computation or a failed
            write means a group the caller asked for has no velocity at all.
        """
        notes: list[str] = []
        warnings: list[str] = []
        mask = (velo_adata.obs[grouping_col].astype(str) == group).to_numpy()
        sub = velo_adata[mask].copy()
        min_cells = int(config.get("min_cells", 30))
        if sub.n_obs < min_cells:
            # Configured, expected and harmless: min_cells is the gate doing its
            # job, so this stays a note.
            return {
                "artifact": None,
                "notes": [f"{group}: {sub.n_obs} cells < min_cells"],
                "warnings": [],
                "writeback": None,
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
                "notes": [],
                "warnings": [f"{group}: no usable representation"],
                "writeback": None,
                "metrics": {
                    "group": group,
                    "n_cells": int(sub.n_obs),
                    "status": "skipped",
                    "skip_reason": "no representation",
                    "rep": None,
                },
            }

        # Collected separately so every degradation carries the group it belongs
        # to: a bare "velocity_pseudotime failed" in a 16-group run says nothing.
        degradations: list[str] = []
        try:
            compute.compute_velocity(
                sub,
                mode=config.get("mode", "dynamical"),
                use_rep=rep,
                min_shared_counts=int(config.get("min_shared_counts", 20)),
                n_top_genes=int(config.get("n_top_genes", 2000)),
                n_pcs=int(config.get("n_pcs", 30)),
                n_neighbors=int(config.get("n_neighbors", 30)),
                n_jobs=n_jobs,
                seed=int(config.get("seed", 1337)),
                warnings=degradations,
            )
        except compute.TrajectoryComputeError as exc:
            return {
                "artifact": None,
                "notes": [],
                "warnings": [f"{group}: velocity computation failed: {exc}"],
                "writeback": None,
                "metrics": {
                    "group": group,
                    "n_cells": int(sub.n_obs),
                    "status": "skipped",
                    "skip_reason": str(exc),
                    "rep": rep,
                },
            }

        warnings.extend(f"{group}: {message}" for message in degradations)

        bases = compute.reproject_velocity(sub, bases=compute.embedding_bases(sub))
        artifact = record_write(
            write_velocity_h5ad(sub, results_dir, group, stem=stem),
            notes=notes,
            warnings=warnings,
        )

        # Build per-group namespaced obs columns for writeback onto the working adata.
        # The stem is collision-free; use it for namespacing.
        import pandas as pd

        writeback: dict[str, pd.Series] = {}
        if "velocity_pseudotime" in sub.obs:
            col_name = f"velocity_pseudotime_{stem}"
            # Series aligned to the working adata's full obs_names; NaN for out-of-group cells.
            writeback[col_name] = pd.Series(sub.obs["velocity_pseudotime"], index=sub.obs_names)
        if "velocity_confidence" in sub.obs:
            col_name = f"velocity_confidence_{stem}"
            writeback[col_name] = pd.Series(sub.obs["velocity_confidence"], index=sub.obs_names)

        return {
            "artifact": artifact,
            "notes": notes,
            "warnings": warnings,
            "writeback": writeback if writeback else None,
            "metrics": {
                "group": group,
                "n_cells": int(sub.n_obs),
                "status": "success",
                "skip_reason": None,
                "rep": rep,
                "mode": config.get("mode", "dynamical"),
                "reprojected": bases,
                # A group can now succeed WITHOUT pseudotime (the eigensolve is
                # allowed to degrade), so the per-group table has to say which.
                "pseudotime": bool("velocity_pseudotime" in sub.obs),
            },
        }


__all__ = ["VelocityMethod"]
