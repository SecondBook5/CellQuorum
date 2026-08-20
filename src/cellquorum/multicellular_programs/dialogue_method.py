"""DIALOGUE method: cross-cell-type multicellular programs (MCPs) via R.

Orchestrates the bundled ``dialogue.R`` script through the Rscript backend:
export per-cell-type inputs (``_dialogue_io``), run DIALOGUE, read the flat-CSV
outputs, compute donor-support + program-stability diagnostics, and emit the
program/score/association tables as stage artifacts. Every precondition failure,
missing backend/package, timeout, or nonzero R exit is a recorded skip -- the
method never raises out of ``_run``.

Ruling R3: any configured confounder that is absent from ``obs`` or non-numeric
skips loudly (never silently dropped), and the confounder list is threaded to
``dialogue.R`` as a trailing argv so DIALOGUE's ``covar`` actually uses it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod
from cellquorum.multicellular_programs._dialogue_io import (
    export_dialogue_inputs,
    read_dialogue_outputs,
)
from cellquorum.multicellular_programs.diagnostics import (
    donor_support,
    match_program_loadings,
    program_stability,
)

_DIALOGUE_R = Path(__file__).parent.parent / "backends" / "r_scripts" / "dialogue.R"


class MulticellularProgramsMethod(RAnalysisMethod):
    """DIALOGUE-based inference of cross-cell-type coordinated programs."""

    name = "dialogue"
    stage_category = "multicellular_programs"
    r_package = "DIALOGUE"

    def input_contract(self, config: dict) -> DataContract:
        required = [c for c in (config.get("cell_type_col"), config.get("sample_col")) if c]
        return DataContract(required_obs=required)

    def requires_obs(self, config: dict) -> list[str]:
        return [c for c in (config.get("cell_type_col"), config.get("sample_col")) if c]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        # ---- Config resolution (cohort overlay already applied by the stage). ----
        use_rep = config.get("use_rep", "X_pca")
        cell_type_col = config.get("cell_type_col")
        sample_col = config.get("sample_col")
        donor_col = config.get("donor_col")
        condition_col = config.get("condition_col")
        layer = config.get("layer")
        quality_col = config.get("quality_col")
        confounders = list(config.get("confounders", []) or [])

        n_pcs = int(config.get("n_pcs", 10))
        n_programs = int(config.get("n_programs", 5))
        n_program_genes = int(config.get("n_program_genes", 200))
        min_cells_per_type = int(config.get("min_cells_per_type", 20))
        min_cell_types = int(config.get("min_cell_types", 2))
        min_samples = int(config.get("min_samples", 4))
        stability_resamples = int(config.get("stability_resamples", 5))
        donor_support_min = int(config.get("donor_support_min", 2))
        seed = int(config.get("seed", 0))
        timeout = int(config.get("timeout_seconds", 7200))

        # ---- Skip order: use_rep -> cols -> cell types -> samples -> confounders. ----
        if use_rep not in adata.obsm:
            return self._skip(f"use_rep '{use_rep}' missing from adata.obsm")

        if not cell_type_col or cell_type_col not in adata.obs.columns:
            return self._skip(f"cell_type_col '{cell_type_col}' unresolved or absent from obs")
        if not sample_col or sample_col not in adata.obs.columns:
            return self._skip(f"sample_col '{sample_col}' unresolved or absent from obs")

        type_counts = adata.obs[cell_type_col].value_counts()
        eligible_types = type_counts[type_counts >= min_cells_per_type]
        if len(eligible_types) < min_cell_types:
            return self._skip(
                f"only {len(eligible_types)} cell type(s) with >= {min_cells_per_type} "
                f"cells (need {min_cell_types})",
                n_eligible_cell_types=int(len(eligible_types)),
                min_cell_types=min_cell_types,
            )

        n_samples = int(adata.obs[sample_col].nunique())
        if n_samples < min_samples:
            return self._skip(
                f"only {n_samples} sample(s) (need {min_samples})",
                n_samples=n_samples,
                min_samples=min_samples,
            )

        # Ruling R3: confounders must be present and numeric -- skip loudly, never drop.
        for name in confounders:
            if name not in adata.obs.columns or not pd.api.types.is_numeric_dtype(adata.obs[name]):
                return self._skip(
                    f"confounder column '{name}' missing or non-numeric",
                    confounder=name,
                )

        # ---- Rscript + backend + DIALOGUE package guards. ----
        backend, skip = self._resolve_rscript_backend(context)
        if skip is not None:
            return skip

        # ---- Export inputs + run DIALOGUE on the full data. ----
        scratch_root = Path(context.paths.scratch) / "multicellular_programs"
        full_scratch = scratch_root / "full"
        out_dir = full_scratch / "dialogue" / "out"

        confounders_arg = ",".join(confounders) if confounders else "NA"
        export = export_dialogue_inputs(
            adata,
            cell_type_col=cell_type_col,
            sample_col=sample_col,
            use_rep=use_rep,
            n_pcs=n_pcs,
            layer=layer,
            quality_col=quality_col,
            condition_col=condition_col,
            confounders=confounders,
            min_cells_per_type=min_cells_per_type,
            scratch=full_scratch,
        )
        cell_types_used = list(export["cell_types"].values())  # ORIGINAL labels

        args = [
            str(full_scratch),
            str(out_dir),
            str(n_programs),
            str(n_program_genes),
            str(seed),
            (condition_col or "NA"),
            str(min_cells_per_type),
            confounders_arg,
        ]
        try:
            proc = backend.run_script(_DIALOGUE_R, args, timeout=timeout)
        except FileNotFoundError as exc:
            return self._skip("R execution failed", error=str(exc)[:500])
        except subprocess.TimeoutExpired as exc:
            return self._skip(f"R timed out after {timeout}s", error=str(exc)[:500])
        if proc.returncode != 0:
            return self._skip("dialogue.R failed", stderr=(proc.stderr or "").strip()[:500])

        outputs = read_dialogue_outputs(out_dir)
        programs = outputs["programs"]
        scores = outputs["scores"]
        associations = outputs["associations"]

        dialogue_version = None
        run_meta_path = out_dir / "run_meta.json"
        if run_meta_path.is_file():
            try:
                dialogue_version = json.loads(run_meta_path.read_text()).get("dialogue_version")
            except Exception:
                dialogue_version = None

        notes: list[str] = []

        # ---- Donor-support diagnostic (always emitted). ----
        donor_support_df = pd.DataFrame(
            columns=["program", "n_donors", "donor_fraction", "supported"]
        )
        per_program_donor_count: dict[str, int] = {}
        if donor_col and donor_col in adata.obs.columns and not scores.empty:
            donor_map = {
                str(cid): str(donor)
                for cid, donor in zip(
                    adata.obs_names, adata.obs[donor_col].astype(str), strict=False
                )
            }
            donor_support_df = donor_support(
                scores[["cell_id", "program", "score"]], donor_map, donor_support_min
            )
            per_program_donor_count = {
                str(p): int(n)
                for p, n in zip(
                    donor_support_df["program"], donor_support_df["n_donors"], strict=False
                )
            }
        elif not donor_col or donor_col not in adata.obs.columns:
            notes.append("donor support skipped (no donor_col resolved)")

        # ---- Program-stability diagnostic (guarded resample orchestration). ----
        per_program_stability: dict[str, float] = {}
        program_stability_df: pd.DataFrame | None = None
        if stability_resamples > 0 and not programs.empty:
            per_program_stability, program_stability_df, stability_notes = (
                self._orchestrate_stability(
                    adata=adata,
                    full_programs=programs,
                    backend=backend,
                    scratch_root=scratch_root,
                    sample_col=sample_col,
                    cell_type_col=cell_type_col,
                    use_rep=use_rep,
                    n_pcs=n_pcs,
                    layer=layer,
                    quality_col=quality_col,
                    condition_col=condition_col,
                    confounders=confounders,
                    confounders_arg=confounders_arg,
                    min_cells_per_type=min_cells_per_type,
                    n_programs=n_programs,
                    n_program_genes=n_program_genes,
                    n_samples=n_samples,
                    stability_resamples=stability_resamples,
                    seed=seed,
                    timeout=timeout,
                )
            )
            notes.extend(stability_notes)
        elif stability_resamples == 0:
            notes.append("program stability disabled (stability_resamples=0)")

        # ---- Write artifacts into results/multicellular_programs/. ----
        results_dir = Path(context.paths.results) / "multicellular_programs"
        results_dir.mkdir(parents=True, exist_ok=True)

        programs_csv = results_dir / "mcp_gene_programs.csv"
        scores_csv = results_dir / "mcp_scores.csv"
        donor_csv = results_dir / "program_donor_support.csv"
        programs.to_csv(programs_csv, index=False)
        scores.to_csv(scores_csv, index=False)
        donor_support_df.to_csv(donor_csv, index=False)

        artifacts = [
            StageArtifact(
                name="mcp_gene_programs",
                path=programs_csv,
                kind="csv",
                description="DIALOGUE multicellular-program gene loadings (per cell type).",
            ),
            StageArtifact(
                name="mcp_scores",
                path=scores_csv,
                kind="csv",
                description="Per-cell DIALOGUE program scores (long form).",
            ),
            StageArtifact(
                name="program_donor_support",
                path=donor_csv,
                kind="csv",
                description="Distinct donors supporting each multicellular program.",
            ),
        ]

        # Associations only when a phenotype/condition column drove the HLM test.
        if condition_col:
            assoc_csv = results_dir / "mcp_associations.csv"
            associations.to_csv(assoc_csv, index=False)
            artifacts.append(
                StageArtifact(
                    name="mcp_associations",
                    path=assoc_csv,
                    kind="csv",
                    description="Program-phenotype associations (HLM z / BH-adjusted p).",
                )
            )

        # Stability CSV only when resamples were requested and at least one succeeded.
        if stability_resamples > 0 and program_stability_df is not None:
            stability_csv = results_dir / "program_stability.csv"
            program_stability_df.to_csv(stability_csv, index=False)
            artifacts.append(
                StageArtifact(
                    name="program_stability",
                    path=stability_csv,
                    kind="csv",
                    description="Mean subsample loading-correlation stability per program.",
                )
            )

        n_programs_recovered = int(programs["program"].nunique()) if not programs.empty else 0
        notes.insert(
            0,
            f"DIALOGUE: {n_programs_recovered} program(s) across "
            f"{len(cell_types_used)} cell type(s).",
        )

        metrics = {
            "n_programs": n_programs_recovered,
            "n_cell_types_used": len(cell_types_used),
            "cell_types_used": cell_types_used,
            "n_samples": n_samples,
            "n_cells": int(adata.n_obs),
            "per_program_stability": per_program_stability,
            "per_program_donor_count": per_program_donor_count,
            "dialogue_version": dialogue_version,
            "seed": seed,
        }

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            metrics=metrics,
            backend="rscript",
        )

    def _orchestrate_stability(
        self,
        *,
        adata: ad.AnnData,
        full_programs: pd.DataFrame,
        backend: object,
        scratch_root: Path,
        sample_col: str,
        cell_type_col: str,
        use_rep: str,
        n_pcs: int,
        layer: str | None,
        quality_col: str | None,
        condition_col: str | None,
        confounders: list[str],
        confounders_arg: str,
        min_cells_per_type: int,
        n_programs: int,
        n_program_genes: int,
        n_samples: int,
        stability_resamples: int,
        seed: int,
        timeout: int,
    ) -> tuple[dict[str, float], pd.DataFrame | None, list[str]]:
        """Run guarded subsample re-runs and score program stability.

        Each round drops ~1/min(6, n_samples) of the samples (deterministically
        seeded by ``seed + round``), re-exports, re-runs DIALOGUE, and matches the
        round's programs back to the full run. Any per-round failure is recorded
        as a note and skipped -- stability never raises.
        """
        notes: list[str] = []
        all_samples = adata.obs[sample_col].astype(str).unique().tolist()
        drop_denom = min(6, n_samples) if n_samples > 0 else 1
        n_drop = max(1, n_samples // drop_denom)
        # Always keep at least two samples so a round can still fit a design.
        n_drop = min(n_drop, max(0, len(all_samples) - 2))

        match_dicts: list[dict[str, float]] = []
        for r in range(stability_resamples):
            try:
                if n_drop <= 0:
                    notes.append(f"stability resample {r} skipped (too few samples to drop)")
                    continue
                rng = np.random.default_rng(seed + r)
                dropped = set(rng.choice(all_samples, size=n_drop, replace=False).tolist())
                keep_mask = ~adata.obs[sample_col].astype(str).isin(dropped)
                sub = adata[keep_mask.to_numpy()].copy()

                round_scratch = scratch_root / f"resample_{r}"
                round_out = round_scratch / "dialogue" / "out"
                export_dialogue_inputs(
                    sub,
                    cell_type_col=cell_type_col,
                    sample_col=sample_col,
                    use_rep=use_rep,
                    n_pcs=n_pcs,
                    layer=layer,
                    quality_col=quality_col,
                    condition_col=condition_col,
                    confounders=confounders,
                    min_cells_per_type=min_cells_per_type,
                    scratch=round_scratch,
                )
                round_args = [
                    str(round_scratch),
                    str(round_out),
                    str(n_programs),
                    str(n_program_genes),
                    str(seed + r),
                    (condition_col or "NA"),
                    str(min_cells_per_type),
                    confounders_arg,
                ]
                rproc = backend.run_script(_DIALOGUE_R, round_args, timeout=timeout)
                if rproc.returncode != 0:
                    notes.append(f"stability resample {r} failed (nonzero R exit)")
                    continue
                round_programs = read_dialogue_outputs(round_out)["programs"]
                match_dicts.append(match_program_loadings(full_programs, round_programs))
            except Exception as exc:  # never let a resample crash the stage
                notes.append(f"stability resample {r} failed: {str(exc)[:200]}")
                continue

        if not match_dicts:
            notes.append("program stability: all resamples failed")
            return {}, None, notes

        stability_df = program_stability(match_dicts)
        per_program = {
            str(p): float(s)
            for p, s in zip(stability_df["program"], stability_df["mean_stability"], strict=False)
        }
        return per_program, stability_df, notes


__all__ = ["MulticellularProgramsMethod"]
