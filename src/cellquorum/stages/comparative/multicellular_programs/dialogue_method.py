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

from cellquorum.backends.script_paths import r_script_path
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod
from cellquorum.stages.comparative.multicellular_programs._dialogue_io import (
    export_dialogue_inputs,
    read_dialogue_outputs,
)
from cellquorum.stages.comparative.multicellular_programs.diagnostics import (
    donor_support,
    match_program_loadings,
    program_stability,
)
from cellquorum.stages.comparative.multicellular_programs.mcp_figures import plot_mcp_summary
from cellquorum.visualization.figstyle import render_figure

_DIALOGUE_R = r_script_path("dialogue.R")


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
        # ---- Config resolution. The cohort overlay (applied by the stage) only
        # maps batch/sample/donor/condition *_key names, not this stage's *_col
        # identity fields, so those are read straight from the per-stage config
        # block; when unset they fall through to the loud skips below. ----
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
        n_rep_dims = int(adata.obsm[use_rep].shape[1])
        if n_rep_dims < n_pcs:
            # Slicing [:, :n_pcs] would silently under-return, then the export's
            # PC-named DataFrame ctor raises -- give the loud, specific reason here.
            return self._skip(
                f"use_rep '{use_rep}' has {n_rep_dims} dim(s), fewer than requested n_pcs={n_pcs}",
                n_rep_dims=n_rep_dims,
                n_pcs=n_pcs,
            )

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

        # condition_col / quality_col drive obs reads in export_dialogue_inputs; an
        # absent column would otherwise raise KeyError out of _run -- skip loudly.
        if condition_col and condition_col not in adata.obs.columns:
            return self._skip(f"condition_col '{condition_col}' absent from obs")
        if quality_col and quality_col not in adata.obs.columns:
            return self._skip(f"quality_col '{quality_col}' absent from obs")

        # ---- Rscript + backend + DIALOGUE package guards. ----
        backend, skip = self._resolve_rscript_backend(context)
        if skip is not None:
            return skip

        # ---- Export inputs + run DIALOGUE on the full data. ----
        scratch_root = Path(context.paths.scratch) / "multicellular_programs"
        full_scratch = scratch_root / "full"
        out_dir = full_scratch / "dialogue" / "out"

        confounders_arg = ",".join(confounders) if confounders else "NA"
        # Belt-and-suspenders for any other latent obs/obsm raise (the explicit
        # guards above give the loud, testable reasons for the common cases).
        try:
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
        except Exception as exc:
            return self._skip("dialogue input export failed", error=str(exc)[:500])
        cell_types_used = list(export["cell_types"].values())  # ORIGINAL labels

        args = [
            str(full_scratch),
            str(out_dir),
            str(n_programs),
            str(n_program_genes),
            str(seed),
            # Export writes condition_col as "pheno" in meta.csv, so tell R "pheno".
            ("pheno" if condition_col else "NA"),
            # 7th argv maps to DIALOGUE's abn.c abundance cutoff in dialogue.R;
            # the min_cells_per_type -> abn.c name drift is intentional/spec'd.
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

        try:
            outputs = read_dialogue_outputs(out_dir)
        except Exception as exc:
            return self._skip("dialogue.R output unreadable", error=str(exc)[:500])
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
        warnings: list[str] = []

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
            # A warning, because donor support is the check that says whether a
            # program is real or one donor's idiosyncrasy. Without it the programs
            # ship with nothing to distinguish those two cases.
            warnings.append("donor support skipped (no donor_col resolved)")

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
                    warnings=warnings,
                )
            )
            notes.extend(stability_notes)
        elif stability_resamples == 0:
            notes.append("program stability disabled (stability_resamples=0)")

        # ---- Write artifacts into results/multicellular_programs/. ----
        results_dir = Path(context.paths.results) / "multicellular_programs"
        results_dir.mkdir(parents=True, exist_ok=True)
        writer = StageArtifactWriter.from_context(context, default_subdir="multicellular_programs")

        artifacts = [
            writer.table(
                programs,
                "mcp_gene_programs.csv",
                name="mcp_gene_programs",
                description="DIALOGUE multicellular-program gene loadings (per cell type).",
                index=False,
            ),
            writer.table(
                scores,
                "mcp_scores.csv",
                name="mcp_scores",
                description="Per-cell DIALOGUE program scores (long form).",
                index=False,
            ),
            writer.table(
                donor_support_df,
                "program_donor_support.csv",
                name="program_donor_support",
                description="Distinct donors supporting each multicellular program.",
                index=False,
            ),
        ]

        # Associations only when a phenotype/condition column drove the HLM test.
        if condition_col:
            artifacts.append(
                writer.table(
                    associations,
                    "mcp_associations.csv",
                    name="mcp_associations",
                    description="Program-phenotype associations (HLM z / BH-adjusted p).",
                    index=False,
                )
            )

        # Stability CSV only when resamples were requested and at least one succeeded.
        if stability_resamples > 0 and program_stability_df is not None:
            artifacts.append(
                writer.table(
                    program_stability_df,
                    "program_stability.csv",
                    name="program_stability",
                    description="Mean subsample loading-correlation stability per program.",
                    index=False,
                )
            )

        # Summary figure (skip-not-crash: plotting failure must not crash method).
        if not programs.empty:
            figs: list[Path] = []
            render_figure(
                "MCP summary",
                lambda: plot_mcp_summary(
                    programs,
                    scores,
                    donor_support_df,
                    cell_type_col_values=cell_types_used,
                    out_dir=results_dir,
                    name="mcp_summary",
                ),
                figures=figs,
                warnings=warnings,
            )
            artifacts.extend(
                StageArtifact(
                    name="mcp_summary",
                    path=fig_path,
                    kind="figure",
                    description="MCP summary: participation heatmap + score distributions.",
                )
                for fig_path in figs
            )

        n_programs_recovered = int(programs["program"].nunique()) if not programs.empty else 0
        notes.insert(
            0,
            f"DIALOGUE: {n_programs_recovered} program(s) across "
            f"{len(cell_types_used)} cell type(s).",
        )

        metrics = {
            "n_programs": n_programs_recovered,
            "n_programs_requested": n_programs,
            "n_cell_types_used": len(cell_types_used),
            "cell_types_used": cell_types_used,
            "n_samples": n_samples,
            "n_cells": int(adata.n_obs),
            # Provenance: which expression source + representation DIALOGUE saw.
            "layer": layer,
            "use_rep": use_rep,
            "n_pcs": n_pcs,
            "per_program_stability": per_program_stability,
            "per_program_donor_count": per_program_donor_count,
            "dialogue_version": dialogue_version,
            "seed": seed,
        }

        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            warnings=warnings,
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
        warnings: list[str],
    ) -> tuple[dict[str, float], pd.DataFrame | None, list[str]]:
        """Run guarded subsample re-runs and score program stability.

        Each round drops ~1/min(6, n_samples) of the samples (deterministically
        seeded by ``seed + round``), re-exports, re-runs DIALOGUE, and matches the
        round's programs back to the full run. Stability never raises: a round that
        fails is skipped.

        A skipped round goes to ``warnings`` and not to the returned notes, because
        stability is averaged over the rounds that survived -- so a run reporting
        0.9 over two of ten rounds and one reporting 0.9 over all ten look
        identical in the metrics, and only the warning distinguishes them.
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
                    ("pheno" if condition_col else "NA"),
                    str(min_cells_per_type),
                    confounders_arg,
                ]
                rproc = backend.run_script(_DIALOGUE_R, round_args, timeout=timeout)
                if rproc.returncode != 0:
                    warnings.append(f"stability resample {r} failed (nonzero R exit)")
                    continue
                round_programs = read_dialogue_outputs(round_out)["programs"]
                match_dicts.append(match_program_loadings(full_programs, round_programs))
            except Exception as exc:  # never let a resample crash the stage
                warnings.append(f"stability resample {r} failed: {str(exc)[:200]}")
                continue

        if not match_dicts:
            warnings.append("program stability: all resamples failed")
            return {}, None, notes

        stability_df = program_stability(match_dicts)
        per_program = {
            str(p): float(s)
            for p, s in zip(stability_df["program"], stability_df["mean_stability"], strict=False)
        }
        return per_program, stability_df, notes


__all__ = ["MulticellularProgramsMethod"]
