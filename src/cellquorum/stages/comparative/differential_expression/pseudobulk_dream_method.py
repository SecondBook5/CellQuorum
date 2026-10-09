"""Specimen-level pseudobulk mixed-model DE (``variancePartition::dream``).

For designs where biological replicates are *specimens* nested in *subjects* -- a
between-subject factor crossed with a within-subject factor, e.g.
``~ genotype * tissue + (1 | mouse)`` -- a fixed subject term would alias the
between-subject factor, so the subject enters as a random intercept instead.

Per cell type the workflow is::

    full specimen universe (built once from the FULL object, before any subsetting)
      -> cell-type subset -> specimen-level pseudobulk counts
      -> support audit against the universe (absent / below-threshold specimens out)
      -> factorial design re-analysis on the INCLUDED specimens -> support gates
      -> contrast matrix (from design rows)
      -> edgeR filterByExpr + TMM -> voomWithDreamWeights -> dream (all contrasts, one fit)
      -> eBayes -> one result table per contrast

Two failure classes are kept strictly apart. A malformed design, an invalid contrast,
corrupt specimen metadata or a missing source column is **fatal** for the whole run. A cell
type whose remaining biological support cannot estimate the requested model is a recorded
**skip** for that cell type only -- the model is never silently simplified.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd
from pydantic import TypeAdapter

import cellquorum
from cellquorum.backends.script_paths import r_script_path
from cellquorum.config.design import FactorialContrast, FactorialDesignConfig
from cellquorum.config.factorial_contrasts import (
    ContrastNotEstimable,
    FactorialDesignEncoder,
    build_factorial_contrasts,
)
from cellquorum.config.factorial_design import (
    analyze_factorial_design,
    required_specimen_columns,
    resolve_factorial_columns,
    validate_factorial_design,
)
from cellquorum.core.context import resolve_n_jobs
from cellquorum.core.contracts import DataContract
from cellquorum.core.exceptions import CellQuorumConfigError
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod
from cellquorum.stages.comparative.differential_expression.config import PseudobulkSupportConfig
from cellquorum.stages.comparative.differential_expression.dream_design import (
    build_machine_design,
    human_formula,
    support_gate_failures,
    unique_slugs,
)
from cellquorum.stages.comparative.differential_expression.pseudobulk import (
    PseudobulkResult,
    aggregate_pseudobulk,
    build_specimen_universe,
    evaluate_pseudobulk_support,
    restrict_to_included,
)
from cellquorum.stages.qc.eligibility import de_eligible_cells

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

# Path to the bundled dream script.
_DREAM_R = r_script_path("dream.R")

# Run-relative output root: results/differential_expression/pseudobulk_dream/...
_OUTPUT_SUBDIR = "differential_expression/pseudobulk_dream"

# Label for the single population analysed when no cell_type_col is configured.
_ALL_CELLS = "all_cells"

_CONTRAST_ADAPTER: TypeAdapter[list[FactorialContrast]] = TypeAdapter(list[FactorialContrast])


class PseudobulkDreamMethod(RAnalysisMethod):
    """Specimen-level mixed-model pseudobulk DE via ``variancePartition::dream``.

    Opt-in only (``differential_expression.method: pseudobulk_dream``); never selected
    automatically. Requires ``design.factorial`` and ``contrasts.factorial``. With no
    random effect (a single-tissue between-subject comparison, one specimen per subject)
    the fit is an ordinary fixed-effect voom-limma model; repeated specimens per subject
    without a random effect are refused.
    """

    name = "pseudobulk_dream"
    stage_category = "differential_expression"
    r_package = "variancePartition"

    # The method owns cell-type stratification: the specimen universe must be built from
    # the FULL object, so the stage must not pre-slice the AnnData per cell type.
    stratifies_internally = True

    def input_contract(self, config: dict) -> DataContract:
        """Require the counts layer plus every obs column the declared design reads."""

        layer = config.get("layer", "counts")
        required: list[str] = []
        raw = config.get("factorial")
        if raw is not None:
            design = FactorialDesignConfig.model_validate(raw)
            sample_col, subject_col = self._columns(design, config)
            required = [
                sample_col,
                *required_specimen_columns(design, sample_col=sample_col, subject_col=subject_col),
            ]
        cell_type_col = config.get("cell_type_col")
        if cell_type_col:
            required.append(cell_type_col)
        return DataContract(
            required_layers=[layer],
            required_obs=list(dict.fromkeys(required)),
            expression_layer=layer,
            expected_kind="counts",
        )

    @staticmethod
    def _columns(
        design: FactorialDesignConfig, config: dict, context: PipelineContext | None = None
    ) -> tuple[str, str | None]:
        """Return the specimen and subject columns.

        The stage resolves them once (design value, else cohort key) into
        ``config['sample_col']`` / ``config['subject_col']``. When called without that
        bridge, they are resolved here from the run configuration.
        """
        if config.get("sample_col"):
            return str(config["sample_col"]), config.get("subject_col") or None
        resolved = resolve_factorial_columns(getattr(context, "config", None), design)
        return resolved.sample_col, resolved.subject_col

    # ------------------------------------------------------------------ run

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        """Validate globally, then fit one mixed model per cell type."""

        # ---- 1. Config + global (fatal) validation; nothing below is a skip. -----
        plan = self._plan(adata, config, context)

        # ---- 2. Backend guard: only now, so config errors surface without R. -----
        backend = self._resolve_rscript_backend(context, config)
        if isinstance(backend, MethodSkip):
            return backend
        if not backend._r_package_available("edgeR"):
            return self._skip("edgeR R package unavailable", r_package="edgeR")

        writer = StageArtifactWriter.from_context(context)
        artifacts: list[StageArtifact] = []
        self._reset_output_dir(context)
        artifacts.append(
            writer.json(
                {
                    "started_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "cellquorum_version": cellquorum.__version__,
                    "populations": [str(p) for p in plan["populations"]],
                    "n_jobs": plan["n_jobs"],
                },
                "invocation.json",
                name="dream_invocation",
                description="This invocation's start time and populations; every file beside "
                "it was written by this run.",
                subdir=_OUTPUT_SUBDIR,
            )
        )

        artifacts.append(
            writer.json(
                plan["global_provenance"],
                "design_report.json",
                name="dream_design_report",
                description="Global factorial design report over the full specimen universe.",
                subdir=_OUTPUT_SUBDIR,
            )
        )
        artifacts.append(
            writer.json(
                plan["contrast_definitions"],
                "contrast_definitions.json",
                name="dream_contrast_definitions",
                description="Structured factorial contrast definitions and design-row weights.",
                subdir=_OUTPUT_SUBDIR,
            )
        )

        # ---- 3. One fit per cell type. -----------------------------------------
        populations = plan["populations"]
        slugs = unique_slugs([str(p) for p in populations])
        summary: list[dict[str, Any]] = []
        combined: list[pd.DataFrame] = []
        warnings: list[str] = []
        for population in populations:
            label = str(population)
            outcome = self._fit_population(
                adata, plan, config, context, backend, writer, label, slugs[label]
            )
            summary.append(outcome["summary"])
            artifacts.extend(outcome["artifacts"])
            if outcome["table"] is not None:
                combined.append(outcome["table"])
            if outcome["summary"]["status"] != "fitted":
                warnings.append(
                    f"pseudobulk_dream: {label} {outcome['summary']['status']}: "
                    f"{outcome['summary']['reason']}"
                )

        summary_df = pd.DataFrame(summary)
        artifacts.append(
            writer.table(
                summary_df,
                "cell_type_summary.csv",
                name="dream_cell_type_summary",
                description="Per-cell-type outcome (fitted / skipped / failed) and the reason.",
                subdir=_OUTPUT_SUBDIR,
            )
        )
        if not combined:
            return self._skip(
                "no cell type produced pseudobulk_dream results",
                outcomes=summary_df.to_dict(orient="records"),
            )

        all_df = pd.concat(combined, ignore_index=True)
        artifacts.append(
            writer.table(
                all_df,
                "de_pseudobulk_dream_all.csv",
                name="de_results",
                description="Pseudobulk dream DE: every cell type x contrast (long format).",
                subdir=_OUTPUT_SUBDIR,
            )
        )
        n_fit = int((summary_df["status"] == "fitted").sum())
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[
                f"Pseudobulk dream DE: {plan['human_formula']}; {n_fit}/{len(populations)} "
                f"population(s) fitted; contrasts: {plan['contrast_names']}.",
            ],
            warnings=warnings,
            metrics={
                "human_formula": plan["human_formula"],
                "contrasts": plan["contrast_names"],
                "n_populations": len(populations),
                "n_fitted": n_fit,
                "ddf": plan["ddf"],
            },
            backend="rscript",
        )

    # ------------------------------------------------------------ output hygiene

    @staticmethod
    def _reset_output_dir(context: PipelineContext) -> Path:
        """Create this method's output directory empty, discarding a previous invocation's.

        A rerun may analyse different cell types than the last one; without this, an old
        cell type's folder would sit beside the new results looking current. Only the
        method-owned directory ``results/differential_expression/pseudobulk_dream`` is
        removed, and only after confirming it is a real directory (not a symlink) that
        resolves inside the run's results directory. Nothing else is touched.

        Raises:
            CellQuorumConfigError: If the directory is a symlink or resolves outside the
                run's results directory.
        """

        results_root = Path(context.paths.results).resolve()
        target = results_root / _OUTPUT_SUBDIR
        if target.is_symlink():
            raise CellQuorumConfigError(
                f"Refusing to clear '{target}': a symbolic link is not a method-owned directory."
            )
        resolved = target.resolve()
        if resolved.name != "pseudobulk_dream" or results_root not in resolved.parents:
            raise CellQuorumConfigError(
                f"Refusing to clear '{resolved}': not inside the run's results directory."
            )
        if resolved.exists():
            shutil.rmtree(resolved)
        resolved.mkdir(parents=True)
        return resolved

    # ------------------------------------------------------------------ plan

    def _plan(self, adata: ad.AnnData, config: dict, context: PipelineContext) -> dict[str, Any]:
        """Validate configuration and the global design; build the specimen universe.

        Everything raised here is fatal by design.

        Raises:
            CellQuorumConfigError: Unsupported options, a malformed or globally invalid
                design, or an invalid / inestimable contrast.
            CellQuorumDataError: Corrupt specimen metadata.
        """

        raw_design = config.get("factorial")
        raw_contrasts = config.get("factorial_contrasts")
        if raw_design is None:
            raise CellQuorumConfigError(
                "pseudobulk_dream needs a specimen-level design: declare design.factorial."
            )
        if not raw_contrasts:
            raise CellQuorumConfigError(
                "pseudobulk_dream needs at least one structured contrast: declare "
                "contrasts.factorial."
            )
        # Options of the other DE path that would be silently ignored here are refused.
        if config.get("covariates"):
            raise CellQuorumConfigError(
                "pseudobulk_dream does not support differential_expression.covariates yet "
                "(factors must be categorical and declared in design.factorial.factors); "
                "refusing to ignore them."
            )
        if config.get("interactions"):
            raise CellQuorumConfigError(
                "pseudobulk_dream takes interactions from design.factorial.interactions, not "
                "differential_expression.interactions."
            )
        if (config.get("ambient_filter") or {}).get("enabled"):
            raise CellQuorumConfigError(
                "differential_expression.ambient_filter is not supported by pseudobulk_dream; "
                "refusing to ignore it."
            )

        design = FactorialDesignConfig.model_validate(raw_design)
        contrasts = _CONTRAST_ADAPTER.validate_python(list(raw_contrasts))
        gates = PseudobulkSupportConfig.model_validate(config.get("pseudobulk") or {})
        ddf = (config.get("dream") or {}).get("ddf", "adaptive")

        sample_col, subject_col = self._columns(design, config, context)

        # Dependence among repeated specimens must be modelled, not merely described.
        within = [n for n, s in design.factors.items() if s.scope == "within_subject"]
        if within and (subject_col is None or subject_col not in design.random_effects):
            raise CellQuorumConfigError(
                f"Within-subject factor(s) {within} are declared but the subject "
                f"('{subject_col}') is not a random effect, so repeated specimens would be "
                "treated as independent. Add the subject to design.factorial.random_effects."
            )

        columns = required_specimen_columns(design, sample_col=sample_col, subject_col=subject_col)
        universe = build_specimen_universe(adata.obs, sample_col=sample_col, metadata_cols=columns)
        specimens = universe.drop(columns="n_cells_total")

        # A design with no random effect is an ordinary fixed-effect fit (dream reduces to
        # limma-voom with residual degrees of freedom). It is valid only when specimens are
        # independent, i.e. every subject contributes exactly one specimen to the universe
        # being analysed (e.g. one tissue of a between-subject comparison). A subject with
        # several specimens and no random intercept would be pseudoreplication.
        if not design.random_effects:
            if subject_col is None or subject_col not in specimens.columns:
                raise CellQuorumConfigError(
                    "pseudobulk_dream without a random effect needs the subject column "
                    "(design.factorial.subject_col or cohort.donor_key) to verify that every "
                    "subject contributes a single specimen."
                )
            repeated = specimens.groupby(subject_col, observed=True).size()
            repeated = repeated[repeated > 1]
            if len(repeated):
                raise CellQuorumConfigError(
                    "pseudobulk_dream without a random effect treats specimens as independent, "
                    f"but subject(s) {sorted(map(str, repeated.index))} contribute several "
                    f"specimens. Add '{subject_col}' to design.factorial.random_effects, or "
                    "restrict the object to one specimen per subject."
                )
        report = validate_factorial_design(
            specimens, design=design, sample_col=sample_col, subject_col=subject_col
        )
        global_contrasts = build_factorial_contrasts(
            contrasts, design=design, sample_meta=specimens
        )

        # DE-eligible cells: the QC fit AND inference permissions for DE. The specimen
        # universe above was built from EVERY cell, so a specimen whose cells are all
        # QC-excluded stays visible; eligibility only decides whose counts enter a profile.
        # QC state present but its DE masks missing is fatal (never a silent fall-back).
        eligible = de_eligible_cells(adata.obs)
        n_total = int(adata.n_obs)
        n_eligible = n_total if eligible is None else int(eligible.sum())
        eligibility: dict[str, Any] = {
            "rule": (
                "qc_fit_de AND qc_inference_de"
                if eligible is not None
                else "none: no QC columns present, every cell eligible"
            ),
            "n_cells_total": n_total,
            "n_cells_eligible": n_eligible,
            "n_cells_excluded": n_total - n_eligible,
        }

        cell_type_col = config.get("cell_type_col")
        if cell_type_col and cell_type_col in adata.obs.columns:
            populations: list[Any] = sorted(
                str(c) for c in adata.obs[cell_type_col].dropna().unique()
            )
        else:
            cell_type_col = None
            populations = [_ALL_CELLS]

        return {
            "design": design,
            "contrasts": contrasts,
            "gates": gates,
            "ddf": ddf,
            "sample_col": sample_col,
            "subject_col": subject_col,
            "universe": universe,
            "metadata_cols": columns,
            "cell_type_col": cell_type_col,
            "populations": populations,
            "layer": config.get("layer", "counts"),
            "n_jobs": resolve_n_jobs(context),
            "eligible": None if eligible is None else eligible.to_numpy(),
            "eligibility": eligibility,
            "human_formula": human_formula(design),
            "contrast_names": global_contrasts.names,
            "global_provenance": {
                "scope": "full specimen universe (before cell-type support filtering)",
                "human_formula": human_formula(design),
                "de_eligibility": eligibility,
                "design_report": report.to_dict(),
            },
            "contrast_definitions": global_contrasts.to_dict(),
        }

    # ------------------------------------------------------------- one cell type

    def _fit_population(
        self,
        adata: ad.AnnData,
        plan: dict[str, Any],
        config: dict,
        context: PipelineContext,
        backend: Any,
        writer: StageArtifactWriter,
        label: str,
        slug: str,
    ) -> dict[str, Any]:
        """Aggregate, audit, validate and fit one population.

        Returns a dict with ``summary`` (always), ``artifacts`` and ``table`` (the
        long-format result, or ``None`` when skipped / failed). Corrupt metadata
        propagates as a fatal exception; only support loss becomes a skip.
        """

        design: FactorialDesignConfig = plan["design"]
        sample_col, subject_col = plan["sample_col"], plan["subject_col"]
        universe: pd.DataFrame = plan["universe"]
        subdir = f"{_OUTPUT_SUBDIR}/{slug}"
        artifacts: list[StageArtifact] = []

        if plan["cell_type_col"]:
            in_population = (adata.obs[plan["cell_type_col"]].astype(str) == label).to_numpy()
        else:
            in_population = np.ones(adata.n_obs, dtype=bool)
        eligible_mask = plan["eligible"]
        contributing = in_population if eligible_mask is None else in_population & eligible_mask
        subset = adata[contributing]
        observed_n_cells = (
            adata.obs.loc[in_population, sample_col].astype(str).value_counts().astype(np.int64)
        )
        row: dict[str, Any] = {
            "cell_type": label,
            "slug": slug,
            "status": "skipped",
            "reason": "",
            "n_cells_observed": int(in_population.sum()),
            "n_cells": int(subset.n_obs),
            "n_cells_qc_excluded": int(in_population.sum() - subset.n_obs),
            "n_specimens_universe": int(len(universe)),
        }

        def skip(reason: str) -> dict[str, Any]:
            row["reason"] = reason
            return {"summary": row, "artifacts": artifacts, "table": None}

        # Observed pseudobulk of THIS population's DE-eligible cells (declared metadata must
        # be constant). A population with no eligible cell at all yields an empty profile set;
        # the support audit below still reports every specimen and why it is out.
        if subset.n_obs:
            pb = aggregate_pseudobulk(
                subset,
                layer=plan["layer"],
                sample_col=sample_col,
                metadata_cols=plan["metadata_cols"],
            )
        else:
            pb = PseudobulkResult(
                counts=pd.DataFrame(columns=list(adata.var_names), dtype=np.int64),
                sample_meta=pd.DataFrame(columns=plan["metadata_cols"]),
                n_cells=pd.Series(dtype=np.int64),
            )
        row["n_specimens_observed"] = int(pb.counts.shape[0])

        # Support audit against the FULL universe: absent specimens stay visible.
        gates: PseudobulkSupportConfig = plan["gates"]
        support = evaluate_pseudobulk_support(
            pb,
            sample_metadata=universe,
            subject_col=subject_col,
            factor_cols=[c for c in plan["metadata_cols"] if c != subject_col],
            min_cells_per_sample=gates.min_cells_per_sample,
            observed_n_cells=observed_n_cells,
        )
        artifacts.append(
            writer.table(
                support.table,
                "support.csv",
                name=f"dream_support:{slug}",
                description=f"Specimen support for {label}: all specimens, with exclusion reasons.",
                subdir=subdir,
            )
        )
        included = restrict_to_included(pb, support)
        row["n_specimens_included"] = support.n_included
        specimens = universe.loc[support.included_ids].drop(columns="n_cells_total")

        if support.n_included < 2:
            return skip(f"fewer than 2 included specimens ({support.n_included})")

        # Cell-type-specific design re-analysis on the included specimens.
        report = analyze_factorial_design(
            specimens, design=design, sample_col=sample_col, subject_col=subject_col
        )
        artifacts.append(
            writer.json(
                report.to_dict(),
                "design_report.json",
                name=f"dream_design_report:{slug}",
                description=f"Factorial design report for {label} on its included specimens.",
                subdir=subdir,
            )
        )
        if not report.ok:
            return skip(
                "requested model not estimable from remaining specimens: "
                + " | ".join(report.errors)
            )
        failures = support_gate_failures(report, design, gates)
        if failures:
            return skip("support gates failed: " + "; ".join(failures))

        try:
            contrast_set = build_factorial_contrasts(
                plan["contrasts"], design=design, sample_meta=specimens
            )
        except ContrastNotEstimable as exc:
            return skip(f"contrast not estimable from remaining specimens: {exc}")

        encoder = FactorialDesignEncoder.fit(specimens, design)
        machine = build_machine_design(
            encoder=encoder,
            specimens=specimens,
            design=design,
            contrasts=contrast_set,
            sample_col=sample_col,
        )

        # R inputs are kept with the results: the exact pseudobulk the model saw.
        cell_dir = Path(context.paths.results) / _OUTPUT_SUBDIR / slug
        out_dir = cell_dir / "r_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        counts_csv, meta_csv, l_csv = (
            cell_dir / "pseudobulk_counts.csv",
            cell_dir / "pseudobulk_meta.csv",
            cell_dir / "contrast_matrix_machine.csv",
        )
        included.counts.reset_index(names="sample").to_csv(counts_csv, index=False)
        machine.meta.to_csv(meta_csv)
        machine.L.rename_axis("coef").reset_index().to_csv(l_csv, index=False)
        for fname, desc in (
            ("pseudobulk_counts.csv", "specimen x gene pseudobulk counts the model saw"),
            ("pseudobulk_meta.csv", "numeric fixed-effect and random-effect columns sent to R"),
            ("contrast_matrix_machine.csv", "contrast matrix L with R coefficient names"),
        ):
            artifacts.append(
                writer.register(
                    name=f"dream_input:{slug}:{fname.removesuffix('.csv')}",
                    filename=fname,
                    kind="csv",
                    description=f"{label}: {desc}.",
                    subdir=subdir,
                )
            )
        human_matrix = contrast_set.matrix()
        artifacts.append(
            writer.table(
                human_matrix.reset_index(names="coefficient"),
                "contrast_matrix.csv",
                name=f"dream_contrast_matrix:{slug}",
                description=f"Contrast weights over the fixed-effect columns for {label}.",
                subdir=subdir,
            )
        )

        args = [
            str(counts_csv), str(meta_csv), str(l_csv), machine.formula, str(out_dir),
            str(plan["ddf"]), str(int(config.get("min_count", 10))),
            str(int(config.get("min_total_count", 15))), str(plan["n_jobs"]),
        ]  # fmt: skip
        try:
            proc = backend.run_script(
                _DREAM_R, args, timeout=int(config.get("timeout_seconds", 1800))
            )
        except FileNotFoundError as exc:
            row.update(status="failed", reason=f"R execution failed: {str(exc)[:500]}")
            return {"summary": row, "artifacts": artifacts, "table": None}
        if proc.returncode != 0:
            row.update(
                status="failed",
                reason="dream script failed: " + proc.stderr.strip()[-800:],
            )
            return {"summary": row, "artifacts": artifacts, "table": None}

        info = json.loads((out_dir / "fit_info.json").read_text())
        if info.get("status") != "ok":
            return skip(f"no model fitted: {info.get('status')}")

        long = pd.read_csv(out_dir / "results_long.csv")
        failed_csv = out_dir / "failed_genes.csv"
        if failed_csv.exists():
            artifacts.append(
                writer.register(
                    name=f"dream_failed_genes:{slug}",
                    filename="failed_genes.csv",
                    kind="csv",
                    description=f"Genes whose mixed model failed for {label}, with the error.",
                    subdir=f"{subdir}/r_output",
                )
            )

        # Attach biological meaning to the internal contrast ids and write one file each.
        csl = unique_slugs(contrast_set.names)
        frames: list[pd.DataFrame] = []
        n_samples, n_subjects = (
            len(specimens),
            int(specimens[subject_col].nunique()) if subject_col else None,
        )
        contrast_support: dict[str, Any] = {}
        for cid, cname in machine.contrast_ids.items():
            contrast = next(c for c in contrast_set.contrasts if c.name == cname)
            sup = contrast.support(specimens, subject_col)
            contrast_support[cname] = sup
            comp_rows: list[dict[str, Any]] = sup["components"]  # type: ignore[assignment]
            part = long[long["contrast"] == cid].drop(columns="contrast").copy()
            part.insert(0, "gene", part.pop("gene"))
            part.insert(1, "cell_type", label)
            part.insert(2, "contrast", cname)
            part.insert(3, "tested_effect", contrast.tested_effect)
            part["n_samples"] = n_samples
            part["n_subjects"] = n_subjects
            part["n_samples_in_contrast"] = sum(int(c["n_specimens"]) for c in comp_rows)
            part["formula"] = machine.human_formula
            part["backend"] = f"variancePartition::dream {info.get('variancePartition', '')}"
            part = part.sort_values("PValue", kind="stable").reset_index(drop=True)
            frames.append(part)
            artifacts.append(
                writer.table(
                    part,
                    f"{csl[cname]}.csv",
                    name=f"dream_de:{slug}:{csl[cname]}",
                    description=f"{label}: {contrast.tested_effect} ({machine.human_formula}).",
                    subdir=subdir,
                )
            )

        provenance = {
            "cellquorum_version": cellquorum.__version__,
            "cell_type": label,
            "counts_layer": plan["layer"],
            "sample_col": sample_col,
            "subject_col": subject_col,
            "n_cells_observed": row["n_cells_observed"],
            "n_cells": int(subset.n_obs),
            "n_cells_qc_excluded": row["n_cells_qc_excluded"],
            "de_eligibility": plan["eligibility"],
            "n_specimens_universe": int(len(universe)),
            "n_specimens_observed": int(pb.counts.shape[0]),
            "n_specimens_included": support.n_included,
            "excluded_specimens": support.table.loc[
                ~support.table["included_for_expression"], ["sample_id", "exclusion_reason"]
            ].to_dict(orient="records"),
            "subjects_included": sorted(map(str, specimens[subject_col].unique()))
            if subject_col
            else [],
            "n_cells_per_included_specimen": {
                str(k): int(v)
                for k, v in (
                    included.n_cells if included.n_cells is not None else pd.Series(dtype=int)
                ).items()
            },
            "support_gates": gates.model_dump(),
            "human_formula": machine.human_formula,
            "machine_formula": machine.formula,
            "fixed_effect_column_mapping": machine.fixed_columns,
            "random_effect_column_mapping": machine.random_columns,
            "factor_levels": {k: list(v) for k, v in encoder.levels.items()},
            "reference_levels": dict(encoder.references),
            "interactions": [list(i) for i in design.interactions],
            "random_effects": list(design.random_effects),
            "ddf_requested": plan["ddf"],
            "parallel": {"n_jobs": plan["n_jobs"], **(info.get("bpparam") or {})},
            "filter_design_columns": info.get("filter_design_columns"),
            "contrast_ids": machine.contrast_ids,
            "contrasts": contrast_set.to_dict(),
            "contrast_support": contrast_support,
            "r": info,
        }
        artifacts.append(
            writer.json(
                provenance,
                "model_provenance.json",
                name=f"dream_provenance:{slug}",
                description=f"Exact model, coding, support and R versions used for {label}.",
                subdir=subdir,
            )
        )
        row.update(
            status="fitted",
            reason="",
            n_genes_after_filter=info.get("n_genes_after_filter"),
            n_genes_failed=info.get("n_genes_failed", 0),
        )
        return {
            "summary": row,
            "artifacts": artifacts,
            "table": pd.concat(frames, ignore_index=True),
        }


__all__ = ["PseudobulkDreamMethod"]
