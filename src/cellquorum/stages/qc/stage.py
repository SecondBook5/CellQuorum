# Pipeline step (order=20): qc — metrics, thresholds, doublets, optional cell-cycle scoring.
"""QC pipeline stage for CellQuorum."""

from __future__ import annotations

# Import logging for loud, auditable QC decisions (no-silent-decisions rule).
import logging

# Import Mapping for dictionary-like config resolution.
# Import dataclass for the concrete stage object.
from dataclasses import dataclass
from typing import ClassVar

# Import Path for stage output directory handling.
# Import AnnData for stage input and output typing.
import anndata as ad

# Import numpy to compare a recomputed metric column against an inherited one.
import numpy as np

# Import pandas for AnnData obs/var decision annotation typing.
import pandas as pd

# Import shared CellQuorum data exception.
from cellquorum.core.context import resolve_n_jobs

# Import pipeline stage artifact and result contracts.
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage

# Import QC artifact writer utilities.
from cellquorum.stages.qc._annotate import (
    annotate_adata_with_qc_metrics,
    build_qc_figure_adata,
    build_qc_output_adata,
)
from cellquorum.stages.qc._context import (
    get_context_adata,
    get_qc_output_dir,
    is_qc_stage_enabled,
    resolve_qc_config,
)
from cellquorum.stages.qc._errors import QCStageError
from cellquorum.stages.qc._report import (
    _audit_to_columns,
    build_disabled_qc_stage_result,
    build_qc_stage_metrics,
    build_qc_stage_notes,
    build_qc_stage_summary_extra,
    build_stage_artifacts_from_manifest,
    collect_qc_stage_warnings,
    resolve_publication_qc_keys,
)
from cellquorum.stages.qc.artifacts import write_qc_artifacts

# Import the differential-attrition audit.
from cellquorum.stages.qc.attrition import audit_qc_design_leaks, audit_qc_stage_attrition

# Import QC configuration.
from cellquorum.stages.qc.config import QCConfig

# Import QC decision construction.
from cellquorum.stages.qc.floors import apply_floors, require_non_empty_qc_result

# Import QC metric calculation.
from cellquorum.stages.qc.lineage import (
    LINEAGE_COLUMN,
    NULL_LEVEL_COLUMN,
    PROVISIONAL_EMBEDDING,
    audit_lineages,
    provisional_lineages,
    resolve_null_groups,
)
from cellquorum.stages.qc.metrics import QCMetricsResult, calculate_qc_metrics
from cellquorum.stages.qc.mixture import MitoMixtureResult, fit_mito_mixture
from cellquorum.stages.qc.selfcheck import run_self_check

# Import QC threshold construction.
from cellquorum.stages.qc.validation import get_qc_matrix

logger = logging.getLogger(__name__)


def _condition_mixture_on_lineage(qc_config: QCConfig) -> QCConfig:
    """Fit the mitochondrial mixture within cell identity, not just within library.

    A constitutively high-mitochondrial cell type — mast cells, neutrophils, erythrocytes —
    receives a high compromised-probability on biology alone when the mixture is fitted per
    library, because the healthy component is defined by the library's ordinary cells. Fitting
    within (library x lineage) makes the posterior mean "compromised *for a cell like this*",
    which is the question QC intends and which needs no rescaling afterwards.

    The fallback chain matters as much as the grouping. A lineage too small to support its own
    mixture in one library must borrow a coarser model rather than go unmodelled, so the chain
    walks library-and-lineage, then library, then pooled. ``per_group`` resolution is required
    for this to mean anything: under ``uniform`` a single unfittable group would drag the entire
    dataset back to the coarsest level, which is precisely the behaviour being fixed.
    """
    mixture = qc_config.mito_mixture
    if not mixture.enabled:
        return qc_config

    grouping = [*mixture.groupby, LINEAGE_COLUMN]
    fallbacks = [list(mixture.groupby), *[list(level) for level in mixture.fallback_groupby]]
    if [] not in fallbacks:
        fallbacks.append([])

    return qc_config.model_copy(
        update={
            "mito_mixture": mixture.model_copy(
                update={
                    "groupby": grouping,
                    "fallback_groupby": fallbacks,
                    "level_policy": "per_group",
                }
            )
        }
    )


def _analysable_mask(
    input_adata: ad.AnnData,
    output_adata: ad.AnnData,
    cell_keep: pd.Series,
) -> pd.Series:
    """Cells that survived QC as analysable, indexed by every cell that ENTERED QC.

    "Analysable" spans both axes on which a cell can be lost, because auditing either one alone
    misreads the run:

    * the floors physically remove a barcode — on real data this is a small share of the loss;
    * graded adjudication quarantines a cell, which keeps it in the object but withdraws every
      permission that could contribute to a conclusion. That is a loss for any purpose the
      attrition audit cares about, and it is where nearly all of the loss now happens.

    Borderline is NOT counted as lost. A borderline cell is retained, projected, annotated, and
    contributes to composition through a sensitivity universe; calling it lost would report
    attrition that has not happened.

    Args:
        input_adata: The object as it entered QC, defining the index.
        output_adata: The post-floor object carrying the graded state column, if any.
        cell_keep: Per-barcode floor mask over the input index.

    Returns:
        Boolean Series over ``input_adata.obs_names``. Barcodes removed by a floor are False;
        a cell present in the output is False when quarantined and True otherwise.
    """
    analysable = cell_keep.reindex(input_adata.obs_names).fillna(False).astype(bool)

    state = output_adata.obs.get("qc_state_initial")
    if state is None:
        return analysable

    quarantined = (
        state.astype(str).eq("quarantine").reindex(input_adata.obs_names).fillna(False).astype(bool)
    )
    return analysable & ~quarantined


@register_stage(name="qc", order=20, config_flag="qc", config_field="qc")
@dataclass(frozen=True)
class QCStage:
    """
    Execute the complete CellQuorum QC module.

    The stage wires together the QC submodule layers:

    1. validate and calculate QC metrics
    2. build fixed and/or MAD thresholds
    3. apply thresholds into explicit decision tables
    4. optionally filter AnnData
    5. write machine-readable artifacts
    6. return a StageResult for provenance and downstream stages

    Args:
        config: Optional QCConfig override. If omitted, the stage resolves QC
            configuration from context.config.qc when available, otherwise it
            uses QCConfig().
        output_subdir: Subdirectory under context.paths.results where QC
            artifacts should be written.
    """

    #: Stage name, injected by ``@register_stage``. Declared so the contract is visible to a
    #: reader and to a type checker rather than appearing by decorator magic.
    name: ClassVar[str]
    config: QCConfig | None = None
    output_subdir: str = "qc"

    def run(self, context: object) -> StageResult:
        """
        Execute the QC stage.

        Args:
            context: PipelineContext-like object containing config, paths, and
                AnnData.

        Returns:
            StageResult containing the QC-updated AnnData object, written
            artifacts, notes, warnings, and structured QC metrics.

        Raises:
            QCStageError: If required context state is missing or QC execution
                fails.
        """

        # Retrieve the active AnnData object.
        adata = get_context_adata(context)

        # Resolve the effective QC configuration.
        qc_config = resolve_qc_config(context, override=self.config)

        # Return an explicit no-op result when QC is disabled.
        if not is_qc_stage_enabled(context, qc_config):
            return build_disabled_qc_stage_result(
                adata=adata,
                stage_name=self.name,
                qc_config=qc_config,
            )

        # Resolve the QC artifact output directory.
        output_dir = get_qc_output_dir(context, self.output_subdir)

        # Calculate cell-level, gene-level, and feature-family QC metrics.
        metrics_result = calculate_qc_metrics(adata, qc_config)

        # Group cells transcriptionally BEFORE anything is fitted, so every fitted quantity
        # downstream can be conditioned on cell identity rather than on the library alone.
        #
        # Computed here and not inside the graded block because the mitochondrial mixture model
        # needs it too, and it needs it more than the graded axes do: the posterior is a
        # calibrated probability, so the only correct way to stop a constitutively
        # high-mitochondrial cell type receiving a high posterior on biology alone is to FIT the
        # mixture within that cell type. Rescaling the posterior afterwards was tried and
        # corrupted it — see the metabolic axis in producers.py.
        lineage = None
        if qc_config.graded.enabled and qc_config.graded.lineage_conditional:
            lineage = provisional_lineages(
                adata,
                layer=qc_config.metrics.layer,
                resolution=qc_config.graded.lineage_resolution,
                min_genes=qc_config.graded.lineage_min_genes,
            )
            # The mixture groups by columns of cell_metrics, so the lineage has to live there.
            metrics_result.cell_metrics[LINEAGE_COLUMN] = lineage.reindex(
                metrics_result.cell_metrics.index
            ).to_numpy()
            qc_config = _condition_mixture_on_lineage(qc_config)

        # Fit the mitochondrial mixture once, here. It is a measurement rather than a
        # judgement, the artifact writer needs its table, and the graded metabolic axis needs its
        # posterior — routing it through the threshold machinery to reach either was the tie that
        # kept two QC systems alive.
        mixture = None
        if qc_config.mito_mixture.enabled:
            mixture = fit_mito_mixture(metrics_result.cell_metrics, qc_config.mito_mixture)

        # Apply the absolute floors: barcodes that are not cells and genes that are not
        # measurable. This is the whole of what the fixed-and-MAD threshold path did that graded
        # adjudication cannot express. Everything that is a *judgement* — is this cell damaged,
        # may it fit a model, may it inform a conclusion — belongs to grading, which never
        # deletes. There is one QC system now, not two.
        floors = apply_floors(
            # The same matrix the metrics were computed from, so a floor cannot disagree with
            # the numbers it is filtering on.
            get_qc_matrix(adata, qc_config)[0],
            adata.obs_names,
            adata.var_names,
            min_genes_per_cell=qc_config.floors.min_genes_per_cell,
            min_counts_per_cell=qc_config.floors.min_counts_per_cell,
            min_cells_per_gene=qc_config.floors.min_cells_per_gene,
        )

        # Floors always filter, so there is no mode to configure. Under the old `flag_no_drop`
        # default the stage computed a verdict, kept every cell, and left three places in the
        # codebase reading a boolean that controlled nothing.
        output_adata = build_qc_output_adata(adata=adata, floors=floors)

        # Stop here when the floors emptied the object.
        #
        # `fail_on_empty_result` was declared and read by nothing, so this case ran on: a
        # 50-gene test matrix met the default 200-gene floor, every cell was removed, and the
        # run continued until a downstream reduction raised `zero-size array to reduction
        # operation minimum which has no identity` five stages later. That is the least useful
        # place to learn the floor was wrong, and it is the exact mistake a first-time user
        # makes — the default floor assumes a filtered whole-transcriptome matrix.
        if qc_config.fail_on_empty_result:
            require_non_empty_qc_result(floors, n_genes=int(output_adata.n_vars))

        # Carry calculated QC metrics onto the QC AnnData before figure/h5ad
        # artifact writing. The durable metric tables remain canonical, but
        # visualization reads from obs/var columns by convention.
        metric_annotation_warnings = annotate_adata_with_qc_metrics(
            adata=output_adata,
            metrics_result=metrics_result,
        )

        # Initialize addon metrics dictionary.
        addon_metrics: dict[str, dict] = {}

        # Warnings raised inside the graded block. Collected separately because the stage's
        # main warning list is assembled further down, after the artifact manifest exists.
        graded_warnings: list[str] = []

        # Optional scoring layers: cell cycle and doublet detection. See _run_addons.
        output_adata = self._run_addons(
            output_adata,
            qc_config=qc_config,
            addon_metrics=addon_metrics,
            context=context,
        )

        # Graded adjudication (schema v2): technical evidence -> core / borderline /
        # quarantine, then per-analysis eligibility. See _adjudicate_graded.
        if qc_config.graded.enabled:
            graded_metrics, graded_block_warnings = self._adjudicate_graded(
                output_adata,
                qc_config=qc_config,
                metrics_result=metrics_result,
                mixture=mixture,
                lineage=lineage,
                context=context,
            )
            addon_metrics["graded"] = graded_metrics
            graded_warnings.extend(graded_block_warnings)

        # Resolve group_key for QC figure grouping. Prefer the central cohort
        # schema (condition, then donor, then sample), then fall back to the
        # design block, then a plain sample_id column.
        group_key = None
        context_config = getattr(context, "config", None)
        cohort = getattr(context_config, "cohort", None)
        design = getattr(context_config, "design", None)
        candidates = [
            getattr(cohort, "condition_key", None),
            getattr(cohort, "donor_key", None),
            getattr(cohort, "sample_key", None),
            getattr(design, "condition_col", None),
            getattr(design, "donor_col", None),
            "sample_id",
        ]
        for candidate in candidates:
            if candidate and candidate in output_adata.obs.columns:
                group_key = candidate
                break

        # Resolve per-cell group labels for the QC report table from the INPUT
        # object, not output_adata: the decision table is indexed by every input
        # cell (before filtering), so removed-cell counts must be attributed
        # using cell-type labels from the unfiltered obs. Prefer the design
        # cell_type_col, then a plain cell_type column; absent both, the report
        # collapses to a single TOTAL row.
        report_groups = None
        report_group_name = "cell_type"
        cell_type_candidates = [
            getattr(design, "cell_type_col", None),
            "cell_type",
        ]
        for candidate in cell_type_candidates:
            if candidate and candidate in adata.obs.columns:
                report_groups = adata.obs[candidate]
                report_group_name = candidate
                break

        # Test whether QC lost cells at the same rate in every arm of the design. This runs on
        # the UNFILTERED obs: output_adata has already lost the sub-floor barcodes, so measured
        # against it every arm's attrition is zero. A loss rate that tracks the condition is a
        # covariate, and nothing downstream can tell the difference, so the engine checks rather
        # than trusting the bars to have been fair.
        #
        # The `keep` series is deliberately NOT `floors.cell_keep` alone. It was, and that made
        # the audit blind to the axis that now does the work: the gene floor removes almost
        # nothing on real data (measured: ~0% of keratinocyte, mast, LEC and SMC removals), while
        # graded quarantine is what actually excludes cells. An audit watching only the floors
        # would have reported no differential attrition on precisely the cohort where the
        # mast-cell arm difference was real.
        attrition_audit = audit_qc_stage_attrition(
            obs=adata.obs,
            keep=_analysable_mask(adata, output_adata, floors.cell_keep),
            config=qc_config,
            cohort=cohort,
            design=design,
        )

        # Read the configuration for the two groupings that produce differential
        # attrition by construction rather than by accident, so the cause is named
        # alongside the measurement instead of leaving a reader to find it.
        design_leak_warnings = audit_qc_design_leaks(
            config=qc_config,
            cohort=cohort,
            design=design,
        )

        # Build the object the FIGURES render from. Under mode="filter",
        # output_adata has already lost the failing cells, so a keep/fail panel
        # drawn from it reports "100% pass" however many cells were dropped —
        # the 2026-09-01 VEC run dropped 503 of 3797 and its barplot read
        # "0 Fail". The decision tables are indexed by every input cell, so the
        # honest figure source is the pre-filter object carrying those decisions.
        figure_adata = build_qc_figure_adata(
            adata=adata,
            output_adata=output_adata,
            metrics_result=metrics_result,
            floors=floors,
        )

        # Resolve the obs columns the publication QC panels need. Their defaults
        # (patient_id/sample_id/condition) do not match any CellQuorum cohort
        # schema, so leaving them unset raised QCPublicationFigureError and the
        # entire publication suite was silently swallowed into a warning.
        publication_keys = resolve_publication_qc_keys(
            adata=figure_adata,
            cohort=cohort,
            design=design,
        )

        # Write all configured QC artifacts.
        artifact_manifest = write_qc_artifacts(
            output_dir=output_dir,
            metrics_result=metrics_result,
            floors=floors,
            mixture=mixture,
            config=qc_config,
            adata=output_adata,
            summary_extra=build_qc_stage_summary_extra(
                context=context,
                qc_config=qc_config,
                stage_name=self.name,
            ),
            group_key=group_key,
            report_groups=report_groups,
            report_group_name=report_group_name,
            figure_adata=figure_adata,
            publication_keys=publication_keys,
            attrition_audit=attrition_audit,
        )

        # Convert artifact manifest paths into StageArtifact records.
        stage_artifacts = build_stage_artifacts_from_manifest(artifact_manifest)

        # Combine warnings from all QC layers.
        warnings = collect_qc_stage_warnings(
            metrics_result=metrics_result,
            floors=floors,
            artifact_manifest=artifact_manifest,
        )

        # Surface any preserved-not-overwritten metric-column conflicts.
        warnings.extend(metric_annotation_warnings)

        # Surface the graded block's findings, notably an archetype whose cells are being
        # removed wholesale — the rare-population loss that no per-cell verdict can see.
        warnings.extend(graded_warnings)

        # Surface the configuration check before the measurement, so a reader sees
        # the guaranteed cause before the observed effect.
        warnings.extend(design_leak_warnings)

        # Surface differentially-filtered design factors. These belong in the
        # stage warnings and not only in the audit table: a reader who never
        # opens qc_attrition.csv is exactly the reader who needs to be told.
        warnings.extend(attrition_audit.warnings)

        # Lift the doublet layer's own channels out of the metrics dict. They were
        # only ever stored there, so a detector that came back unavailable, or one
        # that scored every cell and flagged zero doublets, was recorded in
        # provenance JSON and printed nowhere a reader of the report would look.
        doublet_addon = addon_metrics.get("doublets") or {}
        warnings.extend(doublet_addon.get("warnings", []))

        # The old no-drop guard lived here: it warned when a verdict flagged cells that were
        # then kept. There is nothing to warn about now — floors remove what they judge, and
        # grading assigns permissions rather than a verdict that something must act on.
        # Build human-readable stage notes.
        notes = build_qc_stage_notes(
            qc_config=qc_config,
            floors=floors,
            input_adata=adata,
            output_adata=output_adata,
        )
        notes.extend(doublet_addon.get("notes", []))

        # Build structured stage metrics for provenance.
        stage_metrics = build_qc_stage_metrics(
            stage_name=self.name,
            qc_config=qc_config,
            metrics_result=metrics_result,
            floors=floors,
            artifact_manifest=artifact_manifest,
            input_adata=adata,
            output_adata=output_adata,
        )

        # Merge addon metrics (cell-cycle, doublets) into stage metrics.
        if addon_metrics:
            stage_metrics.update(addon_metrics)

        # Record the attrition audit in provenance, including the tests that were
        # skipped: "checked and clean" and "never checked" must be distinguishable
        # from the run directory alone.
        stage_metrics["attrition_audit"] = attrition_audit.to_summary_dict()

        # Return the stage result.
        return StageResult(
            adata=output_adata,
            artifacts=stage_artifacts,
            notes=notes,
            warnings=warnings,
            metrics=stage_metrics,
        )

    def _run_addons(
        self,
        output_adata: ad.AnnData,
        *,
        qc_config: QCConfig,
        addon_metrics: dict[str, dict],
        context: object,
    ) -> ad.AnnData:
        """Optional per-cell scoring layers: cell cycle and doublet detection.

        Both are opt-in, both write to ``obs``, and neither removes a cell unless its own
        config says to. They live together because they share that shape and because keeping
        them inline made the phase boundaries of ``run`` impossible to see.

        Args:
            output_adata: The QC object, mutated in place with scores and flags.
            qc_config: Resolved QC configuration.
            addon_metrics: Accumulator this method adds ``doublets`` to.
            context: Pipeline context, for the R backend and the cohort sample key.

        Returns:
            The object, which is a NEW one when ``doublets.remove`` dropped cells. Returned
            rather than mutated for exactly that reason: subsetting rebinds, so a method that
            only mutated in place would silently discard the removal — which it did, and
            ``test_doublets_removed_when_remove_true`` caught it.
        """
        # Doublet detection (flag-only unless config.remove): consensus over methods.
        if qc_config.doublets.enabled:
            from cellquorum.stages.qc.doublets import detect_doublets

            backend = None
            registry = getattr(context, "backend_registry", None)
            if registry is not None:
                try:
                    backend = registry.get("rscript")
                except Exception:
                    backend = None

            # Resolve the sample/library key for per-sample doublet detection:
            # prefer the cohort sample_key, then a plain sample_id column. Doublet
            # detectors should model each capture separately, not the pooled set.
            qc_context_config = getattr(context, "config", None)
            qc_cohort = getattr(qc_context_config, "cohort", None)
            doublet_sample_key = None
            for candidate in (
                getattr(qc_cohort, "sample_key", None),
                "sample_id",
            ):
                if candidate and candidate in output_adata.obs.columns:
                    doublet_sample_key = candidate
                    break

            # compute.n_jobs is what makes QC one of the "stages that support
            # parallel execution" the field documents: scDblFinder scores each
            # capture independently, so with a sample key there are exactly
            # n_captures independent jobs to hand out. Defaults to 1, so a config
            # that never set it keeps the serial, single-threaded behaviour.
            doublet_metrics = detect_doublets(
                output_adata,
                qc_config.doublets,
                backend,
                sample_key=doublet_sample_key,
                n_jobs=resolve_n_jobs(context),
            )
            addon_metrics["doublets"] = doublet_metrics

            # Honor doublets.remove (config-gated): drop consensus-flagged
            # doublets from the output object. This is the ONLY QC path that
            # removes cells beyond threshold filtering, and it defaults off.
            if qc_config.doublets.remove and "predicted_doublet" in output_adata.obs.columns:
                doublet_mask = output_adata.obs["predicted_doublet"].to_numpy(dtype=bool)
                n_removed = int(doublet_mask.sum())
                if n_removed > 0:
                    output_adata = output_adata[~doublet_mask].copy()
                # Record the removal in the doublet metrics for provenance.
                doublet_metrics = {**doublet_metrics, "n_removed": n_removed}
                addon_metrics["doublets"] = doublet_metrics

        return output_adata

    def _adjudicate_graded(
        self,
        output_adata: ad.AnnData,
        *,
        qc_config: QCConfig,
        metrics_result: QCMetricsResult,
        mixture: MitoMixtureResult | None,
        lineage: pd.Series | None,
        context: object,
    ) -> tuple[dict[str, object], list[str]]:
        """Score technical evidence, adjudicate, and write per-analysis eligibility.

        Schema v2. Runs alongside the threshold rules rather than replacing them, so one
        object carries both and the comparison is visible. The rules still decide ``keep``;
        this decides downstream ELIGIBILITY, which is the thing ``cellquorum_qc_keep`` never
        actually controlled.

        Extracted from ``run`` because it was 204 lines of a 566-line method nested up to ten
        levels deep. The evidence table, adjudication, eligibility masks, lineage audit and
        archetype audit are all local to this phase — only the metrics and warnings escape —
        so the seam is exact rather than a convenience.

        Args:
            output_adata: The QC object. Mutated in place with evidence, verdict, eligibility
                masks, provisional lineage and archetype columns.
            qc_config: Resolved QC configuration.
            metrics_result: Computed cell/gene metrics.
            mixture: Fitted mitochondrial mixture, or None when it did not run.
            lineage: Provisional lineages computed before thresholding, or None.
            context: Pipeline context, for the cohort sample key and scratch directory.

        Returns:
            ``(graded_metrics, warnings)``.
        """
        graded_metrics: dict[str, object] = {}
        graded_warnings: list[str] = []

        # The RAW posterior, never the adjusted probability: the adjusted one folds miQC's
        # post-processing into hard 0.0/1.0 and gives unfittable cells 0.0 meaning keep. Correct
        # for a threshold; under grading 0.0 reads as "measured, no concern", which is the
        # absent-evidence-as-health failure.
        mito_posterior = None
        if mixture is not None:
            posterior = mixture.posterior
            mito_posterior = posterior if not posterior.empty else None
            graded_warnings.extend(mixture.warnings)

        from cellquorum.stages.qc.archetypes import ARCHETYPE_COLUMN, audit_archetypes
        from cellquorum.stages.qc.eligibility import (
            Analysis,
            Permission,
            build_eligibility_masks,
        )
        from cellquorum.stages.qc.evidence import (
            AdjudicationPolicy,
            adjudicate_initial,
        )
        from cellquorum.stages.qc.producers import build_evidence_table

        graded_config = qc_config.graded

        # Scale severities within the cohort sample key so a shallow library is not
        # judged against a deep one.
        sample_key = getattr(getattr(context, "config", None), "cohort", None)
        sample_key = getattr(sample_key, "sample_key", None) or "sample_id"

        # Judge each cell against cells of its own kind. Without this, severity is
        # measured against a sample-wide median, so a cell type whose *normal* biology is
        # low-complexity and high-mitochondrial reads as damaged on two families at once
        # and is quarantined for being itself. See lineage.py for the measurement.
        # Reuse the grouping computed before thresholding; recomputing it would risk two
        # different groupings deciding the mixture and the severity axes.
        null_grouping = None
        if lineage is not None:
            lineage = lineage.reindex(output_adata.obs_names)
            null_grouping = resolve_null_groups(
                output_adata.obs,
                sample_key=sample_key if sample_key in output_adata.obs.columns else None,
                lineage=lineage,
                min_cells=graded_config.lineage_min_cells,
            )

        evidence = build_evidence_table(
            output_adata,
            metrics_result.cell_metrics.reindex(output_adata.obs_names),
            group_key=sample_key if sample_key in output_adata.obs.columns else None,
            layer=qc_config.metrics.layer,
            mito_posterior=mito_posterior,
            nuclear_axis_applicable=graded_config.nuclear_axis_applicable,
            grouping=null_grouping,
            lineage_conditional=null_grouping is not None,
        )

        # The absolute-scale table, kept only for the per-lineage audit below. Per-cell
        # verdicts come from the lineage-conditional table above; "this entire group looks
        # like debris" is a statement the absolute scale alone can make.
        absolute_evidence = (
            build_evidence_table(
                output_adata,
                metrics_result.cell_metrics.reindex(output_adata.obs_names),
                group_key=sample_key if sample_key in output_adata.obs.columns else None,
                layer=qc_config.metrics.layer,
                mito_posterior=mito_posterior,
                nuclear_axis_applicable=graded_config.nuclear_axis_applicable,
            )
            if null_grouping is not None
            else evidence
        )
        adjudication = adjudicate_initial(
            evidence,
            AdjudicationPolicy(
                concern_severity=graded_config.concern_severity,
                severe_severity=graded_config.severe_severity,
                min_concordant_families=graded_config.min_concordant_families,
                uninformative_capture_severity=graded_config.uninformative_capture_severity,
                min_coverage_for_quarantine=graded_config.min_coverage_for_quarantine,
                multiplet_severity=graded_config.multiplet_severity,
            ),
        )

        # Turn the verdict into per-analysis eligibility. This is the step that makes
        # QC load-bearing: the previous single `keep` boolean was read by three places
        # in the codebase, two of them figure code, so a careful verdict controlled
        # nothing. Stages declare their fit scope at registration and read these masks.
        eligibility = build_eligibility_masks(
            adjudication.state, probable_multiplet=adjudication.probable_multiplet
        )

        # Per-lineage audit. Two things per-cell verdicts cannot say: "this whole group
        # looks like debris" (suspect) and "this whole group is being dropped, and if it is
        # real biology that is the rare-population loss" (vulnerable).
        lineage_audit = None
        if lineage is not None:
            lineage_audit = audit_lineages(
                lineage,
                absolute_evidence.damage_family_severity(),
                ~eligibility.mask(Analysis.MANIFOLD, Permission.FIT),
                adjudication.probable_multiplet,
                suspect_severity=graded_config.lineage_suspect_severity,
                vulnerable_fraction=graded_config.lineage_vulnerable_fraction,
            )
            output_adata.obs[LINEAGE_COLUMN] = lineage.to_numpy()
            if null_grouping is not None:
                # Narrowed on the value being used rather than on the correlated `lineage`,
                # so the guard is checkable instead of merely true in practice.
                output_adata.obs[NULL_LEVEL_COLUMN] = null_grouping.level.to_numpy()
            # Stored column-wise, not as a list of records: anndata has no native
            # representation for a list of dicts and silently writes it as one long string,
            # which makes the audit unreadable to anything but a human squinting at repr.
            output_adata.uns.setdefault("cellquorum", {})["qc_lineage_audit"] = _audit_to_columns(
                lineage_audit, index_name="lineage"
            )

            # Archetype audit: vertices, not blobs, so a population too small for Leiden
            # can still be seen. Optional and self-disabling — partipy is GPL-3 and lives
            # in its own environment, so absence is the normal case.
            if graded_config.archetype_audit and PROVISIONAL_EMBEDDING in output_adata.obsm:
                embedding = np.asarray(output_adata.obsm[PROVISIONAL_EMBEDDING])
                placed = np.isfinite(embedding).all(axis=1)
                if int(placed.sum()) > 50:
                    archetype = audit_archetypes(
                        embedding[placed],
                        output_adata.obs_names[placed],
                        ~eligibility.mask(Analysis.MANIFOLD, Permission.FIT)[placed],
                        (
                            output_adata.layers[qc_config.metrics.layer][placed]
                            if qc_config.metrics.layer in output_adata.layers
                            else output_adata.X[placed]
                        ),
                        n_archetypes_max=graded_config.archetype_max,
                        bootstrap=graded_config.archetype_bootstrap,
                        max_cells=graded_config.archetype_max_cells,
                        n_restarts=graded_config.archetype_restarts,
                        timeout_seconds=graded_config.archetype_timeout_seconds,
                        scratch_dir=getattr(getattr(context, "paths", None), "scratch", None),
                    )
                    if archetype.available and archetype.dominant is not None:
                        output_adata.obs[ARCHETYPE_COLUMN] = (
                            archetype.dominant.reindex(output_adata.obs_names)
                            .fillna("unsampled")
                            .to_numpy()
                        )
                        output_adata.uns["cellquorum"]["qc_archetype_audit"] = _audit_to_columns(
                            archetype.table, index_name="archetype"
                        )
                        for label, row in archetype.flagged().iterrows():
                            graded_warnings.append(
                                f"Archetype {label} (n={int(row['n_supporting'])}) has "
                                f"{100.0 * row['excluded_fraction']:.0f}% of its cells "
                                f"excluded from fitting and is "
                                + (
                                    "transcriptionally coherent, so a real population may "
                                    "be being removed — inspect before trusting the run."
                                    if row["losing_a_population"]
                                    else "incoherent, so it is most likely debris that QC "
                                    "is correctly removing."
                                )
                            )
                    else:
                        logger.info("Archetype audit unavailable: %s", archetype.reason)

        # Write evidence, verdict, and eligibility onto the object so every downstream
        # stage and figure reads one source of truth.
        for frame in (
            evidence.to_obs_frame(),
            adjudication.to_obs_frame(),
            eligibility.to_obs_frame(),
        ):
            for column in frame.columns:
                output_adata.obs[column] = frame[column].to_numpy()

        # Self-check: compare the verdict against the evidence it claims to rest on, and fail
        # rather than report a plausible wrong answer. Every defect in this area was found by a
        # human asking a question; this is that question, asked by the run.
        self_check = run_self_check(
            adjudication.state,
            # The mixture AXIS, not the metabolic family rollup. The family is a max over the
            # posterior and the dissociation-stress fraction, so comparing the rollup against the
            # posterior flags a legitimate difference — the check's own first false positive,
            # caught by the check firing on a clean run.
            metabolic_severity=next(
                (axis.severity for axis in evidence.axes if axis.name == "mito_mixture_posterior"),
                None,
            ),
            mito_posterior=mito_posterior,
            null_level=None if null_grouping is None else null_grouping.level,
            null_keys=None if null_grouping is None else null_grouping.keys,
            lineage_audit=lineage_audit,
            fit_mask=eligibility.mask(Analysis.MANIFOLD, Permission.FIT),
            minimum_core=graded_config.self_check_minimum_core,
        )
        graded_warnings.extend(self_check.warnings())
        if graded_config.self_check_fails_run and self_check.failures():
            raise QCStageError(
                "QC self-check failed, so the run stopped rather than emitting a verdict its own "
                "evidence contradicts:\n"
                + "\n".join(f"  - {check.name}: {check.detail}" for check in self_check.failures())
                + "\n\nSet qc.graded.self_check_fails_run=false to downgrade these to warnings."
            )

        state_counts = adjudication.counts()
        graded_metrics.update(
            {
                **state_counts,
                "self_check": self_check.summary(),
                "null_group_levels": ({} if null_grouping is None else null_grouping.summary()),
                "n_lineages": (0 if lineage_audit is None else int(len(lineage_audit))),
                "n_suspect_lineages": (
                    0 if lineage_audit is None else int(lineage_audit["suspect"].sum())
                ),
                "n_vulnerable_lineages": (
                    0 if lineage_audit is None else int(lineage_audit["vulnerable"].sum())
                ),
                "families": [str(family) for family in evidence.families_present()],
                "median_evidence_coverage": float(adjudication.coverage.median()),
                "n_probable_multiplet": int(adjudication.probable_multiplet.sum()),
                "reasons": {
                    str(reason): int(count)
                    for reason, count in adjudication.reason.value_counts().items()
                },
                # Eligible counts per analysis, so provenance records what the verdict
                # actually permitted rather than only what it decided.
                "eligibility": eligibility.summary(),
            }
        )
        logger.info(
            "QC graded adjudication: core=%s borderline=%s quarantine=%s "
            "(coverage median %.2f over %s families)",
            state_counts.get("core"),
            state_counts.get("borderline"),
            state_counts.get("quarantine"),
            float(adjudication.coverage.median()),
            len(evidence.families_present()),
        )

        return graded_metrics, graded_warnings


# Re-exported for the module's public surface. The helpers now live in _context/_annotate/
# _report; these names stay importable from here because callers outside the package use
# them, and a refactor should not be a breaking change.
__all__ = [
    "QCStage",
    "QCStageError",
    "annotate_adata_with_qc_metrics",
    "build_disabled_qc_stage_result",
    "build_qc_figure_adata",
    "build_qc_output_adata",
    "build_qc_stage_metrics",
    "build_qc_stage_notes",
    "build_qc_stage_summary_extra",
    "build_stage_artifacts_from_manifest",
    "collect_qc_stage_warnings",
    "get_context_adata",
    "get_qc_output_dir",
    "is_qc_stage_enabled",
    "resolve_publication_qc_keys",
    "resolve_qc_config",
]
