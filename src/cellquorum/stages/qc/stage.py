# Pipeline step (order=20): qc — metrics, floors, evidence, and analysis eligibility.
"""QC pipeline stage for CellQuorum."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.context import resolve_n_jobs
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.stages.qc._context import (
    get_context_adata,
    get_qc_output_dir,
    is_qc_stage_enabled,
    resolve_qc_config,
)
from cellquorum.stages.qc._errors import QCStageError
from cellquorum.stages.qc.artifacts import write_qc_artifacts
from cellquorum.stages.qc.attrition import audit_qc_design_leaks, audit_qc_stage_attrition
from cellquorum.stages.qc.config import QCConfig
from cellquorum.stages.qc.floors import floors_from_metrics, require_non_empty_qc_result
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
from cellquorum.stages.qc.reporting import (
    _audit_to_columns,
    annotate_adata_with_qc_metrics,
    build_disabled_qc_stage_result,
    build_qc_figure_adata,
    build_qc_output_adata,
    build_qc_stage_metrics,
    build_qc_stage_notes,
    build_qc_stage_summary_extra,
    build_stage_artifacts_from_manifest,
    collect_qc_stage_warnings,
    compute_qc_donor_comparisons,
    resolve_publication_qc_keys,
    summarize_ambient_correction,
)
from cellquorum.stages.qc.selfcheck import run_self_check
from cellquorum.stages.qc.validation import get_qc_matrix

logger = logging.getLogger(__name__)


def _condition_mixture_on_lineage(qc_config: QCConfig) -> QCConfig:
    """Fit the mitochondrial mixture within cell identity, not just within library."""
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

    Args:
        input_adata: The object as it entered QC, defining the index.
        output_adata: The post-floor object carrying the graded state column, if any.
        cell_keep: Per-barcode floor mask over the input index.

    Returns:
        Boolean Series over ``input_adata.obs_names``. Barcodes removed by a floor are False;
        a cell present in the output is False when quarantined and True otherwise.
    """
    analysable = cell_keep.reindex(input_adata.obs_names).fillna(False).astype(bool)

    analysable &= input_adata.obs_names.isin(output_adata.obs_names)
    multiplet = output_adata.obs.get("qc_probable_multiplet")
    if multiplet is not None:
        analysable &= ~multiplet.reindex(input_adata.obs_names).fillna(False).astype(bool)
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
    """Execute the complete CellQuorum QC module.

    Args:
        config: Optional QCConfig override. If omitted, the stage resolves QC
            configuration from context.config.qc when available, otherwise it
            uses QCConfig().
        output_subdir: Subdirectory under context.paths.results where QC
            artifacts should be written.
    """

    name: ClassVar[str]
    config: QCConfig | None = None
    output_subdir: str = "qc"

    def run(self, context: object) -> StageResult:
        """Execute the QC stage.

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

        adata = get_context_adata(context)

        qc_config = resolve_qc_config(context, override=self.config)

        if not is_qc_stage_enabled(context, qc_config):
            return build_disabled_qc_stage_result(
                adata=adata,
                stage_name=self.name,
                qc_config=qc_config,
            )

        ambient_status = summarize_ambient_correction(adata)
        if qc_config.ambient.correction_enabled and (
            ambient_status["status"] != "upstream_recorded"
            or ambient_status["method"] != qc_config.ambient.method
        ):
            raise QCStageError(
                "qc.ambient cannot perform ambient correction. Configure the upstream "
                "ambient_correction stage with its required inputs, or supply corrected data "
                "with matching correction provenance."
            )

        output_dir = get_qc_output_dir(context, self.output_subdir)

        metrics_result = calculate_qc_metrics(adata, qc_config)

        # Group cells transcriptionally BEFORE anything is fitted, so every fitted quantity
        # downstream can be conditioned on cell identity rather than on the library alone.
        #
        # Computed here and not inside the graded block because the mitochondrial mixture model
        # needs it too, and it needs it more than the graded axes do: the posterior is a
        # calibrated probability, so the only correct way to stop a constitutively
        # high-mitochondrial cell type receiving a high posterior on biology alone is to FIT the
        # mixture within that cell type. Rescaling the posterior afterwards was tried and
        # corrupted it — see the metabolic axis in evidence.py.
        lineage = None
        if qc_config.graded.enabled and qc_config.graded.lineage_conditional:
            lineage = provisional_lineages(
                adata,
                layer=qc_config.metrics.layer,
                use_raw=qc_config.metrics.use_raw,
                resolution=qc_config.graded.lineage_resolution,
                min_genes=qc_config.graded.lineage_min_genes,
            )

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

        sampleqc = None
        if qc_config.sampleqc.enabled:
            from cellquorum.backends.rscript import RscriptBackend
            from cellquorum.stages.qc.sampleqc import fit_sampleqc

            registry = getattr(context, "backend_registry", None)
            backend = registry.get("rscript") if registry is not None else RscriptBackend()
            if backend is None:
                raise QCStageError("SampleQC requires the Rscript backend.")
            sampleqc = fit_sampleqc(metrics_result.cell_metrics, qc_config.sampleqc, backend)
            sampleqc.provenance["matrix_source"] = metrics_result.summary.get(
                "matrix_source", "unknown"
            )
            for column in sampleqc.cells:
                metrics_result.cell_metrics[column] = sampleqc.cells[column]

        # Apply the absolute floors: barcodes that are not cells and genes that are not
        # measurable. This is the whole of what the fixed-and-MAD threshold path did that graded
        # adjudication cannot express. Everything that is a *judgement* — is this cell damaged,
        # may it fit a model, may it inform a conclusion — belongs to grading, which never
        # deletes. There is one QC system now, not two.
        floors = floors_from_metrics(
            metrics_result.cell_metrics,
            metrics_result.gene_metrics,
            min_genes_per_cell=qc_config.floors.min_genes_per_cell,
            min_counts_per_cell=qc_config.floors.min_counts_per_cell,
            min_cells_per_gene=qc_config.floors.min_cells_per_gene,
        )

        output_adata = build_qc_output_adata(adata=adata, floors=floors)
        if sampleqc is None:
            output_adata.obs.drop(
                columns=[
                    "sampleqc_distance",
                    "sampleqc_pvalue",
                    "sampleqc_component",
                    "sampleqc_outlier",
                    "sampleqc_status",
                ],
                errors="ignore",
                inplace=True,
            )
            output_adata.uns.get("cellquorum", {}).pop("sampleqc", None)

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

        metric_annotation_warnings = annotate_adata_with_qc_metrics(
            adata=output_adata,
            metrics_result=metrics_result,
        )

        addon_metrics: dict[str, dict] = {}
        if sampleqc is not None:
            addon_metrics["sampleqc"] = sampleqc.provenance
            output_adata.uns.setdefault("cellquorum", {})["sampleqc"] = sampleqc.provenance

        graded_warnings: list[str] = []

        output_adata = self._score_doublets(
            output_adata,
            qc_config=qc_config,
            addon_metrics=addon_metrics,
            context=context,
        )

        if qc_config.graded.enabled:
            graded_metrics, graded_block_warnings = self._adjudicate_graded(
                output_adata,
                qc_config=qc_config,
                metrics_result=metrics_result,
                mixture=mixture,
                lineage=lineage,
                context=context,
                expression_adata=adata,
            )
            addon_metrics["graded"] = graded_metrics
            graded_warnings.extend(graded_block_warnings)

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

        design_leak_warnings = audit_qc_design_leaks(
            config=qc_config,
            cohort=cohort,
            design=design,
        )

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

        donor_comparisons, comparison_warnings = compute_qc_donor_comparisons(
            figure_adata, publication_keys, paired=bool(getattr(design, "paired", False))
        )

        artifact_manifest = write_qc_artifacts(
            output_dir=output_dir,
            metrics_result=metrics_result,
            floors=floors,
            mixture=mixture,
            config=qc_config,
            adata=output_adata,
            summary_extra={
                **build_qc_stage_summary_extra(
                    context=context, qc_config=qc_config, stage_name=self.name
                ),
                "ambient_correction": ambient_status,
                "sampleqc": sampleqc.provenance if sampleqc is not None else {"status": "disabled"},
            },
            group_key=group_key,
            report_groups=report_groups,
            report_group_name=report_group_name,
            figure_adata=figure_adata,
            publication_keys=publication_keys,
            attrition_audit=attrition_audit,
            donor_comparisons=donor_comparisons,
        )

        stage_artifacts = build_stage_artifacts_from_manifest(artifact_manifest)

        warnings = collect_qc_stage_warnings(
            metrics_result=metrics_result,
            floors=floors,
            artifact_manifest=artifact_manifest,
        )

        warnings.extend(comparison_warnings)
        warnings.extend(metric_annotation_warnings)

        warnings.extend(graded_warnings)

        warnings.extend(design_leak_warnings)

        warnings.extend(attrition_audit.warnings)

        doublet_addon = addon_metrics.get("doublets") or {}
        warnings.extend(doublet_addon.get("warnings", []))

        notes = build_qc_stage_notes(
            qc_config=qc_config,
            floors=floors,
            input_adata=adata,
            output_adata=output_adata,
        )
        notes.extend(doublet_addon.get("notes", []))
        notes.append(
            f"Upstream ambient correction: {ambient_status['method']} recorded."
            if ambient_status["status"] == "upstream_recorded"
            else "Ambient correction is not recorded; QC does not remove ambient RNA."
        )

        stage_metrics = build_qc_stage_metrics(
            stage_name=self.name,
            qc_config=qc_config,
            metrics_result=metrics_result,
            floors=floors,
            artifact_manifest=artifact_manifest,
            input_adata=adata,
            output_adata=output_adata,
        )

        if addon_metrics:
            stage_metrics.update(addon_metrics)

        stage_metrics["attrition_audit"] = attrition_audit.to_summary_dict()
        stage_metrics["ambient_correction"] = ambient_status

        return StageResult(
            adata=output_adata,
            artifacts=stage_artifacts,
            notes=notes,
            warnings=warnings,
            metrics=stage_metrics,
        )

    def _score_doublets(
        self,
        output_adata: ad.AnnData,
        *,
        qc_config: QCConfig,
        addon_metrics: dict[str, dict],
        context: object,
    ) -> ad.AnnData:
        """Run doublet detection using the selected QC count source.

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

        if qc_config.doublets.enabled:
            from cellquorum.stages.qc.doublets import detect_doublets

            backend = None
            registry = getattr(context, "backend_registry", None)
            if registry is not None:
                try:
                    backend = registry.get("rscript")
                except Exception:
                    backend = None

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

            doublet_matrix, doublet_source = get_qc_matrix(output_adata, qc_config)
            doublet_adata = ad.AnnData(
                X=doublet_matrix,
                obs=output_adata.obs.copy(),
                var=pd.DataFrame(
                    index=(
                        output_adata.raw.var_names
                        if qc_config.metrics.use_raw
                        else output_adata.var_names
                    )
                ),
            )
            doublet_metrics = detect_doublets(
                doublet_adata,
                qc_config.doublets,
                backend,
                sample_key=doublet_sample_key,
                n_jobs=resolve_n_jobs(context),
                random_state=getattr(context, "random_seed", 0),
            )
            output_adata.obs = doublet_adata.obs
            doublet_metrics["matrix_source"] = doublet_source
            addon_metrics["doublets"] = doublet_metrics

            # Honor doublets.remove (config-gated): drop consensus-flagged
            # doublets from the output object. This is the ONLY QC path that
            # removes cells beyond threshold filtering, and it defaults off.
            if qc_config.doublets.remove and "predicted_doublet" in output_adata.obs.columns:
                doublet_mask = output_adata.obs["predicted_doublet"].to_numpy(dtype=bool)
                n_removed = int(doublet_mask.sum())
                if n_removed > 0:
                    output_adata = output_adata[~doublet_mask].copy()
                if qc_config.fail_on_empty_result and output_adata.n_obs == 0:
                    raise QCStageError(
                        "Doublet removal left no cells. Inspect the detector calls or disable "
                        "qc.doublets.remove before continuing."
                    )

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
        expression_adata: ad.AnnData | None = None,
    ) -> tuple[dict[str, object], list[str]]:
        """Score technical evidence, adjudicate, and write per-analysis eligibility.

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
            build_evidence_table,
        )

        graded_config = qc_config.graded

        sample_key = getattr(getattr(context, "config", None), "cohort", None)
        sample_key = getattr(sample_key, "sample_key", None) or "sample_id"

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
            use_raw=qc_config.metrics.use_raw,
            expression_adata=expression_adata,
            mito_posterior=mito_posterior,
            nuclear_axis_applicable=graded_config.nuclear_axis_applicable,
            grouping=null_grouping,
            lineage_conditional=null_grouping is not None,
        )

        absolute_evidence = (
            build_evidence_table(
                output_adata,
                metrics_result.cell_metrics.reindex(output_adata.obs_names),
                group_key=sample_key if sample_key in output_adata.obs.columns else None,
                layer=qc_config.metrics.layer,
                use_raw=qc_config.metrics.use_raw,
                expression_adata=expression_adata,
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
            called_doublets=(
                output_adata.obs.get("predicted_doublet") if qc_config.doublets.enabled else None
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
                output_adata.obs[NULL_LEVEL_COLUMN] = null_grouping.level.to_numpy()

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
