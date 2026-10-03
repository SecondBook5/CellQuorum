# Pipeline step (order=155): subclustering — resolve finer structure within populations.
#
# The first line is a machine-read contract (tests/core/test_stage_headers.py): one line,
# `order=` matching the registration, ending in a period. Wrapping the sentence onto a
# second line breaks it, so context goes here instead.
#
# Runs AFTER reference_mapping (120) and population_identity (150); see the registration
# below for why.
"""Subclustering stage implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellquorum.core.exceptions import CellQuorumConfigError
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_artifact_writer import StageArtifactWriter
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.clustering.subclustering.diagnostics import (
    plot_group_recovery,
    plot_marker_identity_dotplot,
    plot_subcluster_qc_panel,
    plot_technical_qc_consistency,
)
from cellquorum.stages.clustering.subclustering.donor_gate import (
    apply_qc_flags,
    donor_reproducibility,
)
from cellquorum.stages.clustering.subclustering.extract import (
    apply_group_filter,
    ensure_focus_embedding,
    extract_focus,
    reembed_focus_batch_aware,
)
from cellquorum.stages.clustering.subclustering.partition import (
    run_choir,
    run_leiden_stability,
    run_scshc_test,
)
from cellquorum.stages.clustering.subclustering.qc_consistency import (
    subcluster_qc_consistency,
)

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd

    from cellquorum.core.context import PipelineContext
    from cellquorum.stages.clustering.subclustering.config import FocusConfig, SubclusteringConfig


class SubclusteringFocusError(CellQuorumConfigError):
    """The configured focus selects no cells, so there is nothing to subcluster."""


@dataclass
class _Report:
    """What a stage run accumulates for its result, passed to each phase as one object."""

    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: list[StageArtifact] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


# order=155, moved from 100.
#
# At 100 this ran BEFORE reference_mapping (120), so the atlas labels it should focus on did
# not exist yet: `focus.label_key` defaulted to `cell_type`, which nothing had written, and
# `focus.labels: []` means "keep all cells" -- so on a real cohort it handed CHOIR all 201,871
# cells and hit its 1800s timeout. There was no way to say "subcluster the LEC" because
# nothing had yet decided which cells were LEC.
#
# 155 puts it after population_identity (150) rather than merely after the mapping. That
# stage audits whether a population is supported by enough cells, samples and donors, or is
# one donor's artifact. Subclustering asks "does this population contain distinct subtypes",
# which is only worth asking of a population that is real -- and CHOIR permutation-tests
# every split, so it is expensive enough to be worth ordering deliberately.
#
# Safe to move: nothing outside this stage reads the `subcluster` column.
@register_stage(
    name="subclustering",
    order=155,
    config_flag="subclustering",
    config_field="subclustering",
    category="subclustering",
)
class SubclusteringStage:
    """
    Principled subclustering stage.

    This stage orchestrates:
    1. Focus extraction (subset to lineage)
    2. Group filter (drop groups with < N focus cells)
    3. Re-embedding (Task 2)
    4. Partition (CHOIR + sc-SHC, Task 2)
    5. Donor gate (Task 3)
    6. Diagnostics (clustree, stability curve)

    Task 1 scope: steps 1-2 only (extract + group_filter).
    """

    def run(self, context: PipelineContext) -> StageResult:
        """
        Run principled subclustering pipeline.

        Args:
            context: Pipeline context exposing config, adata, paths.

        Returns:
            StageResult with focused/filtered subset as adata.
        """
        config = getattr(context, "config", None)
        sc_config = getattr(config, "subclustering", None)
        adata = getattr(context, "adata", None)

        # Skip (non-silent) when disabled.
        if sc_config is None or not getattr(sc_config, "enabled", False):
            return StageResult(
                adata=adata,
                warnings=["subclustering disabled by config"],
                metrics={"skipped": True, "reason": "disabled by config"},
            )

        # Require adata.
        if adata is None:
            return StageResult(
                adata=None,
                warnings=["subclustering skipped: no adata available"],
                metrics={"skipped": True, "reason": "no adata"},
            )

        focus_config = self._resolve_focus_config(config, sc_config)
        report = _Report()
        focused = self._extract_focus(adata, focus_config, sc_config, report)
        filtered = self._filter_groups(context, sc_config, focused, report)

        # Task 2: partition + formal_test.
        # Resolve backend from context.
        backend = self._resolve_rscript_backend(context)
        scratch = Path(getattr(getattr(context, "paths", None), "scratch", "."))

        # The nuisance variable CHOIR corrects for and sc-SHC conditions on.
        nuisance_key = self._resolve_nuisance_key(config, sc_config)
        if nuisance_key is None:
            report.warnings.append(
                "Subclustering has no batch/donor key to correct for: declare "
                "cohort.batch_key or cohort.donor_key. CHOIR will partition an "
                "uncorrected embedding, so its cluster count may be donor-driven."
            )
        else:
            report.notes.append(f"Batch correction / conditioning key: {nuisance_key}")
            report.metrics["nuisance_key"] = nuisance_key

        # Run partition (CHOIR or fallback). Initialized here (not just inside the
        # "choir" branch) so the donor gate below can look at it regardless of
        # which partition method ran, without a NameError.
        partition_reduction_key: str | None = None
        if sc_config.partition.method == "choir":
            filtered, partition_reduction_key = self._partition_choir(
                filtered, sc_config, backend, scratch, nuisance_key, report
            )
        elif sc_config.partition.method == "leiden_stability":
            filtered = self._partition_leiden_stability(
                context, filtered, sc_config, nuisance_key, report
            )
        elif sc_config.partition.method == "leiden_grid":
            # Minimal scanpy fallback (defer detailed implementation).
            report.notes.append("leiden_grid partition: deferred (CHOIR is primary method)")
            report.metrics["partition_skipped"] = True
            report.metrics["partition_skip_reason"] = "leiden_grid not implemented"
        else:
            report.notes.append(f"Unknown partition method: {sc_config.partition.method}")
            report.warnings.append(f"Unknown partition method: {sc_config.partition.method}")

        cluster_key = sc_config.key_added
        has_cluster_labels = (
            cluster_key in filtered.obs.columns and filtered.obs[cluster_key].notna().any()
        )

        filtered = self._run_donor_gate(
            context,
            config,
            sc_config,
            filtered,
            partition_reduction_key,
            cluster_key,
            has_cluster_labels,
            report,
        )
        filtered = self._run_technical_qc(
            context, sc_config, filtered, cluster_key, has_cluster_labels, report
        )
        self._add_marker_dotplot(
            context, sc_config, filtered, cluster_key, has_cluster_labels, report
        )

        # Generate clustree plot (if leiden_grid labels present).
        # Note: clustree expects multiple cluster columns with a common prefix.
        # This will be driven by leiden_grid labels when implemented.
        # For now, defer clustree generation to full leiden_grid implementation.
        if sc_config.diagnostics.clustree:
            report.notes.append("Clustree: deferred (leiden_grid not implemented)")

        # Project subclustering results back onto the ORIGINAL parent object and
        # return THAT, not the focus subset. extract_focus stripped the parent's
        # X_* embeddings and (under a group filter or focus < all cells) shrank
        # the cell set; returning it would deprive downstream stages of the
        # integration embedding they depend on. Instead we keep the parent (all
        # cells, embeddings intact) and merge the per-cell subcluster labels /
        # donor-QC flags back on by obs index. Cells outside the analyzed focus
        # simply carry NaN for those columns.
        result_adata = self._project_onto_parent(
            parent=adata,
            focused=focused,
            filtered=filtered,
            sc_config=sc_config,
        )

        return StageResult(
            adata=result_adata,
            artifacts=report.artifacts,
            notes=report.notes,
            warnings=report.warnings,
            metrics=report.metrics,
        )

    @staticmethod
    def _resolve_focus_config(config: Any, sc_config: SubclusteringConfig) -> FocusConfig:
        """Resolve the focus lineage from the cohort schema if the block declared none."""

        # Resolve the focus lineage from the central cohort schema when the
        # subclustering block did not declare its own (declare-once). The
        # cohort focus is the generic replacement for a hard-coded lineage.
        focus_config = sc_config.focus
        cohort = getattr(config, "cohort", None)
        cohort_focus = getattr(cohort, "focus", None)
        if not focus_config.labels and cohort_focus is not None and cohort_focus.labels:
            focus_config = focus_config.model_copy(
                update={
                    "label_key": cohort_focus.label_key or focus_config.label_key,
                    "labels": list(cohort_focus.labels),
                }
            )

        return focus_config

    @staticmethod
    def _extract_focus(
        adata: ad.AnnData,
        focus_config: FocusConfig,
        sc_config: SubclusteringConfig,
        report: _Report,
    ) -> ad.AnnData:
        """Subset to the focus lineage and refuse a focus that matches no cells."""

        # Extract focus lineage.
        focused = extract_focus(
            adata,
            focus_config,
            sc_config.counts_layer,
        )

        # Record focus extraction provenance.
        if "subcluster_extraction" in focused.uns:
            prov = focused.uns["subcluster_extraction"]
            report.notes.append(
                f"Focus extracted: {prov['n_cells_kept']} / "
                f"{prov['n_cells_total']} cells "
                f"({prov['label_key']} in {prov['labels']})"
            )
            report.metrics["n_cells_before_focus"] = prov["n_cells_total"]
            report.metrics["n_cells_after_focus"] = prov["n_cells_kept"]

        # Guard: a focus that matches zero cells must NOT propagate an empty
        # object as the pipeline's working adata. Doing so previously poisoned
        # every downstream stage (population_identity/embeddings/DE/CCC all ran
        # on 0 cells).
        #
        # Fail rather than skip. The label column is known to exist (extract_focus
        # would have raised otherwise), so zero matches means the configured labels
        # are simply not in it — a config mistake with no valid interpretation. This
        # was previously a warning, and a run whose subclustering block was copied
        # from another lineage completed "successfully" with no subtypes at all;
        # the mistake surfaced a day later while reading provenance.
        if focused.n_obs == 0:
            available = sorted({str(v) for v in adata.obs[focus_config.label_key].unique()})
            raise SubclusteringFocusError(
                f"subclustering focus '{focus_config.label_key} in {focus_config.labels}' "
                f"matched 0 of {adata.n_obs} cells. Present in "
                f"'{focus_config.label_key}': {available}. Fix the focus labels, or set "
                "stages.subclustering false if this lineage should not be subclustered."
            )

        return focused

    @staticmethod
    def _filter_groups(
        context: PipelineContext,
        sc_config: SubclusteringConfig,
        focused: ad.AnnData,
        report: _Report,
    ) -> ad.AnnData:
        """Drop groups with too few focus cells (when configured) and plot the recovery curve."""

        # Apply group filter (if configured).
        group_key = sc_config.group_filter.group_key
        min_cells = sc_config.group_filter.min_cells
        if group_key is not None and min_cells is not None:
            filtered, filter_prov = apply_group_filter(
                focused,
                group_key,
                min_cells,
            )

            # Record group filter provenance.
            if filter_prov["applied"]:
                n_kept = len(filter_prov["kept"])
                n_dropped = len(filter_prov["dropped"])
                report.notes.append(
                    f"Group filter: kept {n_kept} groups, "
                    f"dropped {n_dropped} groups "
                    f"(< {min_cells} cells in {group_key})"
                )
                if filter_prov["dropped"]:
                    report.warnings.append(f"Dropped groups: {', '.join(filter_prov['dropped'])}")
                report.metrics["n_groups_kept"] = n_kept
                report.metrics["n_groups_dropped"] = n_dropped
                report.metrics["n_cells_after_group_filter"] = filtered.n_obs

                # Generate recovery plot.
                paths = getattr(context, "paths", None)
                if paths is not None:
                    figures_dir = Path(getattr(paths, "figures", "."))
                    recovery_path = figures_dir / "subclustering_group_recovery.png"
                    plot_group_recovery(
                        filter_prov["counts"],
                        min_cells,
                        group_key,
                        recovery_path,
                    )
                    report.artifacts.append(
                        StageArtifact(
                            name="subclustering_group_recovery",
                            path=recovery_path,
                            kind="png",
                            description=(
                                f"Group recovery: {group_key} cell counts "
                                f"vs. threshold ({min_cells} cells)"
                            ),
                        )
                    )
            else:
                filtered = focused
                report.notes.append("Group filter: skipped (not configured)")
        else:
            filtered = focused
            report.notes.append("Group filter: skipped (not configured)")

        return filtered

    def _partition_choir(
        self,
        filtered: ad.AnnData,
        sc_config: SubclusteringConfig,
        backend: Any,
        scratch: Path,
        nuisance_key: str | None,
        report: _Report,
    ) -> tuple[ad.AnnData, str | None]:
        """Re-embed batch-aware, partition with CHOIR, then test the partition with sc-SHC.

        Returns the (possibly labelled) object and the reduction CHOIR used, which the donor
        gate reuses so it tests on the same corrected space."""

        # Batch-aware re-embedding must happen BEFORE partitioning: CHOIR
        # needs it to cluster on a corrected space, and the donor gate
        # reuses this SAME corrected embedding below rather than deriving
        # its own uncorrected one -- see the donor-gate section.
        partition_reduction_key = reembed_focus_batch_aware(
            filtered,
            counts_layer=sc_config.counts_layer,
            batch_key=nuisance_key,
            random_state=(sc_config.partition.seeds[0] if sc_config.partition.seeds else 0),
            max_iter_harmony=sc_config.reembed.max_iter_harmony,
            diagnostics=report.warnings,
        )
        if partition_reduction_key:
            report.notes.append(f"CHOIR reduction: {partition_reduction_key}")
            report.metrics["choir_reduction"] = partition_reduction_key
        else:
            # A warning, not a note. CHOIR declares its cluster count
            # significant *in the space it was given*, so an uncorrected
            # space means the count can be a donor effect wearing a
            # subtype's name — and one LEC partition was exactly that.
            report.warnings.append(
                "CHOIR reduction unavailable: CHOIR will compute its own "
                "UNCORRECTED reduction, so the cluster count may be donor-driven."
            )
        partition_result = run_choir(
            filtered,
            sc_config,
            backend,
            scratch,
            reduction_key=partition_reduction_key,
        )

        if isinstance(partition_result, MethodSkip):
            # CHOIR unavailable: record skip and continue without labels. A
            # warning, because the config named CHOIR as the partition method
            # and the object comes out of this stage with no subcluster labels
            # at all — the same shape as a run that never asked for any.
            report.warnings.append(f"CHOIR partition skipped: {partition_result.reason}")
            report.metrics["partition_skipped"] = True
            report.metrics["partition_skip_reason"] = partition_result.reason
        else:
            # CHOIR succeeded: update filtered with labeled adata.
            filtered = partition_result
            n_subclusters = filtered.obs[sc_config.key_added].nunique()
            report.notes.append(
                f"CHOIR partition: {n_subclusters} subclusters identified "
                f"(alpha={sc_config.partition.choir.get('alpha', 0.05)})"
            )
            report.metrics["n_subclusters"] = int(n_subclusters)
            report.metrics["partition_method"] = "choir"

            # Run formal significance test (sc-SHC).
            if sc_config.formal_test.method == "scshc":
                test_result = run_scshc_test(
                    filtered,
                    sc_config.key_added,
                    sc_config,
                    backend,
                    scratch,
                    batch_key=nuisance_key,
                )

                if isinstance(test_result, MethodSkip):
                    # A warning: sc-SHC is the test that says which splits are
                    # real, so without it the subclusters ship unvalidated.
                    report.warnings.append(f"sc-SHC test skipped: {test_result.reason}")
                    report.metrics["formal_test_skipped"] = True
                else:
                    # sc-SHC succeeded: record significance in metrics + uns.
                    n_sig = test_result["n_significant"]
                    n_tested = test_result["n_splits_tested"]
                    report.notes.append(
                        f"sc-SHC test: {n_sig}/{n_tested} splits significant "
                        f"(alpha={test_result['alpha']})"
                    )
                    report.metrics["formal_test_n_significant"] = n_sig
                    report.metrics["formal_test_n_splits"] = n_tested

                    # The reconciled partition, when the R script produced one.
                    surviving = test_result.get("n_labels_surviving")
                    if surviving is not None:
                        report.notes.append(
                            f"sc-SHC reconciled partition: {surviving} of "
                            f"{n_subclusters} CHOIR clusters survive as distinct "
                            f"(labels in obs['{test_result['labels_key']}'])"
                        )
                        report.metrics["formal_test_n_labels_surviving"] = surviving

                    # A warning when the test does not uphold the partition.
                    # This has to be loud: every downstream per-subcluster
                    # result — abundance swaps, per-subtype effect sizes — is
                    # keyed on labels the engine's own significance test has
                    # just declined to defend, and "0/1 splits significant"
                    # in the notes is not a sentence anyone reads. A run whose
                    # eight LEC subclusters all collapsed to one shipped a
                    # headline per-subtype table without this warning.
                    if test_result.get("merged_to_one"):
                        report.warnings.append(
                            f"sc-SHC merged all {n_subclusters} subclusters into ONE: "
                            f"the partition is NOT statistically supported at "
                            f"alpha={test_result['alpha']} (conditioned on "
                            f"{test_result['batch_key']}). Whole-lineage results are "
                            "unaffected; every per-subcluster result is exploratory "
                            "and must be reported as such."
                        )
                    elif n_sig == 0 and n_tested > 0:
                        report.warnings.append(
                            f"sc-SHC found 0 of {n_tested} tested splits significant at "
                            f"alpha={test_result['alpha']}: the CHOIR partition is not "
                            "upheld by the formal test."
                        )

                    # Store full test results in uns.
                    if "subclustering" not in filtered.uns:
                        filtered.uns["subclustering"] = {}
                    filtered.uns["subclustering"]["formal_test"] = test_result

        return filtered, partition_reduction_key

    @staticmethod
    def _partition_leiden_stability(
        context: PipelineContext,
        filtered: ad.AnnData,
        sc_config: SubclusteringConfig,
        nuisance_key: str | None,
        report: _Report,
    ) -> ad.AnnData:
        """Partition with bootstrap-validated Leiden resolution selection (not a CHOIR fallback)."""

        # An alternative to CHOIR, not a fallback: bootstrap-validated Leiden
        # resolution selection, for when CHOIR is unavailable, fails outright,
        # or its own sc-SHC test rejects the partition it proposes -- three
        # distinct failure modes this project hit in practice on real data.
        reduction_key = reembed_focus_batch_aware(
            filtered,
            counts_layer=sc_config.counts_layer,
            batch_key=nuisance_key,
            random_state=(sc_config.partition.seeds[0] if sc_config.partition.seeds else 0),
            max_iter_harmony=sc_config.reembed.max_iter_harmony,
            diagnostics=report.warnings,
        )
        if reduction_key:
            report.notes.append(f"leiden_stability reduction: {reduction_key}")
            report.metrics["leiden_stability_reduction"] = reduction_key
        else:
            report.warnings.append(
                "leiden_stability reduction unavailable: clustering will use an "
                "UNCORRECTED reduction, so the resolution choice may be donor-driven."
            )
            reduction_key = "X_pca"

        partition_result = run_leiden_stability(
            filtered,
            sc_config,
            reduction_key=reduction_key,
            context=context,
        )

        if isinstance(partition_result, MethodSkip):
            report.warnings.append(f"leiden_stability partition skipped: {partition_result.reason}")
            report.metrics["partition_skipped"] = True
            report.metrics["partition_skip_reason"] = partition_result.reason
        else:
            filtered = partition_result
            n_subclusters = filtered.obs[sc_config.key_added].nunique()
            chosen = filtered.uns["subclustering"]["leiden_stability"]["chosen_resolution"]
            report.notes.append(
                f"leiden_stability partition: {n_subclusters} subclusters identified "
                f"(bootstrap-selected resolution={chosen})"
            )
            report.metrics["n_subclusters"] = int(n_subclusters)
            report.metrics["partition_method"] = "leiden_stability"
            report.metrics["leiden_stability_resolution"] = chosen

        return filtered

    def _run_donor_gate(
        self,
        context: PipelineContext,
        config: Any,
        sc_config: SubclusteringConfig,
        filtered: ad.AnnData,
        partition_reduction_key: str | None,
        cluster_key: str,
        has_cluster_labels: bool,
        report: _Report,
    ) -> ad.AnnData:
        """Test whether any cluster is really one donor; flag or drop failures; plot the panel."""

        # Task 3: donor_gate + diagnostics.
        # Run donor gate if group_key set AND cluster labels exist. The donor
        # group_key falls back to the cohort donor_key so a dataset declares its
        # donor column once.
        from cellquorum.config.cohort import resolve_cohort_key

        group_key = resolve_cohort_key(
            config, attr="donor_key", stage_value=sc_config.donor_gate.group_key
        )

        # The donor gate tests whether a cluster's identity is really just donor
        # identity -- the exact same question CHOIR's batch-corrected reduction
        # exists to answer honestly (see the "choir" branch above: an uncorrected
        # embedding still encodes the donor signal the gate is supposed to catch,
        # which is how one LEC partition was 98% a single donor and still
        # "passed"). So reuse CHOIR's corrected reduction when one was computed,
        # rather than silently deriving a separate UNCORRECTED PCA for a test
        # whose whole purpose is detecting donor-driven clusters. Only when no
        # corrected reduction exists (a non-CHOIR partition method, or CHOIR's
        # reduction was unavailable) does this fall back to deriving a fresh
        # "X_pca" -- extract_focus deleted the parent's X_* embeddings, so
        # something must exist for the gate to test on.
        donor_gate_embedding_key = partition_reduction_key or "X_pca"
        embedding_ready = False
        if group_key is not None and has_cluster_labels:
            embedding_ready = ensure_focus_embedding(
                filtered,
                counts_layer=sc_config.counts_layer,
                embedding_key=donor_gate_embedding_key,
                reembed=sc_config.reembed,
                random_state=0,
            )
            if not embedding_ready:
                report.notes.append(
                    "Donor gate: skipped (could not derive an embedding for the "
                    "focus subset; too few cells or genes)."
                )
                report.warnings.append("Donor gate skipped: focus subset could not be re-embedded.")

        if group_key is not None and has_cluster_labels and embedding_ready:
            # Run donor reproducibility gatekeeper.
            gate_result = donor_reproducibility(
                filtered,
                cluster_key=cluster_key,
                group_key=group_key,
                min_groups=sc_config.donor_gate.min_groups,
                min_cells_per_group=sc_config.donor_gate.min_cells_per_group,
                max_group_frac=sc_config.donor_gate.max_group_frac,
                do_lodo=sc_config.donor_gate.leave_one_donor_out,
                do_classifier=sc_config.donor_gate.classifier_separability,
                embedding_key=donor_gate_embedding_key,
                random_state=0,
            )
            report.metrics["donor_gate_embedding_key"] = donor_gate_embedding_key

            # Record summary metrics.
            n_pass = gate_result["summary"]["n_pass"]
            n_fail = gate_result["summary"]["n_fail"]
            report.notes.append(
                f"Donor gate: {n_pass} clusters PASS, "
                f"{n_fail} clusters FAIL (min_groups={sc_config.donor_gate.min_groups})"
            )
            report.metrics["donor_gate_n_pass"] = n_pass
            report.metrics["donor_gate_n_fail"] = n_fail

            if "subclustering" not in filtered.uns:
                filtered.uns["subclustering"] = {}
            filtered.uns["subclustering"]["donor_gate"] = gate_result

            # Generate QC panel plot.
            paths = getattr(context, "paths", None)
            if paths is not None:
                figures_dir = Path(getattr(paths, "figures", "."))
                qc_panel_path = figures_dir / "subclustering_donor_qc_panel.png"
                plot_subcluster_qc_panel(gate_result, qc_panel_path)
                report.artifacts.append(
                    StageArtifact(
                        name="subclustering_donor_qc_panel",
                        path=qc_panel_path,
                        kind="png",
                        description="Donor-reproducibility QC panel",
                    )
                )

            # Apply QC flags to obs.
            apply_qc_flags(
                filtered,
                cluster_key,
                gate_result,
                key_added="donor_qc",
            )

            # Apply action (flag or drop).
            if sc_config.action == "drop":
                # Drop cells in failed clusters.
                n_cells_before = filtered.n_obs
                filtered = filtered[filtered.obs["donor_qc_qc_pass"]].copy()
                n_cells_after = filtered.n_obs
                n_dropped = n_cells_before - n_cells_after
                report.notes.append(f"Action=drop: removed {n_dropped} cells in failed clusters")
                report.warnings.append(f"Dropped {n_dropped} cells (action='drop' configured)")
                report.metrics["donor_gate_n_cells_dropped"] = n_dropped
            else:
                # Default: flag-not-drop.
                report.notes.append("Action=flag: retained all cells with QC flags")
        elif group_key is None:
            report.notes.append("Donor gate: skipped (group_key not configured)")
        elif not has_cluster_labels:
            report.notes.append("Donor gate: skipped (no cluster labels)")
        # The embedding-failed case (group_key set + labels present but
        # embedding_ready is False) already recorded its own skip note above.

        return filtered

    @staticmethod
    def _run_technical_qc(
        context: PipelineContext,
        sc_config: SubclusteringConfig,
        filtered: ad.AnnData,
        cluster_key: str,
        has_cluster_labels: bool,
        report: _Report,
    ) -> ad.AnnData:
        """Compare each subcluster's QC evidence to its siblings and flag or drop the outliers."""

        # Technical-QC consistency: does any subcluster's QC evidence (mito%,
        # counts, complexity, multiplet rate) look like an outlier against its
        # OWN lineage's other subclusters? Global QC runs once, across every
        # cell type, in filter mode -- a threshold permissive enough for a
        # naturally small/RNA-poor cell type will let a degraded cell of a
        # normally RNA-rich lineage (e.g. fibroblasts) through uncaught. This
        # does not re-threshold anything (a per-lineage `report_only` QC pass
        # was tried and rejected for exactly that reason); it only compares a
        # lineage's own subclusters to each other using evidence the global
        # pass already computed. See qc_consistency.py.
        if sc_config.technical_qc.enabled and has_cluster_labels:
            qc_result = subcluster_qc_consistency(
                filtered,
                cluster_key=cluster_key,
                z_threshold=sc_config.technical_qc.z_threshold,
            )
            if qc_result["summary"]["skipped"]:
                report.notes.append(
                    "Technical-QC consistency: skipped (no QC evidence columns present)"
                )
            else:
                n_pass = qc_result["summary"]["n_pass"]
                n_fail = qc_result["summary"]["n_fail"]
                report.notes.append(
                    f"Technical-QC consistency: {n_pass} subclusters within normal range, "
                    f"{n_fail} flagged as outliers vs. their siblings "
                    f"(z_threshold={sc_config.technical_qc.z_threshold})"
                )
                report.metrics["technical_qc_n_pass"] = n_pass
                report.metrics["technical_qc_n_fail"] = n_fail

                if "subclustering" not in filtered.uns:
                    filtered.uns["subclustering"] = {}
                filtered.uns["subclustering"]["technical_qc_consistency"] = qc_result

                if n_fail:
                    failed = {
                        cid: c["qc_reason"]
                        for cid, c in qc_result["clusters"].items()
                        if not c["qc_pass"]
                    }
                    report.warnings.append(
                        "Technical-QC consistency: subcluster(s) "
                        f"{sorted(failed)} look like QC-confounded populations, not "
                        f"necessarily real biological states -- {failed}"
                    )

                    # Apply action (flag or drop) -- same shared switch the donor
                    # gate already honors ("action when clusters fail gate"). A
                    # QC-confounded cluster left in place is not just mislabeled:
                    # every downstream stage keyed on cluster_key (CCC networks,
                    # coexpression, GRN, enrichment) treats it as an ordinary peer.
                    apply_qc_flags(filtered, cluster_key, qc_result, key_added="technical_qc")
                    if sc_config.action == "drop":
                        n_cells_before = filtered.n_obs
                        filtered = filtered[filtered.obs["technical_qc_qc_pass"]].copy()
                        n_cells_after = filtered.n_obs
                        n_dropped = n_cells_before - n_cells_after
                        report.notes.append(
                            f"Action=drop: removed {n_dropped} cells in "
                            f"QC-confounded clusters {sorted(failed)}"
                        )
                        report.warnings.append(
                            f"Dropped {n_dropped} cells in QC-confounded clusters "
                            "(action='drop' configured)"
                        )
                        report.metrics["technical_qc_n_cells_dropped"] = n_dropped
                    else:
                        report.notes.append(
                            "Action=flag: retained all cells despite QC-confounded clusters"
                        )

                paths = getattr(context, "paths", None)
                if paths is not None:
                    figures_dir = Path(getattr(paths, "figures", "."))
                    qc_consistency_path = figures_dir / "subclustering_technical_qc_consistency.png"
                    plot_technical_qc_consistency(qc_result, qc_consistency_path)
                    if qc_consistency_path.exists():
                        report.artifacts.append(
                            StageArtifact(
                                name="subclustering_technical_qc_consistency",
                                path=qc_consistency_path,
                                kind="png",
                                description=(
                                    "Per-subcluster technical-QC consistency vs. siblings"
                                ),
                            )
                        )

                    rows = []
                    for cid, c in qc_result["clusters"].items():
                        row = {"cluster": cid, "n_cells": c["n_cells"], "qc_pass": c["qc_pass"]}
                        for metric_name, metric_vals in c["metrics"].items():
                            row[f"{metric_name}_value"] = metric_vals["value"]
                            row[f"{metric_name}_z"] = metric_vals["z"]
                        rows.append(row)
                    if rows:
                        import pandas as pd

                        writer = StageArtifactWriter.from_context(context)
                        report.artifacts.append(
                            writer.table(
                                pd.DataFrame(rows),
                                "subclustering_technical_qc_consistency.csv",
                                name="subclustering_technical_qc_consistency_table",
                                description=(
                                    "Per-subcluster QC evidence values and z-scores vs. siblings"
                                ),
                            )
                        )
        elif sc_config.technical_qc.enabled and not has_cluster_labels:
            report.notes.append("Technical-QC consistency: skipped (no cluster labels)")

        return filtered

    @staticmethod
    def _add_marker_dotplot(
        context: PipelineContext,
        sc_config: SubclusteringConfig,
        filtered: ad.AnnData,
        cluster_key: str,
        has_cluster_labels: bool,
        report: _Report,
    ) -> None:
        """Draw the marker identity dotplot on the final cluster set; a failure only warns."""

        # Marker-gene identity dotplot: auto-derived top markers per cluster,
        # on the FINAL (post-drop) cluster set -- "what is this cluster,"
        # complementing the technical-QC/donor-gate panels' "is it
        # trustworthy." Skipped gracefully (returns a non-existent path) when
        # fewer than 2 clusters survive or the configured layer is absent.
        if (
            sc_config.diagnostics.marker_dotplot
            and has_cluster_labels
            and cluster_key in filtered.obs.columns
            and sc_config.diagnostics.marker_dotplot_layer in filtered.layers
        ):
            paths = getattr(context, "paths", None)
            if paths is not None:
                figures_dir = Path(getattr(paths, "figures", "."))
                dotplot_path = figures_dir / "subclustering_marker_identity_dotplot.png"
                try:
                    plot_marker_identity_dotplot(
                        filtered,
                        cluster_key=cluster_key,
                        layer=sc_config.diagnostics.marker_dotplot_layer,
                        out_path=dotplot_path,
                        n_top=sc_config.diagnostics.marker_dotplot_n_top,
                        cluster_labels=sc_config.diagnostics.marker_dotplot_cluster_labels,
                    )
                except Exception as exc:  # noqa: BLE001 -- a figure must never sink the stage
                    report.warnings.append(f"marker identity dotplot failed: {str(exc)[:200]}")
                else:
                    if dotplot_path.exists():
                        report.artifacts.append(
                            StageArtifact(
                                name="subclustering_marker_identity_dotplot",
                                path=dotplot_path,
                                kind="png",
                                description=(
                                    "Auto-derived top marker genes per subcluster "
                                    "(fraction expressing x mean expression)"
                                ),
                            )
                        )

    @staticmethod
    def _resolve_nuisance_key(config: Any, sc_config: SubclusteringConfig) -> str | None:
        """
        Resolve the one column CHOIR corrects for and sc-SHC conditions on.

        Both need the same thing: the technical grouping that must not be reported
        as biological structure. Precedence is ``cohort.batch_key``, then
        ``cohort.donor_key``, then the subclustering block's own
        ``donor_gate.group_key``.

        Reading it straight off ``donor_gate.group_key``, as this stage used to,
        broke the declare-once contract at two of three call sites: a config that
        set ``cohort.donor_key`` and left the donor-gate key unset — the pattern
        the donor gate itself documents — got a *correctly* donor-gated run whose
        CHOIR embedding was never corrected and whose sc-SHC was never
        conditioned. Nothing in the output said so.
        """
        from cellquorum.config.cohort import resolve_cohort_key

        stage_value = getattr(getattr(sc_config, "donor_gate", None), "group_key", None)
        batch = resolve_cohort_key(config, attr="batch_key", stage_value=None)
        if batch is not None:
            return batch
        return resolve_cohort_key(config, attr="donor_key", stage_value=stage_value)

    def _project_onto_parent(
        self,
        parent: ad.AnnData,
        focused: ad.AnnData,
        filtered: ad.AnnData,
        sc_config: SubclusteringConfig,
    ) -> ad.AnnData:
        """
        Merge subclustering outputs back onto the parent object.

        Args:
            parent: The pipeline's working AnnData (embeddings intact, all cells).
            focused: The focus subset (carries subcluster_extraction provenance).
            filtered: The analyzed subset (carries subcluster labels + QC flags).
            sc_config: Resolved subclustering config.

        Returns:
            The parent object with subcluster labels / QC flags projected on, and
            (for action='drop') restricted to the cells the donor gate retained.
        """
        # Project the per-cell subcluster labels back onto the parent by index.
        # Cells not present in `filtered` (outside focus or group-filtered out)
        # receive NaN — they were not part of the subclustering analysis.
        key = sc_config.key_added
        if key in filtered.obs.columns:
            parent.obs[key] = self._project_labels(filtered.obs[key], parent.obs_names)

        # Project the sc-SHC reconciled labels the same way. They are the partition
        # the formal test upholds, so a downstream stage that wants a defensible
        # grouping needs them on the object it actually receives.
        scshc_key = f"{key}_scshc"
        if scshc_key in filtered.obs.columns:
            parent.obs[scshc_key] = self._project_labels(filtered.obs[scshc_key], parent.obs_names)

        # Project donor-QC flags the same way when the gate produced them.
        for col in ("donor_qc_qc_pass", "donor_qc_qc_reason"):
            if col in filtered.obs.columns:
                parent.obs[col] = self._project_labels(filtered.obs[col], parent.obs_names)

        # Carry provenance forward on the parent's uns.
        if "subcluster_extraction" in focused.uns:
            parent.uns["subcluster_extraction"] = focused.uns["subcluster_extraction"]
        if "subclustering" in filtered.uns:
            parent.uns["subclustering"] = filtered.uns["subclustering"]

        # action='drop' is an explicit removal of gate-failed cells. Honor it by
        # restricting the parent to the cells that survived in `filtered`
        # (embeddings preserved, unlike returning the stripped subset).
        if sc_config.action == "drop":
            keep = parent.obs_names.isin(filtered.obs_names)
            return parent[keep].copy()

        return parent

    @staticmethod
    def _project_labels(
        series: pd.Series, index: pd.Index
    ) -> pd.Series | pd.api.extensions.ExtensionArray:
        """Reindex a per-cell column onto a larger axis, keeping it writable.

        Widening a column has to put something in the gap, and pandas puts float
        NaN there whatever the column held. For a BOOLEAN column that silently
        produces ``object`` holding ``{True, False, nan}``, which h5py cannot
        encode: ``obs['donor_qc_qc_pass']`` in exactly that state made every h5ad
        write of one LEC run raise ``TypeError: Can't implicitly convert non-string
        objects to strings``, costing that run its final object, its checkpoints
        and its velocity h5ads — and with them CellRank's velocity kernel and the
        CytoTRACE kernel, which read files that were never written.

        Nullable ``boolean`` is the honest destination: the widened column really
        is three-valued (passed / failed / never assessed), and filling the gap
        with False would assert that cells outside the analysis passed a gate they
        never entered. Label columns become categoricals for the same reason —
        missing stays missing rather than becoming the string ``"nan"``.
        """
        import pandas as pd

        widened = series.reindex(index)
        if series.dtype == bool or pd.api.types.is_bool_dtype(series.dtype):
            return pd.array([None if pd.isna(v) else bool(v) for v in widened], dtype="boolean")
        if isinstance(series.dtype, pd.CategoricalDtype):
            return widened
        # Tested for string-ness, not for `object`.
        #
        # The check used to be `series.dtype != object`, which meant "only convert a
        # plain-object column". pandas 3 gives string columns a `StringDtype`, so every label
        # column started taking the early return and came back as strings rather than the
        # categorical this function documents and scanpy expects — `.cat.categories` on a
        # projected label then raises, and a groupby over it carries no level order.
        if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
            return pd.Categorical(widened)
        return widened

    def _resolve_rscript_backend(self, context: PipelineContext) -> Any | None:
        """Return the Rscript backend from context registry, or None."""
        registry = getattr(context, "backend_registry", None)
        if registry is None:
            return None
        try:
            return registry.get("rscript")
        except Exception:
            return None


__all__ = ["SubclusteringStage"]
