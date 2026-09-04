# Pipeline step (order=100): subclustering — resolve finer structure within annotated clusters.
"""Subclustering stage implementation."""

from __future__ import annotations

from pathlib import Path

from cellquorum.core.exceptions import CellQuorumConfigError
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.clustering.subclustering.diagnostics import (
    plot_group_recovery,
    plot_subcluster_qc_panel,
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
from cellquorum.stages.clustering.subclustering.partition import run_choir, run_scshc_test


class SubclusteringFocusError(CellQuorumConfigError):
    """The configured focus selects no cells, so there is nothing to subcluster."""


@register_stage(
    name="subclustering",
    order=100,
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

    def run(self, context: object) -> StageResult:
        """
        Run principled subclustering pipeline.

        Args:
            context: Pipeline context exposing config, adata, paths.

        Returns:
            StageResult with focused/filtered subset as adata.
        """
        # Resolve config and adata.
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

        # Extract focus lineage.
        focused = extract_focus(
            adata,
            focus_config,
            sc_config.counts_layer,
        )

        notes = []
        warnings = []
        artifacts = []
        metrics = {}

        # Record focus extraction provenance.
        if "subcluster_extraction" in focused.uns:
            prov = focused.uns["subcluster_extraction"]
            notes.append(
                f"Focus extracted: {prov['n_cells_kept']} / "
                f"{prov['n_cells_total']} cells "
                f"({prov['label_key']} in {prov['labels']})"
            )
            metrics["n_cells_before_focus"] = prov["n_cells_total"]
            metrics["n_cells_after_focus"] = prov["n_cells_kept"]

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
                notes.append(
                    f"Group filter: kept {n_kept} groups, "
                    f"dropped {n_dropped} groups "
                    f"(< {min_cells} cells in {group_key})"
                )
                if filter_prov["dropped"]:
                    warnings.append(f"Dropped groups: {', '.join(filter_prov['dropped'])}")
                metrics["n_groups_kept"] = n_kept
                metrics["n_groups_dropped"] = n_dropped
                metrics["n_cells_after_group_filter"] = filtered.n_obs

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
                    artifacts.append(
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
                notes.append("Group filter: skipped (not configured)")
        else:
            filtered = focused
            notes.append("Group filter: skipped (not configured)")

        # Task 2: partition + formal_test.
        # Resolve backend from context.
        backend = self._resolve_rscript_backend(context)
        scratch = Path(getattr(getattr(context, "paths", None), "scratch", "."))

        # The nuisance variable CHOIR corrects for and sc-SHC conditions on.
        nuisance_key = self._resolve_nuisance_key(config, sc_config)
        if nuisance_key is None:
            warnings.append(
                "Subclustering has no batch/donor key to correct for: declare "
                "cohort.batch_key or cohort.donor_key. CHOIR will partition an "
                "uncorrected embedding, so its cluster count may be donor-driven."
            )
        else:
            notes.append(f"Batch correction / conditioning key: {nuisance_key}")
            metrics["nuisance_key"] = nuisance_key

        # Run partition (CHOIR or fallback).
        if sc_config.partition.method == "choir":
            # Batch-aware re-embedding must happen BEFORE partitioning: CHOIR
            # needs it to cluster on a corrected space, and the donor-gate
            # embedding below is computed too late (and uncorrected) to serve.
            choir_reduction_key = reembed_focus_batch_aware(
                filtered,
                counts_layer=sc_config.counts_layer,
                batch_key=nuisance_key,
                random_state=(sc_config.partition.seeds[0] if sc_config.partition.seeds else 0),
                max_iter_harmony=sc_config.reembed.max_iter_harmony,
                diagnostics=warnings,
            )
            if choir_reduction_key:
                notes.append(f"CHOIR reduction: {choir_reduction_key}")
                metrics["choir_reduction"] = choir_reduction_key
            else:
                # A warning, not a note. CHOIR declares its cluster count
                # significant *in the space it was given*, so an uncorrected
                # space means the count can be a donor effect wearing a
                # subtype's name — and one LEC partition was exactly that.
                warnings.append(
                    "CHOIR reduction unavailable: CHOIR will compute its own "
                    "UNCORRECTED reduction, so the cluster count may be donor-driven."
                )
            partition_result = run_choir(
                filtered,
                sc_config,
                backend,
                scratch,
                reduction_key=choir_reduction_key,
            )

            if isinstance(partition_result, MethodSkip):
                # CHOIR unavailable: record skip and continue without labels. A
                # warning, because the config named CHOIR as the partition method
                # and the object comes out of this stage with no subcluster labels
                # at all — the same shape as a run that never asked for any.
                warnings.append(f"CHOIR partition skipped: {partition_result.reason}")
                metrics["partition_skipped"] = True
                metrics["partition_skip_reason"] = partition_result.reason
            else:
                # CHOIR succeeded: update filtered with labeled adata.
                filtered = partition_result
                n_subclusters = filtered.obs[sc_config.key_added].nunique()
                notes.append(
                    f"CHOIR partition: {n_subclusters} subclusters identified "
                    f"(alpha={sc_config.partition.choir.get('alpha', 0.05)})"
                )
                metrics["n_subclusters"] = int(n_subclusters)
                metrics["partition_method"] = "choir"

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
                        warnings.append(f"sc-SHC test skipped: {test_result.reason}")
                        metrics["formal_test_skipped"] = True
                    else:
                        # sc-SHC succeeded: record significance in metrics + uns.
                        n_sig = test_result["n_significant"]
                        n_tested = test_result["n_splits_tested"]
                        notes.append(
                            f"sc-SHC test: {n_sig}/{n_tested} splits significant "
                            f"(alpha={test_result['alpha']})"
                        )
                        metrics["formal_test_n_significant"] = n_sig
                        metrics["formal_test_n_splits"] = n_tested

                        # The reconciled partition, when the R script produced one.
                        surviving = test_result.get("n_labels_surviving")
                        if surviving is not None:
                            notes.append(
                                f"sc-SHC reconciled partition: {surviving} of "
                                f"{n_subclusters} CHOIR clusters survive as distinct "
                                f"(labels in obs['{test_result['labels_key']}'])"
                            )
                            metrics["formal_test_n_labels_surviving"] = surviving

                        # A warning when the test does not uphold the partition.
                        # This has to be loud: every downstream per-subcluster
                        # result — abundance swaps, per-subtype effect sizes — is
                        # keyed on labels the engine's own significance test has
                        # just declined to defend, and "0/1 splits significant"
                        # in the notes is not a sentence anyone reads. A run whose
                        # eight LEC subclusters all collapsed to one shipped a
                        # headline per-subtype table without this warning.
                        if test_result.get("merged_to_one"):
                            warnings.append(
                                f"sc-SHC merged all {n_subclusters} subclusters into ONE: "
                                f"the partition is NOT statistically supported at "
                                f"alpha={test_result['alpha']} (conditioned on "
                                f"{test_result['batch_key']}). Whole-lineage results are "
                                "unaffected; every per-subcluster result is exploratory "
                                "and must be reported as such."
                            )
                        elif n_sig == 0 and n_tested > 0:
                            warnings.append(
                                f"sc-SHC found 0 of {n_tested} tested splits significant at "
                                f"alpha={test_result['alpha']}: the CHOIR partition is not "
                                "upheld by the formal test."
                            )

                        # Store full test results in uns.
                        if "subclustering" not in filtered.uns:
                            filtered.uns["subclustering"] = {}
                        filtered.uns["subclustering"]["formal_test"] = test_result
        elif sc_config.partition.method == "leiden_grid":
            # Minimal scanpy fallback (defer detailed implementation).
            notes.append("leiden_grid partition: deferred (CHOIR is primary method)")
            metrics["partition_skipped"] = True
            metrics["partition_skip_reason"] = "leiden_grid not implemented"
        else:
            notes.append(f"Unknown partition method: {sc_config.partition.method}")
            warnings.append(f"Unknown partition method: {sc_config.partition.method}")

        # Task 3: donor_gate + diagnostics.
        # Run donor gate if group_key set AND cluster labels exist. The donor
        # group_key falls back to the cohort donor_key so a dataset declares its
        # donor column once.
        from cellquorum.config.cohort import resolve_cohort_key

        group_key = resolve_cohort_key(
            config, attr="donor_key", stage_value=sc_config.donor_gate.group_key
        )
        cluster_key = sc_config.key_added
        has_cluster_labels = (
            cluster_key in filtered.obs.columns and filtered.obs[cluster_key].notna().any()
        )

        # The donor gate needs an embedding, but extract_focus deleted the
        # parent object's X_* embeddings and no full re-embedding runs yet.
        # Derive a minimal PCA on the focus subset so the gate is meaningful;
        # if the subset is too small/degenerate to embed, skip the gate with a
        # recorded note instead of crashing.
        embedding_ready = False
        if group_key is not None and has_cluster_labels:
            embedding_ready = ensure_focus_embedding(
                filtered,
                counts_layer=sc_config.counts_layer,
                embedding_key="X_pca",
                reembed=sc_config.reembed,
                random_state=0,
            )
            if not embedding_ready:
                notes.append(
                    "Donor gate: skipped (could not derive an embedding for the "
                    "focus subset; too few cells or genes)."
                )
                warnings.append("Donor gate skipped: focus subset could not be re-embedded.")

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
                embedding_key="X_pca",
                random_state=0,
            )

            # Record summary metrics.
            n_pass = gate_result["summary"]["n_pass"]
            n_fail = gate_result["summary"]["n_fail"]
            notes.append(
                f"Donor gate: {n_pass} clusters PASS, "
                f"{n_fail} clusters FAIL (min_groups={sc_config.donor_gate.min_groups})"
            )
            metrics["donor_gate_n_pass"] = n_pass
            metrics["donor_gate_n_fail"] = n_fail

            # Store gate results in uns.
            if "subclustering" not in filtered.uns:
                filtered.uns["subclustering"] = {}
            filtered.uns["subclustering"]["donor_gate"] = gate_result

            # Generate QC panel plot.
            paths = getattr(context, "paths", None)
            if paths is not None:
                figures_dir = Path(getattr(paths, "figures", "."))
                qc_panel_path = figures_dir / "subclustering_donor_qc_panel.png"
                plot_subcluster_qc_panel(gate_result, qc_panel_path)
                artifacts.append(
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
                notes.append(f"Action=drop: removed {n_dropped} cells in failed clusters")
                warnings.append(f"Dropped {n_dropped} cells (action='drop' configured)")
                metrics["donor_gate_n_cells_dropped"] = n_dropped
            else:
                # Default: flag-not-drop.
                notes.append("Action=flag: retained all cells with QC flags")
        elif group_key is None:
            # Skip donor gate: not configured.
            notes.append("Donor gate: skipped (group_key not configured)")
        elif not has_cluster_labels:
            # Skip donor gate: no cluster labels to gate.
            notes.append("Donor gate: skipped (no cluster labels)")
        # The embedding-failed case (group_key set + labels present but
        # embedding_ready is False) already recorded its own skip note above.

        # Generate clustree plot (if leiden_grid labels present).
        # Note: clustree expects multiple cluster columns with a common prefix.
        # This will be driven by leiden_grid labels when implemented.
        # For now, defer clustree generation to full leiden_grid implementation.
        if sc_config.diagnostics.clustree:
            notes.append("Clustree: deferred (leiden_grid not implemented)")

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
            artifacts=artifacts,
            notes=notes,
            warnings=warnings,
            metrics=metrics,
        )

    @staticmethod
    def _resolve_nuisance_key(config: object, sc_config: object) -> str | None:
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
        parent: object,
        focused: object,
        filtered: object,
        sc_config: object,
    ) -> object:
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
    def _project_labels(series: object, index: object) -> object:
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
        if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype != object:
            return widened
        return pd.Categorical(widened)

    def _resolve_rscript_backend(self, context: object) -> object | None:
        """Return the Rscript backend from context registry, or None."""
        registry = getattr(context, "backend_registry", None)
        if registry is None:
            return None
        try:
            return registry.get("rscript")
        except Exception:
            return None


__all__ = ["SubclusteringStage"]
