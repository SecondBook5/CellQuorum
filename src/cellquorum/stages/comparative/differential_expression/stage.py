# Pipeline step (order=210): differential_expression — test expression shifts (DE methods).
"""Differential-expression stage: dispatches to the configured DE method."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Import the package so the method registers itself as a side effect.
import cellquorum.stages.comparative.differential_expression  # noqa: F401
from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.exceptions import CellQuorumConfigError
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.stage_base import MethodDispatchStage, _apply_cohort_overlay


@register_stage(
    name="differential_expression",
    order=210,
    config_flag="differential_expression",
    config_field="differential_expression",
    category="differential_expression",
)
class DifferentialExpressionStage(MethodDispatchStage):
    """Run the configured pseudobulk differential-expression method."""

    def _select_method_name(self, config: dict) -> str:
        """Return the configured DE method name."""

        # Default to the donor-blocked pseudobulk edgeR method.
        return config.get("method", "pseudobulk_edger")

    def run(self, context: object) -> StageResult:
        """Run the configured DE method, pooled and (if configured) per subcluster.

        ``cell_type_col`` unset: identical to the base dispatch (one pooled
        comparison). Set: ALSO runs the same method within each group of
        that column, skipping (not failing the stage on) any group without
        enough donor replication for its own fit, and writes a combined
        long-format table (a ``cell_type`` column identifies each row's
        group) alongside the pooled result -- a pooled-only comparison
        cannot tell "this gene changed within a state" apart from "the mix
        of states shifted between conditions". See DifferentialExpressionConfig.
        """
        from cellquorum.methods.context_access import resolve_stage_config

        adata = context.require_adata()
        stage_config = resolve_stage_config(context, self.name)
        stage_config = _apply_cohort_overlay(context, stage_config)
        stage_config = self._augment_config(context, stage_config)

        if not stage_config.get("enabled", True):
            return StageResult.skipped(
                adata=adata,
                reason="disabled by config",
                warnings=[f"{self.name} disabled by config"],
            )

        cell_type_col = stage_config.get("cell_type_col")
        if not cell_type_col or cell_type_col not in adata.obs.columns:
            outcome = self._run_single_method(adata, stage_config, context)
            if isinstance(outcome, MethodSkip):
                return StageResult.skipped(
                    adata=adata,
                    reason=outcome.reason,
                    warnings=[outcome.reason],
                    metrics=outcome.details,
                )
            self._validate_output(outcome)
            return outcome

        results_dir = Path(context.paths.results)
        pooled_csv = results_dir / "de_pseudobulk_edger.csv"

        notes: list[str] = []
        warnings: list[str] = []
        artifacts: list[StageArtifact] = []
        per_group_frames: list = []

        groups = sorted(str(g) for g in adata.obs[cell_type_col].dropna().unique())
        for group in groups:
            mask = (adata.obs[cell_type_col].astype(str) == group).to_numpy()
            subset = adata[mask].copy()
            try:
                outcome = self._run_single_method(subset, stage_config, context)
            except CellQuorumConfigError as exc:
                # The pre-flight estimability gate fails LOUD by design when the
                # DECLARED comparison is broken -- correct for the pooled run,
                # where that means the run's own config is wrong. Within one
                # subcluster it more often just means that subcluster happens to
                # be missing a whole condition arm (a small group can easily
                # have zero cells from one donor in one condition by chance),
                # which is a fact about that subcluster, not a fatal config
                # error for the whole stage.
                warnings.append(f"differential_expression: subcluster '{group}' skipped: {exc}")
                continue
            if isinstance(outcome, MethodSkip):
                warnings.append(
                    f"differential_expression: subcluster '{group}' skipped: {outcome.reason}"
                )
                continue
            if pooled_csv.exists():
                group_df = pd.read_csv(pooled_csv)
                group_df.insert(0, "cell_type", group)
                per_group_frames.append(group_df)
            notes.extend(outcome.notes)

        if per_group_frames:
            combined = pd.concat(per_group_frames, ignore_index=True)
            combined_path = results_dir / "de_pseudobulk_edger_by_subcluster.csv"
            combined.to_csv(combined_path, index=False)
            artifacts.append(
                StageArtifact(
                    name="de_results_by_subcluster",
                    path=combined_path,
                    kind="csv",
                    description="Pseudobulk edgeR DE, stratified by subcluster.",
                )
            )
            notes.append(
                f"Per-subcluster DE: {len(per_group_frames)}/{len(groups)} subclusters "
                "produced results."
            )

        # Pooled last, on the full object: its natural output IS the canonical
        # de_pseudobulk_edger.csv, so it must not be overwritten by a group run.
        pooled_outcome = self._run_single_method(adata, stage_config, context)
        metrics: dict[str, object] = {}
        if isinstance(pooled_outcome, MethodSkip):
            warnings.append(
                f"differential_expression: pooled comparison skipped: {pooled_outcome.reason}"
            )
            result_adata = adata
        else:
            notes.extend(pooled_outcome.notes)
            artifacts.extend(pooled_outcome.artifacts)
            metrics.update(pooled_outcome.metrics)
            result_adata = pooled_outcome.adata

        result = StageResult(
            adata=result_adata,
            artifacts=artifacts,
            notes=notes,
            warnings=warnings,
            metrics=metrics,
        )
        self._validate_output(result)
        return result

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        """
        Bridge the project-level ``design`` block into the DE method's config.

        The biological question (donor/condition columns, case/control tokens,
        pairing) is declared once in ``config.design``; the pseudobulk method
        reads it from its own config dict. This resolves each key with the
        established precedence — an explicit stage value wins, then the central
        cohort key (structural columns only), then the design block, then the
        method's own default — so nothing here is study-specific. When no design
        is present the case/control keys stay unset and the method records a
        clean skip rather than crashing.

        Args:
            context: Pipeline context exposing ``config`` (and its ``design``).
            stage_config: The resolved (cohort-overlaid) DE config dict.

        Returns:
            A copy of the config with design-derived keys filled in.
        """

        config = getattr(context, "config", None)
        design = getattr(config, "design", None)
        if design is None:
            return stage_config

        augmented = dict(stage_config)

        # Structural obs columns: stage value → cohort key → design → default.
        if not augmented.get("donor_col"):
            augmented["donor_col"] = resolve_cohort_key(
                config,
                attr="donor_key",
                stage_value=getattr(design, "donor_col", "patient_id"),
            )
        if not augmented.get("condition_col"):
            augmented["condition_col"] = resolve_cohort_key(
                config,
                attr="condition_key",
                stage_value=getattr(design, "condition_col", "condition"),
            )

        # The comparison itself: stage value → design (no cohort equivalent).
        if not augmented.get("case"):
            augmented["case"] = getattr(design, "case", None)
        if not augmented.get("control"):
            augmented["control"] = getattr(design, "control", None)
        if "paired" not in augmented:
            augmented["paired"] = bool(getattr(design, "paired", False))

        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No structural postcondition on obs/var; DE writes a table artifact."""
