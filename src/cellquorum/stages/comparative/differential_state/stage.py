# Pipeline step (order=215): differential_state — donor-paired testing of state/program scores.
"""Differential-state stage: is a per-cell state/program score different by condition?"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.artifacts import ArtifactManager
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.context_access import resolve_stage_config
from cellquorum.stages.comparative.differential_state.compute import (
    drop_underpowered_donor_groups,
    resolve_score_columns,
)
from cellquorum.stages.comparative.differential_state.config import DifferentialStateConfig
from cellquorum.stats.donor_comparison import two_group_test_on_donor_medians

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


@register_stage(
    name="differential_state",
    order=215,
    config_flag="differential_state",
    config_field="differential_state",
    category="differential_state",
)
class DifferentialStateStage:
    """Donor-paired test of every resolved state/program score, case vs control.

    ``state_scoring`` writes per-cell program scores but never tests whether they
    differ between conditions; four separate ad hoc scripts in the Mehrara
    manuscript each reinvented that test slightly differently. This stage is the
    engine capability that replaces them, reusing the already-validated
    ``two_group_test_on_donor_medians`` rather than a new statistical
    implementation. One program failing to reach enough donors records a
    per-program "insufficient_donors" row, never a stage-wide crash.
    """

    def run(self, context: PipelineContext) -> StageResult:
        """
        Execute the differential-state stage.

        Args:
            context: Pipeline context with AnnData, config, and output paths.

        Returns:
            StageResult with unchanged AnnData plus a differential-state results table.
        """

        adata = context.require_adata()
        config = DifferentialStateConfig.model_validate(
            resolve_stage_config(context, "differential_state")
        )

        top_config = getattr(context, "config", None)
        design = getattr(top_config, "design", None)
        condition_key = config.condition_key or resolve_cohort_key(
            top_config,
            attr="condition_key",
            stage_value=getattr(design, "condition_col", "condition"),
        )
        donor_key = config.donor_key or resolve_cohort_key(
            top_config, attr="donor_key", stage_value=getattr(design, "donor_col", "patient_id")
        )
        case = config.case or getattr(design, "case", None)
        control = config.control or getattr(design, "control", None)

        if not case or not control:
            return StageResult.skipped(
                adata=adata,
                reason="case/control labels not set (config.differential_state or design)",
            )
        missing = [k for k in (condition_key, donor_key) if not k or k not in adata.obs.columns]
        if missing:
            return StageResult.skipped(
                adata=adata, reason=f"missing required obs column(s): {missing}"
            )
        assert condition_key is not None and donor_key is not None  # narrows for mypy

        columns = resolve_score_columns(
            adata.obs, explicit=config.obs_columns, prefix=config.score_prefix
        )
        if not columns:
            return StageResult.skipped(
                adata=adata, reason="no state/program score columns resolved"
            )

        rows = []
        for column in columns:
            frame = adata.obs[[column, condition_key, donor_key]].copy()
            frame = frame[frame[condition_key].isin([case, control])]
            frame = drop_underpowered_donor_groups(
                frame,
                donor_col=donor_key,
                condition_col=condition_key,
                min_cells=config.min_cells_per_donor,
            )
            result = two_group_test_on_donor_medians(
                frame,
                value_col=column,
                group_col=condition_key,
                donor_col=donor_key,
                group1=control,
                group2=case,
                paired=config.paired,
                min_donors=config.min_donors,
                # A donor thinned below min_cells_per_donor in only ONE arm is an
                # expected consequence of that gate, not a data-integrity problem
                # worth halting the whole stage over -- drop it from this program's
                # pair set rather than erroring (the function's stricter default).
                incomplete_pairs="drop",
            )
            if result is None:
                rows.append(
                    {
                        "program": column,
                        "effect": np.nan,
                        "p_value": np.nan,
                        "test": None,
                        "n_pairs": 0,
                        "status": "insufficient_donors",
                    }
                )
            else:
                rows.append(
                    {
                        "program": column,
                        "effect": result.effect,
                        "p_value": result.p_value,
                        "test": result.test,
                        "n_pairs": result.n_group1,
                        "status": "ok",
                    }
                )

        results = pd.DataFrame(rows)
        tested = results["status"] == "ok"
        results["padj"] = np.nan
        if tested.any():
            results.loc[tested, "padj"] = multipletests(
                results.loc[tested, "p_value"].to_numpy(), method=config.fdr_method
            )[1]
        results["significant"] = results["padj"] < config.fdr

        manager = ArtifactManager.from_root(context.paths.root)
        artifact = manager.write_dataframe(
            results,
            name="differential_state_results",
            relative_path=f"results/{config.output_prefix}_results.csv",
            description=f"Donor-paired {case} vs {control} test per state/program score.",
        )

        n_tested = int(tested.sum())
        n_significant = int(results["significant"].sum())
        return StageResult(
            adata=adata,
            artifacts=[artifact],
            notes=[
                f"differential_state tested {n_tested} program(s), "
                f"{n_significant} significant (FDR<{config.fdr})."
            ],
            warnings=[],
            metrics={
                "case": case,
                "control": control,
                "condition_key": condition_key,
                "donor_key": donor_key,
                "n_programs_resolved": len(columns),
                "n_programs_tested": n_tested,
                "n_significant": n_significant,
            },
        )


__all__ = ["DifferentialStateStage"]
