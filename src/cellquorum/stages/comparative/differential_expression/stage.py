"""Differential-expression stage: dispatches to the configured DE method."""

from __future__ import annotations

# Import the package so the method registers itself as a side effect.
import cellquorum.stages.comparative.differential_expression  # noqa: F401
from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


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
