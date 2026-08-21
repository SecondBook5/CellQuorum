"""Differential-abundance stage: dispatches to the configured DA method."""

from __future__ import annotations

# Import the package so the methods register themselves as a side effect.
import cellquorum.comparative.differential_abundance  # noqa: F401
from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="differential_abundance",
    order=220,
    config_flag="differential_abundance",
    config_field="differential_abundance",
    category="differential_abundance",
)
class DifferentialAbundanceStage(MethodDispatchStage):
    """Run the configured differential-abundance method(s)."""

    def _select_method_name(self, config: dict) -> str:
        """Return the configured DA method name."""

        # Default to milo.
        return config.get("method", "milo")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        """
        Bridge the project-level ``design`` block into the DA method's config.

        The biological question (donor/condition columns, case/control tokens,
        pairing) is declared once in ``config.design``; the DA method reads it
        from its own config dict. This resolves each key with the established
        precedence — an explicit stage value wins, then the central cohort key
        (structural columns only), then the design block, then the method's own
        default — so nothing here is study-specific. When no design is present
        the case/control keys stay unset and the method records a clean skip
        rather than crashing.

        Additionally, if the config has neither a ``methods`` list nor a scalar
        ``method`` key, this injects the default 4-method list so a bare DA
        config runs all available methods.

        Args:
            context: Pipeline context exposing ``config`` (and its ``design``).
            stage_config: The resolved (cohort-overlaid) DA config dict.

        Returns:
            A copy of the config with design-derived keys filled in and the
            default methods list injected if appropriate.
        """

        # Always work on a copy so we never mutate the input.
        augmented = dict(stage_config)

        # Bridge the design block if present.
        config = getattr(context, "config", None)
        design = getattr(config, "design", None)
        if design is not None:
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

        # Inject the default methods list if the config is bare (no methods, no method).
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [
                {"method": "milo"},
                {"method": "sccoda"},
                {"method": "propeller"},
                {"method": "proportion_ttest"},
            ]

        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: DA writes table artifacts, no obs/var postcondition."""
