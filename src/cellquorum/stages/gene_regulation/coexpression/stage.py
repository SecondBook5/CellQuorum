"""Co-expression stage: dispatches to the configured co-expression method."""

from __future__ import annotations

# Import the package so the method registers itself as a side effect.
import cellquorum.stages.gene_regulation.coexpression  # noqa: F401
from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="coexpression",
    order=260,
    config_flag="coexpression",
    config_field="coexpression",
    category="coexpression",
)
class CoexpressionStage(MethodDispatchStage):
    """Run the configured co-expression (hdWGCNA) method."""

    def _select_method_name(self, config: dict) -> str:
        """Return the configured co-expression method name."""

        # Default to hdWGCNA.
        return config.get("method", "hdwgcna")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        """
        Bridge the project-level ``design`` block into the method's config.

        Co-expression is not a two-group contrast, so only the condition column
        is bridged (used for the optional module-condition correlation). It is
        resolved with the established precedence — explicit stage value wins,
        then the central cohort key, then the design block, then the default.
        When no design is present the config passes through unchanged and the
        method falls back to its own generic column handling.

        Args:
            context: Pipeline context exposing ``config`` (and its ``design``).
            stage_config: The resolved (cohort-overlaid) stage config dict.

        Returns:
            A copy of the config with the condition column filled in, or the
            input unchanged when no design block exists.
        """

        config = getattr(context, "config", None)
        design = getattr(config, "design", None)
        if design is None:
            return stage_config

        augmented = dict(stage_config)

        # Condition column: stage value → cohort key → design → default.
        if not augmented.get("condition_col"):
            augmented["condition_col"] = resolve_cohort_key(
                config,
                attr="condition_key",
                stage_value=getattr(design, "condition_col", "condition"),
            )

        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No structural postcondition on obs/var; the stage writes tables + a figure."""
