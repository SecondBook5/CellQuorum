# Pipeline step (order=360): module_remodeling — condition effects on module activity.
"""Module-remodeling stage: dispatches the module-remodeling method."""

from __future__ import annotations

# Import the package so the method registers itself as a side effect.
import cellquorum.stages.comparative.module_remodeling  # noqa: F401
from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="module_remodeling",
    order=360,
    config_flag="module_remodeling",
    config_field="module_remodeling",
    category="module_remodeling",
)
class ModuleRemodelingStage(MethodDispatchStage):
    """Run the configured module-remodeling method.

    Ordered last on purpose: it is the synthesis stage. It reads the module scores
    ``state_scoring`` wrote, the partition ``subclustering`` derived, and the design
    the cohort declares, so anything earlier would run before its inputs exist.
    """

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "module_remodeling")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        """Bridge the project-level ``design``/``cohort`` blocks into this config.

        A manifest declares the donor column, the condition column, the sample
        column and the case/control tokens once. Without this bridge the stage
        would need a second copy of all five under ``module_remodeling:``, and a
        second copy is a second thing that can disagree with the first — a module
        table testing a different contrast from the DE table beside it, with
        nothing on either to show they diverged.

        Precedence is the house order: an explicit stage value wins, then the
        cohort key for structural columns, then the design block. Nothing is
        invented — with no design present the case/control keys stay unset and the
        method records a clean skip.
        """
        augmented = dict(stage_config)
        config = getattr(context, "config", None)

        # Structural obs columns come from the cohort block when it declares them,
        # which is where a dataset's identity columns are named for every stage.
        for key, cohort_attr, design_attr in (
            ("donor_col", "donor_key", "donor_col"),
            ("condition_col", "condition_key", "condition_col"),
            ("sample_col", "sample_key", None),
        ):
            if augmented.get(key):
                continue
            design = getattr(config, "design", None)
            fallback = getattr(design, design_attr, None) if design and design_attr else None
            resolved = resolve_cohort_key(config, attr=cohort_attr, stage_value=fallback)
            if resolved:
                augmented[key] = resolved

        design = getattr(config, "design", None)
        if design is not None:
            for key in ("case", "control"):
                if not augmented.get(key):
                    augmented[key] = getattr(design, key, None)
            if "paired" not in augmented:
                augmented["paired"] = bool(getattr(design, "paired", False))

        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: the obs additions are config-named and optional by design.

        ``key_added`` is only written when the stage labels a partition itself
        (a caller supplying ``group_col`` already has the labels), and the
        contrast index only when both sides of it are configured. A postcondition
        on either would fail runs that are behaving correctly.
        """


__all__ = ["ModuleRemodelingStage"]
