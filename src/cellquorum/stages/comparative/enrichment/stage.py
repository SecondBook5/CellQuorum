# Pipeline step (order=230): enrichment — score pathway/gene-set enrichment via the configured method(s).
"""Enrichment stage: dispatches to the configured enrichment method(s)."""

from __future__ import annotations

# Import the package so the methods register themselves as a side effect.
import cellquorum.stages.comparative.enrichment  # noqa: F401
from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="enrichment",
    order=230,
    config_flag="enrichment",
    config_field="enrichment",
    category="enrichment",
)
class EnrichmentStage(MethodDispatchStage):
    """Run the configured enrichment / pathway-activity method(s)."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "gsea")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        """Bridge ``config.design`` + ``config.organism`` into the method config."""
        augmented = dict(stage_config)
        config = getattr(context, "config", None)

        # Organism is always bridged (gene-set fetches are species-specific).
        organism = getattr(config, "organism", None)
        if organism is not None and not augmented.get("organism"):
            augmented["organism"] = organism

        design = getattr(config, "design", None)
        if design is not None:
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
            if not augmented.get("case"):
                augmented["case"] = getattr(design, "case", None)
            if not augmented.get("control"):
                augmented["control"] = getattr(design, "control", None)
            if "paired" not in augmented:
                augmented["paired"] = bool(getattr(design, "paired", False))

        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [
                {"method": "gsea"},
                {"method": "ora"},
                {"method": "gsva"},
                {"method": "activity"},
            ]

        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: enrichment writes table artifacts, no obs/var postcondition."""


__all__ = ["EnrichmentStage"]
