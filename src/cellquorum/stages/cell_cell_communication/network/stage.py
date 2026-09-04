# Pipeline step (order=340): ccc_network — analyze CCC network topology and Ricci curvature.
"""ccc_network stage: dispatches topology (always) and ricci (dep-guarded)."""

from __future__ import annotations

# Import the package so both methods register themselves as a side effect.
import cellquorum.stages.cell_cell_communication.network  # noqa: F401
from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="ccc_network",
    order=340,
    config_flag="network_analysis",
    config_field="ccc_network",
    category="ccc_network",
)
class CCCNetworkStage(MethodDispatchStage):
    """Run the configured ccc_network method(s): topology then ricci."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "topology")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        augmented = dict(stage_config)

        # Bridge the project-level design block (structural + comparison keys).
        config = getattr(context, "config", None)
        design = getattr(config, "design", None)
        if design is not None:
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

        # sample_col has to be bridged too, and it was the one that was missing.
        # The methods default it to "sample" while the rest of the engine — the CCC
        # stage that produced the LR table this stage reads, the velocity stage, the
        # cohort block — uses "sample_id". So on a run declaring
        # cohort.sample_key: sample_id, resolve_condition_arms found no "sample"
        # column, returned no arms, and the comparative Lymphedema-vs-Normal
        # topology and curvature were skipped for a column-name mismatch. The
        # condition side was bridged and the sample side was not, so nothing else
        # in the run looked wrong.
        if not augmented.get("sample_col"):
            sample_col = resolve_cohort_key(config, attr="sample_key", stage_value=None)
            if sample_col:
                augmented["sample_col"] = sample_col

        # Default to both methods when nothing was specified.
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [{"method": "topology"}, {"method": "ricci"}]

        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: writes CSV artifacts + uns['ccc_network']; no obs/var postcondition."""


__all__ = ["CCCNetworkStage"]
