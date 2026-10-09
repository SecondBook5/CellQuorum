# Pipeline step (order=320): cell_cell_communication — infer signalling (LIANA, tensor).
"""Cell-cell communication stage: dispatches LIANA then Tensor-cell2cell."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cellquorum.stages.cell_cell_communication  # noqa: F401
from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

# tensor_c2c has a hard data dependency on liana's uns['liana_res']; enforce order.
_METHOD_ORDER = {"liana": 0, "tensor_c2c": 1}


@register_stage(
    name="cell_cell_communication",
    order=320,
    config_flag="cell_cell_communication",
    config_field="cell_cell_communication",
    category="cell_cell_communication",
)
class CellCellCommunicationStage(MethodDispatchStage):
    """Run the configured CCC method(s): LIANA (LR) then Tensor-cell2cell."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "liana")

    def _augment_config(self, context: PipelineContext, stage_config: dict) -> dict:
        augmented = dict(stage_config)

        # Default to both methods when nothing was specified.
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [{"method": "liana"}, {"method": "tensor_c2c"}]

        # Enforce liana-before-tensor regardless of configured order.
        methods = augmented.get("methods")
        if methods:
            augmented["methods"] = sorted(
                methods,
                key=lambda m: _METHOD_ORDER.get(m.get("method", ""), 99),
            )

        # Bridge the dataset-wide sample key into sample_col when set.
        config = getattr(context, "config", None)
        if config is not None:
            resolved = resolve_cohort_key(
                config, attr="sample_key", stage_value=augmented.get("sample_col", "sample_id")
            )
            if resolved:
                augmented["sample_col"] = resolved

        # Bridge config.design into multinichenet's case/control/condition_col,
        # the same declare-the-question-once contract differential_expression
        # already has. Without this, multinichenet's own config.get("case")/
        # get("control") are always None and the method always skips with "no
        # contrast declared" -- even when design.case/control name exactly the
        # contrast the run was built around. Precedence: an explicit value
        # already in this stage's own config wins, then the central cohort
        # key (structural columns only), then design, then the method's own
        # default -- identical to DifferentialExpressionStage._augment_config.
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

        if design is not None:
            augmented.setdefault("paired", bool(getattr(design, "paired", False)))
            if not augmented.get("donor_col"):
                augmented["donor_col"] = resolve_cohort_key(
                    config, attr="donor_key", stage_value=getattr(design, "donor_col", "donor_id")
                )
        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: writes table artifacts + uns keys, no obs/var postcondition."""


__all__ = ["CellCellCommunicationStage"]
