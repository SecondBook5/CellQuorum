"""Cell-cell communication stage: dispatches LIANA then Tensor-cell2cell."""

from __future__ import annotations

# Import the package so both methods register themselves as a side effect.
import cellquorum.cell_cell_communication  # noqa: F401
from cellquorum.config.cohort import resolve_cohort_key
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage

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

    def _augment_config(self, context: object, stage_config: dict) -> dict:
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

        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: writes table artifacts + uns keys, no obs/var postcondition."""


__all__ = ["CellCellCommunicationStage"]
