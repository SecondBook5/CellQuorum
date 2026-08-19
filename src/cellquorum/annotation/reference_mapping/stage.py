"""Reference mapping stage: dispatch to configured reference method."""

from __future__ import annotations

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage


class ReferenceMappingStage(MethodDispatchStage):
    """Config-driven reference mapping stage."""

    # Stage identity.
    name = "reference_mapping"
    stage_category = "reference_mapping"

    def _select_method_name(self, config: dict) -> str:
        """Return the configured reference mapping method (default 'scarches')."""

        # Read the method key from the resolved sub-block.
        return config.get("method", "scarches")

    def _validate_output(self, result: StageResult) -> None:
        """Validate the transferred label and embedding landed in adata."""

        # Skipped results pass through without validation.
        if result.metrics.get("skipped"):
            return

        # The key_added column must be present in obs. Read it from metrics
        # recorded by the method, else fall back to the default.
        key_added = result.metrics.get("key_added", "ref_state")

        # scANVI embedding must be present in obsm.
        DataContract(required_obs=[key_added], required_obsm=["X_scANVI"]).validate(result.adata)


__all__ = ["ReferenceMappingStage"]
