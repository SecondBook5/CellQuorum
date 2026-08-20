"""Clustering stage: dispatch to the configured clustering method.

Mirrors DimensionalityStage: resolves its config sub-block from a pydantic-or-dict
context, dispatches to the configured method via the registry, and validates that
cluster labels landed in obs before handing the AnnData downstream.
"""

from __future__ import annotations

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.registry import MethodRegistry
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="clustering",
    order=80,
    config_flag="clustering",
    config_field="clustering",
    category="clustering",
)
class ClusteringStage(MethodDispatchStage):
    """Config-driven clustering stage."""

    def __init__(self, registry: MethodRegistry | None = None) -> None:
        super().__init__(registry)
        # Store the key_added from config so _validate_output can access it.
        self._key_added = None

    def _select_method_name(self, config: dict) -> str:
        """Return the configured clustering method (default 'leiden')."""

        # Read the method key from the resolved sub-block.
        return config.get("method", "leiden")

    def run(self, context: object) -> StageResult:
        """Override run to store key_added for validation and auto-couple use_rep."""
        # Resolve config to extract key_added before calling base run.
        from cellquorum.methods.context_access import resolve_stage_config

        stage_config = resolve_stage_config(context, self.name)
        self._key_added = stage_config.get("key_added", "leiden")

        # Auto-couple use_rep when integration is enabled and user didn't explicitly set it.
        # This ensures clustering operates on the integration output (X_pca_harmony) by default
        # instead of ignoring it and reading raw X_pca.
        use_rep_from_config = stage_config.get("use_rep", "X_pca")
        if use_rep_from_config == "X_pca":
            # Check if integration is enabled via context.config
            integration_enabled = False
            integration_output_rep = "X_pca_harmony"

            if hasattr(context, "config"):
                cfg = context.config
                # Handle both dict and pydantic config objects
                if isinstance(cfg, dict):
                    stages = cfg.get("stages", {})
                    integration_enabled = stages.get("integration", False)
                    integration_cfg = cfg.get("integration", {})
                    # When integration ran a methods list, couple to the last method's output_rep.
                    integration_methods = integration_cfg.get("methods", [])
                    if integration_methods:
                        integration_output_rep = integration_methods[-1].get(
                            "output_rep", "X_pca_harmony"
                        )
                    else:
                        integration_output_rep = integration_cfg.get("output_rep", "X_pca_harmony")
                else:
                    integration_enabled = getattr(cfg.stages, "integration", False)
                    if hasattr(cfg, "integration"):
                        # When integration ran a methods list, couple to the last output_rep.
                        integration_methods = getattr(cfg.integration, "methods", [])
                        if integration_methods:
                            integration_output_rep = integration_methods[-1].get(
                                "output_rep", "X_pca_harmony"
                            )
                        else:
                            integration_output_rep = getattr(
                                cfg.integration, "output_rep", "X_pca_harmony"
                            )

            # Check if the user explicitly set use_rep via the pydantic model's model_fields_set
            user_set_use_rep = False
            if hasattr(context, "config") and not isinstance(context.config, dict):
                clustering_model = getattr(context.config, "clustering", None)
                if clustering_model is not None and hasattr(clustering_model, "model_fields_set"):
                    user_set_use_rep = "use_rep" in clustering_model.model_fields_set
            elif isinstance(getattr(context, "config", None), dict):
                # For dict configs (tests): treat use_rep as explicit if present and != "X_pca"
                clustering_dict = context.config.get("clustering", {})
                if "use_rep" in clustering_dict and clustering_dict["use_rep"] != "X_pca":
                    user_set_use_rep = True

            # Override use_rep if integration is enabled and user didn't explicitly set it
            if integration_enabled and not user_set_use_rep:
                stage_config["use_rep"] = integration_output_rep
                # Store a note for the result to make this non-silent
                self._auto_coupled_use_rep = integration_output_rep

        # Now call base run which will handle enabled check, dispatch, and validation.
        result = super().run(context)

        # Add note if we auto-coupled use_rep
        if hasattr(self, "_auto_coupled_use_rep") and not result.metrics.get("skipped"):
            result.notes.append(
                f"clustering.use_rep auto-set to {self._auto_coupled_use_rep} "
                "because integration is enabled."
            )

        return result

    def _validate_output(self, result: StageResult) -> None:
        """Validate that cluster labels landed in the configured obs column."""

        # Skipped results pass through without validation.
        if not result.metrics.get("skipped"):
            key_added = self._key_added or "leiden"
            DataContract(required_obs=[key_added]).validate(result.adata)


__all__ = ["ClusteringStage"]
