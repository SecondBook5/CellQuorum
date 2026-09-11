# Pipeline step (order=80): clustering — cluster cells via the configured clustering method.
"""Clustering stage: dispatch to the configured clustering method.

Mirrors DimensionalityStage: resolves its config sub-block from a pydantic-or-dict
context, dispatches to the configured method via the registry, and validates that
cluster labels landed in obs before handing the AnnData downstream.
"""

from __future__ import annotations

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import CellScope, CellScopePolicy, register_stage
from cellquorum.methods.registry import MethodRegistry
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="clustering",
    order=80,
    config_flag="clustering",
    config_field="clustering",
    category="clustering",
    # Fits the neighbour graph and cluster structure, so damaged cells could
    # otherwise shape the biological reference every later stage is measured against.
    cell_scope=CellScopePolicy(fit_scope=CellScope.CORE),
)
class ClusteringStage(MethodDispatchStage):
    """Config-driven clustering stage."""

    def __init__(self, registry: MethodRegistry | None = None) -> None:
        super().__init__(registry)
        # Store the key_added from config so _validate_output can access it.
        self._key_added = None
        self._auto_coupled_use_rep: str | None = None
        self._integration_output_missing: str | None = None

    def _select_method_name(self, config: dict) -> str:
        """Return the configured clustering method (default 'leiden')."""

        # Read the method key from the resolved sub-block.
        return config.get("method", "leiden")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        """Store key_added for validation, and auto-couple use_rep to integration's output.

        This must mutate and return the SAME config MethodDispatchStage.run() dispatches
        with, not a copy resolved separately in an overridden ``run()`` -- ``run()`` calls
        ``resolve_stage_config`` itself, which builds a fresh dict every time, so a mutation
        made outside this hook is silently discarded before the method ever sees it. That
        was the actual bug: the note said "auto-set to X_pca_harmony" while LeidenMethod
        still received X_pca, so clustering ran on the uncorrected embedding whenever
        integration was enabled, without a crash or a wrong-looking note to catch it.

        Ensures clustering operates on the integration output (X_pca_harmony) by default
        instead of ignoring it and reading raw X_pca.
        """
        self._key_added = stage_config.get("key_added", "leiden")
        self._auto_coupled_use_rep = None
        self._integration_output_missing = None

        use_rep_from_config = stage_config.get("use_rep", "X_pca")
        if use_rep_from_config != "X_pca":
            return stage_config

        # Check if integration is enabled via context.config
        integration_enabled = False
        integration_output_rep = "X_pca_harmony"

        cfg = getattr(context, "config", None)
        if cfg is not None:
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
        if cfg is not None and not isinstance(cfg, dict):
            clustering_model = getattr(cfg, "clustering", None)
            if clustering_model is not None and hasattr(clustering_model, "model_fields_set"):
                user_set_use_rep = "use_rep" in clustering_model.model_fields_set
        elif isinstance(cfg, dict):
            # For dict configs (tests): treat use_rep as explicit if present and != "X_pca"
            clustering_dict = cfg.get("clustering", {})
            if "use_rep" in clustering_dict and clustering_dict["use_rep"] != "X_pca":
                user_set_use_rep = True

        # Override use_rep if integration is enabled and user didn't explicitly set it.
        #
        # `integration_enabled` reflects config (stages.integration), not outcome: Harmony
        # (or scVI/scANVI) can internally MethodSkip -- a missing batch_key column, most
        # commonly -- which the stage records as skipped without raising, so
        # stages.integration stays True either way. Coupling on the flag alone crashed
        # clustering with a contract error on an obsm key integration never wrote, the
        # first time this path ran against data without the expected batch/patient column.
        # Checking the actual AnnData is the only way to know integration really produced
        # the embedding, so that is what gates the couple.
        if integration_enabled and not user_set_use_rep:
            adata = context.require_adata()
            if integration_output_rep in adata.obsm:
                stage_config["use_rep"] = integration_output_rep
                # Store for the result note below, to make this non-silent.
                self._auto_coupled_use_rep = integration_output_rep
            else:
                self._integration_output_missing = integration_output_rep

        return stage_config

    def run(self, context: object) -> StageResult:
        """Override run to disclose the auto-couple outcome. Both flags are set inside
        _augment_config, which runs synchronously as part of super().run() below, so
        they are already populated by the time it returns."""
        result = super().run(context)

        if result.metrics.get("skipped"):
            return result

        if self._auto_coupled_use_rep:
            result.notes.append(
                f"clustering.use_rep auto-set to {self._auto_coupled_use_rep} "
                "because integration is enabled."
            )
        elif self._integration_output_missing:
            # A warning, not a note: clustering silently fell back to raw, uncorrected
            # PCA -- exactly the batch-effect-uncorrected result auto-coupling exists to
            # avoid -- so this must print unconditionally, not only in verbose runs.
            result.warnings.append(
                f"integration is enabled but obsm['{self._integration_output_missing}'] "
                "is not present (its method likely skipped -- check the integration "
                "stage's own warnings). clustering fell back to X_pca, uncorrected."
            )

        return result

    def _validate_output(self, result: StageResult) -> None:
        """Validate that cluster labels landed in the configured obs column."""

        # Skipped results pass through without validation.
        if not result.metrics.get("skipped"):
            key_added = self._key_added or "leiden"
            DataContract(required_obs=[key_added]).validate(result.adata)


__all__ = ["ClusteringStage"]
