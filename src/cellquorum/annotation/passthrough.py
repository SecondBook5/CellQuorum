"""Passthrough annotation: preserve an already-trusted cell-type label.

For the hypothesis-repo workflow a per-cell-type subrepo subsets an ALREADY
annotated global object, so every cell arrives with a trusted label (e.g.
``cell_type=Fibroblasts``). The mandatory annotation stage must not recompute
(and thereby destroy) that label. This method copies an existing obs column into
``key_added`` (a no-op when ``source_key == key_added``) and fails loud if the
trusted label is absent — never silently nulling identities. Deterministic,
offline, CPU.
"""

from __future__ import annotations

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod


class PassthroughAnnotationMethod(AnalysisMethod):
    """Preserve an existing trusted label instead of recomputing it."""

    # Registry identity.
    name = "passthrough"
    stage_category = "annotation"
    backend = "python"

    def _source_key(self, config: dict) -> str:
        """Return the obs column the trusted label is read from."""

        # source_key wins when set; otherwise the label already lives in key_added.
        return config.get("source_key") or config.get("key_added", "cell_type")

    def input_contract(self, config: dict) -> DataContract:
        """Require the trusted source label to exist on the input."""

        # The source column must be present; there is nothing to preserve without it.
        return DataContract(required_obs=[self._source_key(config)])

    # NOTE: the missing-source case is enforced through ``input_contract`` (which
    # raises) rather than ``requires_obs`` (which would downgrade it to a soft
    # skip). A subrepo pointed at an object with no trusted label is a hard
    # misconfiguration that must fail loud, not silently leave key_added absent.

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """
        Carry an existing trusted label into ``key_added``.

        Args:
            adata: Subset AnnData already carrying the trusted label.
            config: Resolved annotation config sub-block.
            context: Pipeline context (unused).

        Returns:
            StageResult with obs[key_added] set to the preserved label.
        """

        # Resolve source and destination columns.
        key_added = config.get("key_added", "cell_type")
        source_key = self._source_key(config)

        # Copy the trusted label across when the destination differs; otherwise
        # the label already lives in key_added and this is a no-op preserve.
        if source_key != key_added:
            adata.obs[key_added] = adata.obs[source_key]

        # Normalize to categorical for consistency with the other methods.
        adata.obs[key_added] = adata.obs[key_added].astype("category")

        # Count the distinct preserved labels (drop nulls so an empty count is honest).
        n_types = int(adata.obs[key_added].nunique(dropna=True))

        return StageResult(
            adata=adata,
            metrics={
                "n_types": n_types,
                "key_added": key_added,
                "source_key": source_key,
                "method": "passthrough",
            },
            notes=[
                f"passthrough preserved {n_types} existing label(s) "
                f"from obs['{source_key}'] -> obs['{key_added}']."
            ],
        )


__all__ = ["PassthroughAnnotationMethod"]
