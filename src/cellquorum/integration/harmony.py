"""Harmony batch integration via a direct harmonypy.run_harmony call.

harmonypy 0.2.0 is a PyTorch build whose Z_corr is a tensor that scanpy's
harmony_integrate wrapper mishandles (it silently falls back to uncorrected PCA).
We therefore call run_harmony directly, convert the tensor, orient the result to
(n_cells, n_pcs), and assert the written embedding shape — turning that silent
fallback into a loud failure.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod


class HarmonyMethod(AnalysisMethod):
    """Harmony integration strategy (CPU-capable, direct run_harmony)."""

    # Registry identity.
    name = "harmony"
    stage_category = "integration"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """Require the input embedding and the batch obs column."""

        # Read the input embedding key + batch column from config.
        input_rep = config.get("input_rep", "X_pca")
        batch_key = config.get("batch_key", "patient_id")

        # Require both to exist before running.
        return DataContract(required_obsm=[input_rep], required_obs=[batch_key])

    def requires_obs(self, config: dict) -> list[str]:
        """Return the batch key that must exist for integration to run."""

        # Read the batch column from config.
        batch_key = config.get("batch_key", "patient_id")

        # Require the batch column to exist.
        return [batch_key]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """
        Run Harmony on the input embedding and write the corrected embedding.

        Args:
            adata: Input AnnData with obsm[input_rep] and obs[batch_key].
            config: Resolved integration config sub-block.
            context: Pipeline context (unused).

        Returns:
            StageResult with obsm[output_rep] set and integration provenance.
        """

        # Import harmonypy lazily so importing the package doesn't require it.
        import logging

        import harmonypy

        # Resolve settings.
        input_rep = config.get("input_rep", "X_pca")
        output_rep = config.get("output_rep", "X_pca_harmony")
        batch_key = config.get("batch_key", "patient_id")
        random_state = int(config.get("random_state", 0))

        # The input embedding and its expected corrected shape.
        embedding = np.asarray(adata.obsm[input_rep])
        expected_shape = embedding.shape

        # Run Harmony directly (NOT via scanpy's wrapper). Silence harmonypy's
        # INFO logs only for the duration of the call, then restore the prior
        # level so we never mutate process-wide logging state permanently.
        harmony_logger = logging.getLogger("harmonypy")
        original_level = harmony_logger.level
        harmony_logger.setLevel(logging.WARNING)
        try:
            harmony_obj = harmonypy.run_harmony(
                embedding,
                adata.obs,
                [batch_key],
                random_state=random_state,
            )
        finally:
            harmony_logger.setLevel(original_level)

        # Z_corr may be a torch tensor (PyTorch build) or ndarray; normalize.
        z = harmony_obj.Z_corr
        z = z.cpu().numpy() if hasattr(z, "cpu") else np.asarray(z)

        # run_harmony may return (n_pcs, n_cells) or (n_cells, n_pcs) depending on
        # the build; orient to (n_cells, n_pcs) by matching the input shape.
        if z.shape == expected_shape:
            corrected = z
        elif z.T.shape == expected_shape:
            corrected = z.T
        else:
            # Neither orientation matches — the exact silent-fallback bug. Fail loud.
            raise ValueError(
                f"Harmony output shape {z.shape} matches neither {expected_shape} "
                f"nor its transpose; refusing to write a mis-oriented embedding."
            )

        # Write the corrected embedding and record provenance.
        adata.obsm[output_rep] = np.ascontiguousarray(corrected)
        cq = adata.uns.setdefault("cellquorum", {})
        cq["integration"] = {
            "method": "harmony",
            "batch_key": batch_key,
            "input_rep": input_rep,
            "output_rep": output_rep,
        }

        return StageResult(
            adata=adata,
            metrics={"n_cells": int(adata.n_obs), "output_rep": output_rep, "method": "harmony"},
            notes=[f"Harmony corrected {input_rep} over '{batch_key}' -> {output_rep}."],
        )


__all__ = ["HarmonyMethod"]
