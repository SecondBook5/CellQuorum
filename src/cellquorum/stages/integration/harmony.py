"""Harmony batch integration on GPU (rapids-singlecell) or CPU (harmonypy).

GPU is attempted first via the shared compute router, so this stage honours
compute.backend / prefer_gpu like PCA and Leiden do instead of always running on the
CPU. rapids-singlecell's default flavor is "harmony2", a modified objective; this
stage pins flavor="harmony1", the original Korsunsky algorithm, so routing a run to
the GPU changes only where the arithmetic happens and not which algorithm runs.

The CPU path calls harmonypy.run_harmony directly rather than through scanpy's
harmony_integrate wrapper: harmonypy 0.2.0 is a PyTorch build whose Z_corr is a
tensor that the wrapper mishandles, silently leaving the PCA uncorrected. Both paths
end in the same orientation check and shape assertion, which turns that silent
fallback into a loud failure.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.backends.harmonypy_backend import (
    DEFAULT_MAX_ITER_HARMONY,
    HarmonyDiagnostics,
    harmony_correct,
)
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.stages.qc.eligibility import fitting_cells


class HarmonyMethod(AnalysisMethod):
    """Harmony integration strategy (GPU via rapids-singlecell, CPU via harmonypy)."""

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

    @staticmethod
    def _harmony_gpu(
        adata: ad.AnnData,
        input_rep: str,
        output_rep: str,
        batch_key: str,
        random_state: int,
    ) -> np.ndarray:
        """
        Run rapids-singlecell Harmony and return the corrected embedding.

        Args:
            adata: Input AnnData with obsm[input_rep].
            input_rep: Embedding to correct.
            output_rep: Key harmony_integrate writes into.
            batch_key: obs column identifying the batches.
            random_state: Seed passed through to harmony.

        Returns:
            The corrected embedding as written by harmony_integrate.
        """

        import rapids_singlecell as rsc

        # harmony_integrate moves obsm[basis] to the device itself and writes a host
        # array back, so no whole-object GPU transfer is needed. flavor="harmony1" is
        # the original algorithm, i.e. what the CPU path runs.
        rsc.pp.harmony_integrate(
            adata,
            key=batch_key,
            basis=input_rep,
            adjusted_basis=output_rep,
            flavor="harmony1",
            random_state=random_state,
        )
        return np.asarray(adata.obsm[output_rep])

    @staticmethod
    def _harmony_cpu(
        adata: ad.AnnData,
        embedding: np.ndarray,
        batch_key: str,
        random_state: int,
        max_iter_harmony: int = DEFAULT_MAX_ITER_HARMONY,
    ) -> tuple[np.ndarray, HarmonyDiagnostics]:
        """
        Run harmonypy on the embedding and return it corrected, plus diagnostics.

        Thin wrapper over :func:`cellquorum.backends.harmonypy_backend.harmony_correct`
        so this stage and ``subclustering`` cannot drift on how harmonypy is called.

        Args:
            adata: Input AnnData, for the obs table Harmony conditions on.
            embedding: The (n_cells, n_pcs) embedding to correct.
            batch_key: obs column identifying the batches.
            random_state: Seed passed through to run_harmony.
            max_iter_harmony: Iteration cap.

        Returns:
            ``(corrected, diagnostics)``, corrected already oriented (n_cells, n_pcs).
        """
        return harmony_correct(
            embedding,
            adata.obs[batch_key],
            batch_key,
            random_state=random_state,
            max_iter_harmony=max_iter_harmony,
        )

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """
        Run Harmony on the input embedding and write the corrected embedding.

        Args:
            adata: Input AnnData with obsm[input_rep] and obs[batch_key].
            config: Resolved integration config sub-block.
            context: Pipeline context, for the GPU/CPU routing decision.

        Returns:
            StageResult with obsm[output_rep] set and integration provenance.
        """

        # Decide GPU vs CPU once via the shared router, as PCA and Leiden do.
        from cellquorum.backends.compute import resolve_compute

        # Resolve settings.
        input_rep = config.get("input_rep", "X_pca")
        output_rep = config.get("output_rep", "X_pca_harmony")
        batch_key = config.get("batch_key", "patient_id")
        random_state = int(config.get("random_state", 0))
        max_iter_harmony = int(config.get("max_iter_harmony", DEFAULT_MAX_ITER_HARMONY))

        # The input embedding and its expected corrected shape.
        embedding = np.ascontiguousarray(adata.obsm[input_rep])
        expected_shape = embedding.shape

        routing = resolve_compute(context)
        compute_used = "cpu"
        gpu_fallback_note = None
        z = None
        if routing["use_gpu"]:
            try:
                z = self._harmony_gpu(adata, input_rep, output_rep, batch_key, random_state)
                compute_used = "gpu"
            except Exception as exc:  # noqa: BLE001
                # GPU path failed; fall back to CPU when permitted.
                if not routing["fallback_to_cpu"]:
                    raise
                # Discard any partial write so the CPU result cannot be mixed with it.
                if output_rep in adata.obsm:
                    del adata.obsm[output_rep]
                gpu_fallback_note = (
                    f"GPU Harmony failed ({type(exc).__name__}: {str(exc)[:80]}); "
                    "fell back to CPU."
                )
        diagnostics: HarmonyDiagnostics | None = None
        if z is None:
            z, diagnostics = self._harmony_cpu(
                adata, embedding, batch_key, random_state, max_iter_harmony
            )

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
        # Single-method provenance (backward-compatible path, last-wins).
        cq["integration"] = {
            "method": "harmony",
            "batch_key": batch_key,
            "input_rep": input_rep,
            "output_rep": output_rep,
        }
        # Per-method provenance (multi-method path, namespaced by output_rep).
        cq.setdefault("integration_methods", {})[output_rep] = {
            "method": "harmony",
            "batch_key": batch_key,
            "input_rep": input_rep,
            "output_rep": output_rep,
        }

        notes = [
            f"Harmony corrected {input_rep} over '{batch_key}' -> {output_rep} "
            f"on {compute_used.upper()}."
        ]
        if gpu_fallback_note:
            notes.append(gpu_fallback_note)

        # The integration stage declares fit_scope=CORE, and Harmony is the one method that
        # cannot honour it. Harmony returns corrected coordinates directly rather than a
        # reusable correction: harmonypy exposes no way to apply a fitted correction to cells
        # that were not in the optimisation, so there is no out-of-sample transform to
        # project the excluded cells through. Fitting on core and stopping there would leave
        # them without an integrated embedding at all, which is worse.
        #
        # Recorded rather than ignored, because the alternative is a declaration that reads
        # as compliant while nothing enforces it — the precise failure the cell_scope contract
        # was added to prevent. scVI and scANVI have real encoders and do honour the scope.
        if fitting_cells(adata.obs) is not None:
            notes.append(
                "Harmony fitted on all cells: it has no out-of-sample transform, so the QC "
                "fit population could not be honoured. The corrected embedding is influenced "
                "by non-core cells. Use method='scvi' or 'scanvi' for a core-only "
                "integration, and note that PCA and clustering are still core-fitted."
            )

        # Non-convergence is a WARNING, not a note: every stage downstream reads
        # obsm[output_rep], so a Harmony that stopped early propagates a partially
        # corrected embedding into clustering, UMAP, PAGA and velocity alike, and a
        # note would be printed only at --verbose and counted in no report.
        stage_warnings = [gpu_fallback_note] if gpu_fallback_note else []
        if diagnostics is not None and diagnostics.message:
            stage_warnings.append(
                f"{diagnostics.message} obsm['{output_rep}'] is affected; raise "
                f"integration.max_iter_harmony."
            )

        metrics = {
            "n_cells": int(adata.n_obs),
            "output_rep": output_rep,
            "method": "harmony",
            "compute": compute_used,
        }
        if diagnostics is not None:
            metrics["harmony_n_iter"] = diagnostics.n_iter
            metrics["harmony_converged"] = diagnostics.converged
            metrics["max_iter_harmony"] = diagnostics.max_iter

        return StageResult(
            adata=adata,
            metrics=metrics,
            notes=notes,
            warnings=stage_warnings,
        )


__all__ = ["HarmonyMethod"]
