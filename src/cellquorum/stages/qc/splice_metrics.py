# Pipeline step (order=15): qc_splice_metrics — optional, lightweight splice-QC
# extraction from existing velocyto loom output, before qc_evidence (20) consumes it.
"""Per-cell intronic fraction from reconciled spliced/unspliced loom counts.

Optional and lightweight by design (docs/design/qc-graded-adjudication.md's stage
layout): this stage only READS spliced/unspliced counts a prior velocyto run already
produced, via the same manifest/reconcile_looms mechanism the trajectory stage uses for
RNA velocity. It never triggers BAM-level loom generation — that is
``VelocityGenerationConfig``'s heavier job, and would make an "optional" QC axis depend
on a multi-minute-per-sample subprocess. Missing looms or a missing manifest are a
skip, not a failure: the axis this feeds (``NUCLEAR_INTEGRITY``, alongside
``malat1_fraction``) is one of several, and QC must still run without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.context_access import resolve_stage_config
from cellquorum.stages.qc.config import QCSpliceMetricsConfig
from cellquorum.stages.trajectory._loom_io import reconcile_looms

INTRONIC_FRACTION_COLUMN = "qc_splice_intronic_fraction"


def _column_sums(matrix: object) -> np.ndarray:
    """Per-cell (row) sum over genes, for a dense array or scipy sparse matrix."""

    if sp.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel()
    return np.asarray(matrix).sum(axis=1)


@register_stage(
    name="qc_splice_metrics",
    order=15,
    config_flag="qc_splice_metrics",
    config_field="qc_splice_metrics",
)
class QCSpliceMetricsStage:
    """Attach obs['qc_splice_intronic_fraction'] from reconciled velocyto looms."""

    def run(self, context: object) -> StageResult:
        """Execute the qc_splice_metrics stage."""
        adata = context.require_adata()
        config = QCSpliceMetricsConfig.model_validate(
            resolve_stage_config(context, "qc_splice_metrics")
        )

        try:
            manifest = context.require_manifest()
        except Exception:
            return StageResult.skipped(
                adata=adata,
                reason="no manifest available",
                warnings=["qc_splice_metrics needs a sample manifest with a loom-path column"],
            )

        if config.loom_path_col not in manifest.columns:
            return StageResult.skipped(
                adata=adata,
                reason=f"manifest has no '{config.loom_path_col}' column",
                warnings=[
                    f"qc_splice_metrics needs manifest column '{config.loom_path_col}' "
                    "naming each sample's velocyto loom"
                ],
            )

        velo_adata, notes = reconcile_looms(
            adata, manifest, sample_col=config.sample_col, loom_path_col=config.loom_path_col
        )
        if velo_adata is None:
            return StageResult.skipped(
                adata=adata,
                reason="no loom counts reconciled to any cell",
                warnings=[f"qc_splice_metrics: {note}" for note in notes] or None,
            )

        spliced = _column_sums(velo_adata.layers["spliced"])
        unspliced = _column_sums(velo_adata.layers["unspliced"])
        total = spliced + unspliced
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(total > 0, unspliced / total, np.nan)

        intronic_fraction = pd.Series(np.nan, index=adata.obs_names, dtype=float)
        intronic_fraction.loc[velo_adata.obs_names] = fraction
        adata.obs[INTRONIC_FRACTION_COLUMN] = intronic_fraction.to_numpy()

        return StageResult(
            adata=adata,
            artifacts=[],
            notes=[f"qc_splice_metrics: {int(np.isfinite(fraction).sum())} of {adata.n_obs} cells"],
            warnings=[f"qc_splice_metrics: {note}" for note in notes],
            metrics={
                "n_reconciled": int(velo_adata.n_obs),
                "n_total": int(adata.n_obs),
            },
        )


__all__ = ["INTRONIC_FRACTION_COLUMN", "QCSpliceMetricsStage"]
