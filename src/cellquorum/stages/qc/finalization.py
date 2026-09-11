# Pipeline step (order=135): qc_finalization — decide rescue and write qc_state_final.
"""QC finalization: per-cell rescue of borderline cells into a final QC state.

Reads the query projection (stage 105), the reference-mapping support (stage 120), and
the per-family QC severities (stage 20), and applies the rescue rule from
``docs/design/qc-graded-adjudication.md`` §6::

    Rescue = BiologicalSupport AND NOT SevereTechnicalContradiction

deliberately NOT ``kNN > 0.9`` — damaged cells can still map near a legitimate
population, so a high neighbour vote alone is not enough; it must also be free of severe,
independent technical failure. Per-cell, with no minimum rescued-cluster size, because
rare real populations are exactly where rescue must work and a size floor would delete
them (§6, "A minimum rescued-cluster size is explicitly rejected").

Writes ``qc_state_final`` ∈ {core, rescued, unresolved_borderline, quarantine}. Core and
quarantine pass through unchanged; only borderline cells are adjudicated here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.context_access import resolve_stage_config
from cellquorum.stages.qc.config import QCFinalizationConfig

_STATE_INITIAL = "qc_state_initial"
_STATE_FINAL = "qc_state_final"


_CONTRADICTION_SEVERITY_COLUMNS = (
    "qc_ev_family_nuclear_integrity_severity",
    "qc_ev_family_metabolic_stress_severity",
    "qc_ev_family_multiplet_severity",
)


@register_stage(
    name="qc_finalization",
    order=135,
    config_flag="qc_finalization",
    config_field="qc_finalization",
)
class QCFinalizationStage:
    """Adjudicate borderline cells into qc_state_final via the rescue rule."""

    def run(self, context: object) -> StageResult:
        """Execute QC finalization."""
        adata = context.require_adata()
        config = QCFinalizationConfig.model_validate(
            resolve_stage_config(context, "qc_finalization")
        )

        if _STATE_INITIAL not in adata.obs.columns:
            return StageResult.skipped(
                adata=adata,
                reason=f"no {_STATE_INITIAL} column; QC graded adjudication did not run",
                warnings=[f"qc_finalization needs obs['{_STATE_INITIAL}']."],
            )

        support_min = config.min_neighborhood_support
        ood_max = config.max_ood_score
        severe_min = config.severe_severity

        state = adata.obs[_STATE_INITIAL].astype(str)
        final = state.copy()
        borderline = (state == "borderline").to_numpy()

        rescued_mask = np.zeros(adata.n_obs, dtype=bool)
        have_projection = all(
            column in adata.obs.columns
            for column in ("query_top_label_probability", "query_ood_score")
        )

        if borderline.any() and have_projection:
            support = adata.obs["query_top_label_probability"].to_numpy(dtype=float)
            ood = adata.obs["query_ood_score"].to_numpy(dtype=float)

            biological_support = (
                np.isfinite(support)
                & np.isfinite(ood)
                & (support >= support_min)
                & (support <= 1.0)
                & (ood >= 0.0)
                & (ood <= ood_max)
            )

            contradiction = np.zeros(adata.n_obs, dtype=bool)
            for col in _CONTRADICTION_SEVERITY_COLUMNS:
                if col in adata.obs.columns:
                    sev = adata.obs[col].to_numpy(dtype=float)
                    contradiction |= np.nan_to_num(sev, nan=0.0) >= severe_min

            if "qc_probable_multiplet" in adata.obs.columns:
                contradiction |= adata.obs["qc_probable_multiplet"].to_numpy(dtype=bool)

            rescued_mask = borderline & biological_support & ~contradiction

        final_values = final.to_numpy().astype(object)
        final_values[rescued_mask] = "rescued"
        unresolved_mask = borderline & ~rescued_mask
        final_values[unresolved_mask] = "unresolved_borderline"
        adata.obs[_STATE_FINAL] = pd.Categorical(
            final_values, categories=["core", "rescued", "unresolved_borderline", "quarantine"]
        )

        counts = {k: int(v) for k, v in pd.Series(final_values).value_counts().items()}
        adata.uns.setdefault("cellquorum", {})["qc_finalization"] = {
            "min_neighborhood_support": support_min,
            "max_ood_score": ood_max,
            "severe_severity": severe_min,
            "projection_available": have_projection,
            "state_final_counts": counts,
        }

        warnings = []
        if borderline.any() and not have_projection:
            warnings.append(
                "qc_finalization: no query projection found, so no borderline cell could be "
                "rescued. Enable the query_projection stage (105). All borderline cells were "
                "marked unresolved_borderline."
            )

        n_border = int(borderline.sum())
        n_rescued = int(rescued_mask.sum())
        return StageResult(
            adata=adata,
            notes=[
                f"qc_finalization: {n_rescued:,}/{n_border:,} borderline cells rescued "
                f"(support>={support_min}, OOD<={ood_max}, no family severity>={severe_min}).",
                f"qc_state_final: {counts}.",
            ],
            warnings=warnings,
            metrics={
                "n_borderline": n_border,
                "n_rescued": n_rescued,
                "n_unresolved_borderline": int(unresolved_mask.sum()),
                "state_final_counts": counts,
                "min_neighborhood_support": support_min,
                "max_ood_score": ood_max,
                "severe_severity": severe_min,
            },
        )


__all__ = ["QCFinalizationStage"]
