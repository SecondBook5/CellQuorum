# Pipeline step (order=135): qc_finalization — decide rescue and write qc_state_final.
#
# The first line is a machine-read contract (tests/test_stage_headers.py): one line,
# `order=` matching the registration, ending in a period. Context goes below it.
#
# Sits at 135, AFTER reference_mapping (120) and annotation_consensus (130), because
# rescue uses atlas support as evidence. Placing it at 95 and mutating the result later
# was rejected in the frozen design: qc_state_final must be genuinely final, not a
# provisional value something downstream overwrites.
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

_STATE_INITIAL = "qc_state_initial"
_STATE_FINAL = "qc_state_final"

#: Per-family severity columns whose presence at severe level is a technical
#: contradiction. Nuclear integrity and metabolic (mitochondrial) stress are damage
#: signatures; a high-confidence multiplet is not one biological cell. Capture/complexity
#: is deliberately NOT here: low complexity is the constitutive biology of the rare
#: populations rescue exists to protect, so it cannot by itself veto a rescue.
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
        config = resolve_stage_config(context, "qc_finalization")

        if _STATE_INITIAL not in adata.obs.columns:
            return StageResult.skipped(
                adata=adata,
                reason=f"no {_STATE_INITIAL} column; QC graded adjudication did not run",
                warnings=[f"qc_finalization needs obs['{_STATE_INITIAL}']."],
            )

        # Thresholds. Stated, not inherited: a methods section cannot cite a default, and a
        # library upgrade must not silently re-adjudicate a published cohort.
        support_min = float(config.get("min_neighborhood_support", 0.5))
        ood_max = float(config.get("max_ood_score", 0.95))
        severe_min = float(config.get("severe_severity", 0.9))

        state = adata.obs[_STATE_INITIAL].astype(str)
        final = state.copy()  # core / quarantine pass through unchanged
        borderline = (state == "borderline").to_numpy()

        rescued_mask = np.zeros(adata.n_obs, dtype=bool)
        have_projection = "query_top_label_probability" in adata.obs.columns

        if borderline.any() and have_projection:
            support = adata.obs["query_top_label_probability"].to_numpy(dtype=float)
            ood = adata.obs.get("query_ood_score")
            ood = (
                ood.to_numpy(dtype=float) if ood is not None else np.zeros(adata.n_obs, dtype=float)
            )

            # Biological support: the neighbourhood agrees on a label AND the cell is not
            # out-of-distribution. NaN support (a cell that was not projected) fails
            # closed — no evidence is not evidence for rescue.
            biological_support = (np.nan_to_num(support, nan=0.0) >= support_min) & (
                np.nan_to_num(ood, nan=1.0) <= ood_max
            )

            # Severe technical contradiction: any damage/multiplet family at severe level.
            # Missing columns contribute nothing rather than raising, so the rule degrades
            # to "biological support alone" if evidence is absent rather than crashing.
            contradiction = np.zeros(adata.n_obs, dtype=bool)
            for col in _CONTRADICTION_SEVERITY_COLUMNS:
                if col in adata.obs.columns:
                    sev = adata.obs[col].to_numpy(dtype=float)
                    contradiction |= np.nan_to_num(sev, nan=0.0) >= severe_min

            # A probable multiplet is not one biological cell; it can never be rescued,
            # regardless of how convincingly its transcriptome maps.
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
