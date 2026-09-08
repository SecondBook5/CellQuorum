# Pipeline step (order=105): query_projection — place borderline cells against core.
#
# The first line is a machine-read contract (tests/test_stage_headers.py): one line,
# `order=` matching the registration, ending in a period. Context goes below it.
#
# Runs AFTER annotation (90) so core cells carry the labels this projects onto, and
# BEFORE qc_finalization (135) which consumes the projection to decide rescue. Reads the
# frozen manifold produced by integration (fit_scope=CORE); it never retrains anything.
"""Query-projection stage: project QC-borderline cells onto the frozen core manifold.

Writes per-cell ``query_*`` columns for borderline cells — neighbourhood support, label
probabilities, and an OOD score — the evidence ``qc_finalization`` needs to decide which
borderline cells land convincingly inside a legitimate core population. Core and
quarantine cells are left untouched (NaN / empty), because the question "does this fit the
reference?" is only meaningful for the cells that were held out of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.context_access import resolve_stage_config
from cellquorum.stages.qc.projection import project_query_cells

#: obsm keys tried in order for the frozen manifold. scVI first because integration fits
#: it with fit_scope=CORE and encodes all cells through the trained model, so borderline
#: cells are already projected onto a core-only manifold there. PCA is the CPU fallback.
_DEFAULT_REP_CANDIDATES = ("X_scvi", "X_pca_harmony", "X_pca")

#: obs columns tried in order for the reference labels the neighbourhood votes on.
_DEFAULT_LABEL_CANDIDATES = ("cell_type", "qc_provisional_lineage", "leiden")

_STATE_COLUMN = "qc_state_initial"


@register_stage(
    name="query_projection",
    order=105,
    config_flag="query_projection",
    config_field="query_projection",
)
class QueryProjectionStage:
    """Project borderline cells onto the frozen core manifold and record the evidence."""

    def _resolve_rep(self, adata: object, config: dict) -> str | None:
        requested = config.get("use_rep")
        candidates = (requested, *_DEFAULT_REP_CANDIDATES) if requested else _DEFAULT_REP_CANDIDATES
        for key in candidates:
            if key and key in adata.obsm:
                return key
        return None

    def _resolve_label_column(self, adata: object, config: dict) -> str | None:
        requested = config.get("label_column")
        candidates = (
            (requested, *_DEFAULT_LABEL_CANDIDATES) if requested else _DEFAULT_LABEL_CANDIDATES
        )
        for col in candidates:
            if col and col in adata.obs.columns:
                return col
        return None

    def run(self, context: object) -> StageResult:
        """Execute the query-projection stage."""
        adata = context.require_adata()
        config = resolve_stage_config(context, "query_projection")

        if _STATE_COLUMN not in adata.obs.columns:
            return StageResult.skipped(
                adata=adata,
                reason=f"no {_STATE_COLUMN} column; QC graded adjudication did not run",
                warnings=[
                    f"query_projection needs obs['{_STATE_COLUMN}']; is the graded QC "
                    "stage enabled?"
                ],
            )

        state = adata.obs[_STATE_COLUMN].astype(str)
        core_mask = (state == "core").to_numpy()
        query_mask = (state == "borderline").to_numpy()

        if not query_mask.any():
            return StageResult.skipped(
                adata=adata,
                reason="no borderline cells to project",
                metrics={"n_borderline": 0, "n_core": int(core_mask.sum())},
            )
        if core_mask.sum() < 2:
            return StageResult.skipped(
                adata=adata,
                reason="fewer than 2 core cells to project against",
                warnings=["query_projection: the core reference is too small to project onto."],
                metrics={"n_borderline": int(query_mask.sum()), "n_core": int(core_mask.sum())},
            )

        rep = self._resolve_rep(adata, config)
        if rep is None:
            return StageResult.skipped(
                adata=adata,
                reason="no frozen representation in obsm (tried X_scvi/X_pca_harmony/X_pca)",
                warnings=[
                    "query_projection needs a core-fit embedding; run integration or "
                    "dimensionality first."
                ],
            )
        label_column = self._resolve_label_column(adata, config)
        if label_column is None:
            return StageResult.skipped(
                adata=adata,
                reason="no reference label column (tried cell_type/qc_provisional_lineage/leiden)",
                warnings=["query_projection needs core-cell labels; run annotation first."],
            )

        k = int(config.get("k", 15))
        coords = np.asarray(adata.obsm[rep])
        labels = adata.obs[label_column].astype(str).to_numpy()

        projection = project_query_cells(
            reference_coords=coords[core_mask],
            query_coords=coords[query_mask],
            reference_labels=labels[core_mask],
            k=k,
        )

        # Write query_* columns, aligned to the full obs by leaving non-borderline cells
        # NaN (numeric) or empty (label). Assembled once and concatenated, not written
        # column-by-column, to avoid the fragmentation sawtooth on a 200k-row frame.
        n = adata.n_obs
        qi = np.flatnonzero(query_mask)

        def _fill_num(values: np.ndarray) -> np.ndarray:
            out = np.full(n, np.nan, dtype=float)
            out[qi] = values
            return out

        top_label_full = np.array([""] * n, dtype=object)
        top_label_full[qi] = projection.top_label.astype(str)

        new_cols = {
            "query_top_label": top_label_full,
            "query_top_label_probability": _fill_num(projection.top_label_probability),
            "query_second_label_probability": _fill_num(projection.second_label_probability),
            "query_label_margin": _fill_num(projection.margin),
            "query_neighbor_label_entropy": _fill_num(projection.neighbor_label_entropy),
            "query_nearest_reference_distance": _fill_num(projection.nearest_reference_distance),
            "query_mean_neighbor_distance": _fill_num(projection.mean_neighbor_distance),
            "query_ood_score": _fill_num(projection.ood_score),
            "query_effective_neighbor_count": _fill_num(projection.effective_neighbor_count),
        }
        adata.obs = pd.concat([adata.obs, pd.DataFrame(new_cols, index=adata.obs_names)], axis=1)

        adata.uns.setdefault("cellquorum", {})["query_projection"] = {
            "representation": rep,
            "label_column": label_column,
            "k": k,
            "n_borderline": int(query_mask.sum()),
            "n_core_reference": int(core_mask.sum()),
        }

        median_ood = float(np.median(projection.ood_score))
        median_support = float(np.median(projection.top_label_probability))
        return StageResult(
            adata=adata,
            notes=[
                f"query_projection projected {int(query_mask.sum()):,} borderline cells onto "
                f"{int(core_mask.sum()):,} core cells in '{rep}' (labels from "
                f"'{label_column}', k={k}).",
                f"median neighbourhood support={median_support:.2f}, median OOD={median_ood:.2f}.",
            ],
            metrics={
                "representation": rep,
                "label_column": label_column,
                "k": k,
                "n_borderline": int(query_mask.sum()),
                "n_core": int(core_mask.sum()),
                "median_top_label_probability": median_support,
                "median_ood_score": median_ood,
            },
        )


__all__ = ["QueryProjectionStage"]
