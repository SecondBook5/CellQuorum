# Pipeline step (order=20): qc — turn graded states into per-analysis eligibility masks.
"""Eligibility: which cells may fit a model, receive its output, or inform a conclusion.

This module exists because of a specific failure. The QC stage used to write one boolean,
``cellquorum_qc_keep``, and across the entire codebase three places read it — two of which
draw figures. Not preprocessing, not feature selection, not PCA, not integration, not
clustering, not annotation, not DE, not DA. So a careful verdict had no effect on any
analysis, and the production default of ``flag_no_drop`` meant QC was reporting without
control.

That was not a QC-design failure. It was an **engine-contract failure**: nothing stopped a
developer writing ``model.fit(adata)`` on everything. Replacing one boolean with six
prettier columns would recreate it exactly. So the masks here are paired with a
registration-level contract in :mod:`cellquorum.core.stage_catalog`, and a test that fails
when a stage which fits a model does not declare whose cells it may fit on.

## Fit, transform, and infer are three different permissions

A rescued keratinocyte may legitimately *receive* an scVI coordinate and a cell label while
being forbidden from *influencing* the model that produced either. Collapsing that into one
boolean is how the circularity creeps back:

    FIT         may determine parameters, statistics, or structure
    TRANSFORM   may receive a representation or an output
    INFERENCE   may contribute to a scientific conclusion

Only ``core`` cells may ever fit. Borderline cells are projected, not joined. Quarantined
cells inform nothing.

## The rule that is easy to miss

FIT is not only about models. **Any cohort-derived quantity used to transform biological
data must be estimated from the permitted fit population** — normalization targets, gene
prevalence filters, HVG dispersions, scaling means, PCA loadings, batch-correction
parameters, neighbour graphs, cluster centroids.

The PFlog1pPF recipe is the trap, and worth stating precisely because it was misread twice.
The quantity is *not* ``target_sum`` in ``normalization.py`` — every pure-matrix recipe there
is per-cell and fits nothing. It is the scclr backend's ``scclr_target``, which defaults to
``auto``: an estimate of the negative-binomial overdispersion alpha **across cells**
(``mean``/``median`` take a cohort depth instead). One damaged cell moves that target, and the
target scales every cell's normalized values, upstream of HVG and PCA both. It does not look
like a fitted model and it is one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc.evidence import QCStateInitial


class Permission(StrEnum):
    """What a cell may do in a given analysis."""

    FIT = "fit"

    TRANSFORM = "transform"

    INFERENCE = "inference"


class Analysis(StrEnum):
    """Analyses whose eligibility is decided separately."""

    MANIFOLD = "manifold"
    CLUSTERING = "clustering"
    ANNOTATION = "annotation"
    COMPOSITION = "composition"
    DIFFERENTIAL_EXPRESSION = "de"
    TRAJECTORY = "trajectory"
    CELL_CELL_COMMUNICATION = "ccc"


_ELIGIBILITY: dict[str, dict[Analysis, frozenset[Permission]]] = {
    str(QCStateInitial.CORE): {
        analysis: frozenset({Permission.FIT, Permission.TRANSFORM, Permission.INFERENCE})
        for analysis in Analysis
    },
    str(QCStateInitial.BORDERLINE): {
        Analysis.MANIFOLD: frozenset({Permission.TRANSFORM}),
        Analysis.CLUSTERING: frozenset({Permission.TRANSFORM}),
        Analysis.ANNOTATION: frozenset({Permission.TRANSFORM, Permission.INFERENCE}),
        Analysis.COMPOSITION: frozenset({Permission.TRANSFORM}),
        Analysis.DIFFERENTIAL_EXPRESSION: frozenset(),
        Analysis.TRAJECTORY: frozenset(),
        Analysis.CELL_CELL_COMMUNICATION: frozenset(),
    },
    str(QCStateInitial.QUARANTINE): {
        Analysis.MANIFOLD: frozenset({Permission.TRANSFORM}),
        Analysis.CLUSTERING: frozenset(),
        Analysis.ANNOTATION: frozenset(),
        Analysis.COMPOSITION: frozenset(),
        Analysis.DIFFERENTIAL_EXPRESSION: frozenset(),
        Analysis.TRAJECTORY: frozenset(),
        Analysis.CELL_CELL_COMMUNICATION: frozenset(),
    },
}


_MULTIPLET_REVOKES: frozenset[Analysis] = frozenset(
    {
        Analysis.MANIFOLD,
        Analysis.CLUSTERING,
        Analysis.COMPOSITION,
        Analysis.DIFFERENTIAL_EXPRESSION,
        Analysis.TRAJECTORY,
        Analysis.CELL_CELL_COMMUNICATION,
    }
)


@dataclass(frozen=True)
class EligibilityMasks:
    """Per-cell boolean masks, one per (analysis, permission) that any state grants.

    Args:
        masks: Mapping from column name to per-cell boolean mask.
        state: The states the masks were derived from, carried for provenance.
    """

    masks: dict[str, pd.Series]
    state: pd.Series

    @staticmethod
    def column_name(analysis: Analysis, permission: Permission) -> str:
        """Canonical obs column name for one permission on one analysis."""
        return f"qc_{permission}_{analysis}"

    def mask(self, analysis: Analysis, permission: Permission) -> pd.Series:
        """The mask for one permission on one analysis.

        Raises:
            KeyError: If no state grants that combination in :data:`_ELIGIBILITY`.
        """
        name = self.column_name(analysis, permission)
        if name in self.masks:
            return self.masks[name]

        grantable = any(
            permission in per_analysis.get(analysis, frozenset())
            for per_analysis in _ELIGIBILITY.values()
        )
        if not grantable:
            raise KeyError(
                f"No QC state grants {permission!s} on {analysis!s}, so '{name}' can never "
                f"exist. Available: {sorted(self.masks)}"
            )
        return pd.Series(False, index=self.state.index, name=name)

    def is_empty(self, analysis: Analysis, permission: Permission) -> bool:
        """Whether a grantable permission is held by no cell on this run."""
        return not bool(self.mask(analysis, permission).any())

    def to_obs_frame(self) -> pd.DataFrame:
        """Flatten to ``adata.obs`` columns."""
        return pd.DataFrame(self.masks, index=self.state.index)

    def summary(self) -> dict[str, int]:
        """Eligible cell count per mask, for provenance and the run report."""
        return {name: int(mask.sum()) for name, mask in sorted(self.masks.items())}


def build_eligibility_masks(
    state: pd.Series,
    *,
    probable_multiplet: pd.Series | None = None,
) -> EligibilityMasks:
    """Derive per-analysis eligibility from graded QC states.

    Args:
        state: :class:`QCStateInitial` value per cell.
        probable_multiplet: Per-cell multiplet flag. A multiplet keeps its damage-based
            permissions but loses every analysis that counts or compares cells, because it
            is not one cell.

    Returns:
        The masks, one column per granted (analysis, permission) pair.
    """
    text = state.astype(str)
    multiplet = (
        probable_multiplet.reindex(state.index).fillna(False).astype(bool)
        if probable_multiplet is not None
        else pd.Series(False, index=state.index)
    )

    masks: dict[str, pd.Series] = {}
    for analysis in Analysis:
        for permission in Permission:
            granted = pd.Series(False, index=state.index)
            for state_value, per_analysis in _ELIGIBILITY.items():
                if permission in per_analysis.get(analysis, frozenset()):
                    granted |= text == state_value

            if not any(
                permission in per_analysis.get(analysis, frozenset())
                for per_analysis in _ELIGIBILITY.values()
            ):
                continue

            if analysis in _MULTIPLET_REVOKES:
                granted &= ~multiplet

            masks[EligibilityMasks.column_name(analysis, permission)] = granted

    return EligibilityMasks(masks=masks, state=text)


def fit_mask(state: pd.Series, analysis: Analysis) -> pd.Series:
    """Cells permitted to fit ``analysis`` — the mask a cohort statistic must respect."""
    return build_eligibility_masks(state).mask(analysis, Permission.FIT)


def fitting_cells(
    obs: pd.DataFrame,
    analysis: Analysis = Analysis.MANIFOLD,
) -> pd.Series | None:
    """The fit population a stage must estimate cohort statistics from, or None.

    Args:
        obs: The ``adata.obs`` frame to read the mask from.
        analysis: Which analysis's fit permission is being claimed. Defaults to
            ``MANIFOLD``, which governs the normalize → HVG → scale → PCA → integrate chain.

    Returns:
        A boolean per-cell mask, or ``None`` when the stage should fit on every cell.
    """
    column = EligibilityMasks.column_name(analysis, Permission.FIT)
    if column not in obs.columns:
        if "qc_state_initial" in obs.columns:
            raise CellQuorumDataError(
                f"QC state is present but its fitting mask '{column}' is missing."
            )
        return None

    if not pd.api.types.is_bool_dtype(obs[column].dtype) or obs[column].isna().any():
        raise CellQuorumDataError(f"QC fitting mask '{column}' must contain non-missing booleans.")
    mask = obs[column].astype(bool)
    if not bool(mask.any()):
        raise CellQuorumDataError(
            f"QC fitting mask '{column}' permits no cells. Review QC decisions before fitting; "
            "excluded cells cannot be used as a fallback."
        )
    return mask


__all__ = [
    "Analysis",
    "EligibilityMasks",
    "Permission",
    "build_eligibility_masks",
    "fit_mask",
    "fitting_cells",
]
