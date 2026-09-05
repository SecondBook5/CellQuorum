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

from cellquorum.stages.qc.evidence import QCStateInitial

# ─── The permissions ────────────────────────────────────────────────────────────────


class Permission(StrEnum):
    """What a cell may do in a given analysis."""

    #: May determine parameters, statistics, or structure.
    FIT = "fit"

    #: May receive a representation or output produced by a model it did not fit.
    TRANSFORM = "transform"

    #: May contribute to a scientific conclusion.
    INFERENCE = "inference"


class Analysis(StrEnum):
    """Analyses whose eligibility is decided separately.

    Separate because they have genuinely different sensitivity to a questionable cell. A
    borderline cell can sit in an embedding for inspection while being excluded from a
    differential test, and one boolean cannot express that.
    """

    MANIFOLD = "manifold"
    CLUSTERING = "clustering"
    ANNOTATION = "annotation"
    COMPOSITION = "composition"
    DIFFERENTIAL_EXPRESSION = "de"
    TRAJECTORY = "trajectory"
    CELL_CELL_COMMUNICATION = "ccc"


#: Which permissions each state carries, per analysis. The single source of truth for
#: "what may this cell do", and deliberately a table rather than scattered conditionals.
#:
#: Reading the rows: ``core`` may do everything. ``borderline`` is projected into the
#: manifold but never fits it, may be annotated by query transfer, contributes to
#: composition only through a sensitivity universe, and never informs DE, trajectory, or
#: communication — those are the inferences a questionable cell would most distort.
#: ``quarantine`` may receive an embedding coordinate so figures can show what was
#: excluded, and informs nothing.
_ELIGIBILITY: dict[str, dict[Analysis, frozenset[Permission]]] = {
    str(QCStateInitial.CORE): {
        analysis: frozenset({Permission.FIT, Permission.TRANSFORM, Permission.INFERENCE})
        for analysis in Analysis
    },
    str(QCStateInitial.BORDERLINE): {
        Analysis.MANIFOLD: frozenset({Permission.TRANSFORM}),
        Analysis.CLUSTERING: frozenset({Permission.TRANSFORM}),
        Analysis.ANNOTATION: frozenset({Permission.TRANSFORM, Permission.INFERENCE}),
        # Sensitivity only: composition is recomputed with and without these cells rather
        # than silently including them.
        Analysis.COMPOSITION: frozenset({Permission.TRANSFORM}),
        Analysis.DIFFERENTIAL_EXPRESSION: frozenset(),
        Analysis.TRAJECTORY: frozenset(),
        Analysis.CELL_CELL_COMMUNICATION: frozenset(),
    },
    str(QCStateInitial.QUARANTINE): {
        # Transform-only on the manifold so a figure can show what was excluded and where.
        Analysis.MANIFOLD: frozenset({Permission.TRANSFORM}),
        Analysis.CLUSTERING: frozenset(),
        Analysis.ANNOTATION: frozenset(),
        Analysis.COMPOSITION: frozenset(),
        Analysis.DIFFERENTIAL_EXPRESSION: frozenset(),
        Analysis.TRAJECTORY: frozenset(),
        Analysis.CELL_CELL_COMMUNICATION: frozenset(),
    },
}

#: A probable multiplet is not a damaged cell, so it keeps its damage-based permissions —
#: except that it is not one biological cell, which disqualifies it from anything counting
#: or comparing cells. It may still be annotated, because knowing *what* was doubleted is
#: how you find out whether one population is being disproportionately called.
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

        Two absences that look identical in ``masks`` are not the same thing, and conflating
        them crashed a run: a pipeline on a degenerate input where **no cell reached core** asked
        for ``qc_fit_manifold``, got ``KeyError``, and failed the stage. "Nobody may fit the
        manifold" is a legitimate — if alarming — outcome that the caller should be able to read
        and report, not an invalid request.

        So the distinction is made against the eligibility table rather than against this run:

        - the pair is grantable by some state, but no cell on this run holds it → all-False,
          because that is the honest answer and the caller can act on it;
        - the pair appears nowhere in the table, e.g. FIT on an analysis no state may ever fit →
          ``KeyError``, because asking is a caller error at any input.

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
        """Whether a grantable permission is held by no cell on this run.

        The condition worth reporting rather than crashing on: ``is_empty(MANIFOLD, FIT)`` means
        the biological reference cannot be built, which a caller should surface as a failed run
        with a reason, not as a ``KeyError`` from an eligibility lookup.
        """
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

            # A permission no state grants is not written at all: an all-False column
            # invites a reader to think the analysis exists and excluded everyone.
            if not bool(granted.any()):
                continue

            if analysis in _MULTIPLET_REVOKES:
                granted &= ~multiplet

            masks[EligibilityMasks.column_name(analysis, permission)] = granted

    return EligibilityMasks(masks=masks, state=text)


def fit_mask(state: pd.Series, analysis: Analysis) -> pd.Series:
    """Cells permitted to fit ``analysis`` — the mask a cohort statistic must respect.

    A convenience for the case that matters most and is easiest to get wrong: estimating a
    normalization target, an HVG dispersion, or a PCA loading. Those are cohort-derived
    quantities used to transform biological data, so they must come from fitting cells even
    though none of them looks like a model.
    """
    return build_eligibility_masks(state).mask(analysis, Permission.FIT)


def fitting_cells(
    obs: pd.DataFrame,
    analysis: Analysis = Analysis.MANIFOLD,
) -> pd.Series | None:
    """The fit population a stage must estimate cohort statistics from, or None.

    This is the read side of the contract: :func:`build_eligibility_masks` writes the
    columns during QC, and every stage that estimates a quantity across cells calls this to
    find out whose cells it may learn from. It lives here, next to the writer, so a stage
    cannot drift on the column name or on what an absent column means.

    Args:
        obs: The ``adata.obs`` frame to read the mask from.
        analysis: Which analysis's fit permission is being claimed. Defaults to
            ``MANIFOLD``, which governs the normalize → HVG → scale → PCA → integrate chain.

    Returns:
        A boolean per-cell mask, or ``None`` when the stage should fit on every cell.

    ``None`` is returned in two distinct situations, both meaning "fit on everything":

    * **QC has not run.** Returning an all-True mask instead would make every downstream
      stage silently depend on a column that need not exist; returning ``None`` keeps a
      dataset that never ran graded QC behaving exactly as it did before.
    * **The fit population is empty.** An all-False mask is a QC misconfiguration, not an
      instruction to fit on zero cells. Fitting on nothing raises deep inside scanpy with
      an unrelated-looking error, so the fallback is the prior behaviour.
    """
    column = EligibilityMasks.column_name(analysis, Permission.FIT)
    if column not in obs.columns:
        return None

    mask = obs[column].astype(bool)
    return mask if bool(mask.any()) else None


__all__ = [
    "Analysis",
    "EligibilityMasks",
    "Permission",
    "build_eligibility_masks",
    "fit_mask",
    "fitting_cells",
]
