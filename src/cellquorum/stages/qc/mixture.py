"""Mixture-model mitochondrial QC for CellQuorum.

Why this module exists: every other way of setting a mitochondrial cut-off in this
pipeline answers the wrong question.

A fixed ceiling asks "what percentage is too high?", which has no data-driven
answer -- baseline mitochondrial fraction varies with cell type and dissociation
protocol, and on the skin atlas the per-sample median spans 0.76% to 5.27%, so no
single constant is right for all 18 samples.

A MAD threshold asks "how far from the median is unusual?", which assumes the
metric is unimodal noise around one healthy centre. Mitochondrial fraction is not:
it is a MIXTURE of a healthy mode and a damaged tail. A location-scale estimator
on a mixture reports the spread of the healthy mode, so it TIGHTENS as a sample
gets cleaner -- on the same atlas it produced a 2.0% ceiling for the cleanest
sample and 11.2% for the dirtiest, which is exactly backwards.

This module asks the question that has an answer: "is this cell better explained
by the intact population or the damaged one?" It implements the model of Hippen et
al. (miQC, PLoS Comput Biol 2021, doi:10.1371/journal.pcbi.1009290) as a mixture
of two linear regressions of mitochondrial percentage on library complexity, fit
per sample by expectation-maximisation. The two components encode the actual
biology of a dying cell: when the membrane is compromised, cytoplasmic mRNA leaks
out, so FEWER genes are detected while the mitochondrial fraction RISES. Intact
cells show no such relationship. So the damaged component is the one with the
higher intercept, and each cell gets a posterior probability of belonging to it.

The relationship this relies on is present in all 18 atlas samples: mitochondrial
percentage correlates with detected genes at -0.10 to -0.43 within sample.

Implemented natively rather than by calling the R package because QC is the one
stage every single run executes, and making it depend on R would put an
interpreter and a Bioconductor tree in the critical path of every pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc.config import QCMitoMixtureConfig

MIQC_PROBABILITY_COLUMN = "miqc_prob_compromised"


# Published for the record and never thresholded. MIQC_PROBABILITY_COLUMN has had the
# two post-processing rules folded into it as 0.0 and 1.0 values, and under projection is
# a step function of mitochondrial percentage, so it cannot show what the mixture actually
# believed about a cell. This one can: it is the only column that still varies with
# library complexity, which is the entire reason for preferring a mixture model to a
# ceiling in the first place.
MIQC_POSTERIOR_COLUMN = "miqc_posterior_compromised"


MIQC_RULE_NAME = "mixture_mito_compromised"


MIQC_CEILING_RULE_NAME = "mixture_max_mito_percent"


# Share of a group's cells that may be decided differently by the projected ceiling than
# by the model itself before the projection is reported as unfaithful. A faithful
# projection disagrees with the model on almost nothing, because the model's decision is
# already essentially a mitochondrial cut. A large disagreement means the mixture
# separated that group on library complexity instead, so the ceiling is a poor summary AND
# the underlying fit was not measuring viability. This is a reporting level, not a decision
# threshold -- nothing changes when it is crossed except that a warning is emitted.
PROJECTION_WARN_DISAGREEMENT = 0.02


N_COMPONENTS = 2


class QCMixtureError(CellQuorumDataError):
    """Report mitochondrial mixture-model failures that cannot be worked around.

    Fit failures are deliberately NOT routed here. A model that will not converge is
    reported as a warning and the affected cells are kept, because deleting a sample's
    worth of data on the strength of a numerical failure is never the right response. This
    exception is reserved for inputs that make the request itself incoherent, such as a
    missing metric column.
    """


@dataclass(frozen=True)
class MitoMixtureModel:
    """Store one fitted two-component mixture, for one group of cells.

    Args:
        group: Group label the model was fit on, or ``"all"`` when pooled.
        n_cells: Number of cells used in the fit.
        converged: Whether expectation-maximisation reached the tolerance.
        n_iterations: Iterations used.
        log_likelihood: Final log-likelihood of the retained fit.
        compromised_weight: Mixing weight of the compromised component.
        compromised_intercept: Intercept of the compromised component.
        compromised_slope: Slope of the compromised component.
        intact_intercept: Intercept of the intact component.
        intact_slope: Slope of the intact component.
        compromised_variance: Residual variance of the compromised component.
        intact_variance: Residual variance of the intact component.
        n_compromised: Cells whose adjusted probability exceeds the cut-off.
        n_raw_compromised: Cells the RAW posterior exceeds the cut-off for,
            before either post-processing rule. The gap between this and
            ``n_compromised`` is how much of the filter is the model and how much
            is policy -- a distinction that is otherwise invisible, and the first
            thing to look at when a mitochondrial filter behaves unexpectedly.
        n_rescued_below_boundary: Cells ``keep_all_below_boundary`` forced to
            keep, having been assigned to the compromised component despite a
            mitochondrial fraction beneath the intact trend line.
        n_swept_left_cutoff: Cells ``enforce_left_cutoff`` forced to discard,
            being no more complex and no less mitochondrial than an
            already-discarded cell.
        level: Grouping-hierarchy level, 0 being the requested grouping and each
            higher number a coarser fallback.
        n_assigned: Cells this model actually scored, which is fewer than
            ``n_cells`` for a fallback model estimated on a wider group.
        fallback: Reason the group could not be fit, or None on success.
    """

    group: str
    n_cells: int
    converged: bool
    n_iterations: int
    log_likelihood: float
    compromised_weight: float
    compromised_intercept: float
    compromised_slope: float
    intact_intercept: float
    intact_slope: float
    n_compromised: int

    compromised_variance: float = 0.0
    intact_variance: float = 0.0
    n_raw_compromised: int = 0
    n_rescued_below_boundary: int = 0
    n_swept_left_cutoff: int = 0
    level: int = 0
    n_assigned: int = 0
    fallback: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert the model record into a JSON-friendly dictionary.

        Returns:
            Dictionary representation of the fitted model.
        """

        return {
            "group": self.group,
            "n_cells": self.n_cells,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "log_likelihood": self.log_likelihood,
            "compromised_weight": self.compromised_weight,
            "compromised_intercept": self.compromised_intercept,
            "compromised_slope": self.compromised_slope,
            "intact_intercept": self.intact_intercept,
            "intact_slope": self.intact_slope,
            "compromised_variance": self.compromised_variance,
            "intact_variance": self.intact_variance,
            "n_compromised": self.n_compromised,
            "n_raw_compromised": self.n_raw_compromised,
            "n_rescued_below_boundary": self.n_rescued_below_boundary,
            "n_swept_left_cutoff": self.n_swept_left_cutoff,
            "level": self.level,
            "n_assigned": self.n_assigned,
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class MitoCeiling:
    """Store the mitochondrial ceiling one group's fitted model reduces to.

    Args:
        groupby_columns: Columns defining the group, empty when pooled.
        group_values: Values of those columns, empty when pooled.
        group: Readable group label.
        n_cells: Cells in the group with a usable mitochondrial value.
        ceiling: Mitochondrial percentage above which cells are discarded, or
            None when the model flagged nothing and no ceiling applies.
        n_removed: Cells the projected ceiling discards.
        disagreement: Cells decided differently by the ceiling and the model.
        disagreement_fraction: ``disagreement`` as a share of ``n_cells``.
    """

    groupby_columns: tuple[str, ...]
    group_values: tuple[str, ...]
    group: str
    n_cells: int
    ceiling: float | None
    n_removed: int
    disagreement: int
    disagreement_fraction: float

    def to_dict(self) -> dict[str, object]:
        """Convert the ceiling record into a JSON-friendly dictionary.

        Returns:
            Dictionary representation of the projected ceiling.
        """

        return {
            "group": self.group,
            "groupby_columns": list(self.groupby_columns),
            "group_values": list(self.group_values),
            "n_cells": self.n_cells,
            "ceiling": self.ceiling,
            "n_removed": self.n_removed,
            "disagreement": self.disagreement,
            "disagreement_fraction": self.disagreement_fraction,
        }


@dataclass(frozen=True)
class MitoMixtureResult:
    """Store per-cell compromised probabilities and the models that produced them.

    Args:
        probabilities: Adjusted probability of being compromised, per cell. This
            is the value the filter acts on: hardened to 0.0 and 1.0 by the two
            post-processing rules, and after projection a pure function of
            mitochondrial percentage.
        posterior: RAW probability of being compromised, per cell, before either
            post-processing rule. The model's own verdict, and the only version
            that still varies with library complexity -- which makes it the one
            worth plotting, and the one that answers "would the model have kept
            this cell if our policy had not intervened".
        models: Fitted model record per group.
        warnings: Non-fatal fitting warnings.
        ceilings: Projected mitochondrial ceiling per group, when the model was
            projected onto the mitochondrial axis.
    """

    probabilities: pd.Series

    posterior: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    models: list[MitoMixtureModel] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ceilings: list[MitoCeiling] = field(default_factory=list)

    def ceilings_to_dataframe(self) -> pd.DataFrame:
        """Convert the projected ceiling records into a table.

        Returns:
            One row per projected group, with an explicit schema when empty.
        """

        if not self.ceilings:
            return pd.DataFrame(
                columns=[
                    "group",
                    "groupby_columns",
                    "group_values",
                    "n_cells",
                    "ceiling",
                    "n_removed",
                    "disagreement",
                    "disagreement_fraction",
                ]
            )

        return pd.DataFrame([ceiling.to_dict() for ceiling in self.ceilings])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the fitted model records into a table.

        Returns:
            One row per group, with an explicit schema when no group was fit.
        """

        if not self.models:
            return pd.DataFrame(
                columns=[
                    "group",
                    "n_cells",
                    "converged",
                    "n_iterations",
                    "log_likelihood",
                    "compromised_weight",
                    "compromised_intercept",
                    "compromised_slope",
                    "intact_intercept",
                    "intact_slope",
                    "compromised_variance",
                    "intact_variance",
                    "n_compromised",
                    "n_raw_compromised",
                    "n_rescued_below_boundary",
                    "n_swept_left_cutoff",
                    "level",
                    "n_assigned",
                    "fallback",
                ]
            )

        return pd.DataFrame([model.to_dict() for model in self.models])


@dataclass(frozen=True)
class _ComponentFit:
    """Store the raw output of one expectation-maximisation run.

    Args:
        coefficients: Per-component intercept and slope, shape (2, 2).
        variances: Per-component residual variance.
        weights: Per-component mixing weight.
        responsibilities: Per-cell component membership probabilities.
        log_likelihood: Final log-likelihood.
        n_iterations: Iterations used.
        converged: Whether the tolerance was reached.
    """

    coefficients: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    responsibilities: np.ndarray
    log_likelihood: float
    n_iterations: int
    converged: bool


def fit_mito_mixture(
    cell_metrics: pd.DataFrame,
    config: QCMitoMixtureConfig,
) -> MitoMixtureResult:
    """Fit the miQC mixture model and return a compromised probability per cell.

    The returned probability is *adjusted*: the two post-processing rules of miQC
    (``keep_all_below_boundary`` and ``enforce_left_cutoff``) are folded into it as
    hard 0.0 and 1.0 values. That is deliberate — it keeps the entire policy in one
    column, so nothing downstream needs a special case for it.

    That column is now consumed as the graded **metabolic/stress** axis rather than as a
    threshold rule. The distinction matters in two ways a reader has to know about:

    * the posterior is a **calibrated probability** and is used at its own scale. Passing it
      through a robust z, as any other severity axis gets, once moved 22,541 cells between
      states — a cell the model gave a 10% chance of being compromised scored 0.59. The way to
      make it mean "compromised *for a cell like this*" is to fit the mixture within lineage,
      which the stage does, never to rescale the output;
    * metabolic evidence is a **supporting** family, so a high posterior can never on its own
      quarantine a cell. It needs concordance from an independent family. That is what stopped
      mitochondrion-rich populations — keratinocytes, mast cells — being removed for their
      biology.

    Cells the model cannot speak to -- a group too small to fit, a fit that will
    not converge, a missing metric value -- receive 0.0, meaning no evidence of compromise. A
    numerical failure must never be the reason a cell is discarded. Note that there is no longer
    a fixed ``max_mito_percent`` backstop behind them: that ceiling was deleted with the
    threshold path, so a cell the mixture cannot model carries no metabolic evidence at all and
    is judged on the other families alone.

    Args:
        cell_metrics: Cell-level QC metric table.
        config: Mitochondrial mixture configuration.

    Returns:
        MitoMixtureResult holding per-cell probabilities, per-group models, and
        warnings.

    Raises:
        QCMixtureError: If a required metric column is absent.
    """

    missing = [
        column
        for column in (config.mito_metric, config.complexity_metric)
        if column not in cell_metrics.columns
    ]
    if missing:
        raise QCMixtureError(
            "Mitochondrial mixture modelling requires metric column(s) "
            f"{missing}, which are absent from cell_metrics. Available columns: "
            f"{sorted(cell_metrics.columns)[:10]}..."
        )

    probabilities = pd.Series(0.0, index=cell_metrics.index, dtype=float)

    posterior = pd.Series(0.0, index=cell_metrics.index, dtype=float)

    models: list[MitoMixtureModel] = []
    warnings: list[str] = []

    mito = pd.to_numeric(cell_metrics[config.mito_metric], errors="coerce")
    complexity = pd.to_numeric(cell_metrics[config.complexity_metric], errors="coerce")

    # `copy=True` is required, not defensive. Under pandas 3 `.to_numpy()` returns a READ-ONLY
    # view of the Series' buffer, so the `pending[...] = False` that retires each fitted group
    # below raised `ValueError: assignment destination is read-only` — sixteen tests, one line.
    # The engine's dependency floors are lower bounds, so CI resolves the newest pandas while a
    # developer environment may sit on 2.x, and the same code passes locally and fails there.
    pending = (np.isfinite(mito) & np.isfinite(complexity)).to_numpy(copy=True)

    resolved_groupby = [column for column in config.groupby if column in cell_metrics.columns]

    accepted_levels: set[int] = set()

    for level, requested in enumerate([config.groupby, *config.fallback_groupby]):
        if not pending.any():
            break

        groupby = [column for column in requested if column in cell_metrics.columns]
        if len(groupby) != len(requested):
            absent = [column for column in requested if column not in cell_metrics.columns]
            warnings.append(
                f"Mitochondrial mixture grouping {requested} names column(s) "
                f"{absent} that are absent from cell_metrics, so cells were "
                f"grouped by {groupby or 'nothing (one pooled model)'} instead. "
                "Grouping must isolate both sample and cell identity; check the "
                "metadata before trusting the result."
            )

        fitted, unfittable = _fit_groups_at_level(
            cell_metrics,
            groupby=groupby,
            level=level,
            pending=pending,
            mito=mito,
            complexity=complexity,
            config=config,
        )

        if unfittable and config.level_policy == "uniform":
            models.extend(outcome.model for outcome in unfittable)
            warnings.append(
                _describe_abandoned_level(
                    groupby=groupby,
                    n_groups=len(fitted) + len(unfittable),
                    unfittable=unfittable,
                    has_next_level=level < len(config.fallback_groupby),
                )
            )
            continue

        for outcome in fitted:
            probabilities.iloc[outcome.positions] = np.asarray(  # type: ignore[call-overload]
                outcome.adjusted, dtype=float
            )
            if outcome.raw is not None:
                posterior.iloc[outcome.positions] = np.asarray(  # type: ignore[call-overload]
                    outcome.raw, dtype=float
                )
            pending[outcome.positions] = False
            models.append(outcome.model)
            warnings.extend(outcome.warnings)
            accepted_levels.add(level)

        models.extend(outcome.model for outcome in unfittable)

        if fitted and config.level_policy == "uniform":
            resolved_groupby = groupby

    if pending.any():
        warnings.append(
            f"{int(pending.sum())} cell(s) received no mitochondrial mixture model "
            "because every grouping level left them in a group too small or "
            "unfittable. They were KEPT, and remain subject to the fixed "
            "mitochondrial ceiling. Add a coarser fallback_groupby to cover them."
        )

    if len(accepted_levels) > 1:
        warnings.append(
            f"Cells were judged by mitochondrial models fit at {len(accepted_levels)} "
            f"different grouping levels {sorted(accepted_levels)} because "
            "level_policy='per_group'. Which cells got the finer model follows "
            "group SIZE, and group size usually tracks the study arm, so this "
            "threshold can act as a design covariate: check the stage's attrition "
            "audit before reporting these counts, or set level_policy='uniform'."
        )

    ceilings: list[MitoCeiling] = []
    if config.monotone_mito_projection:
        ceilings, probabilities, projection_warnings = _project_to_mito_ceilings(
            cell_metrics=cell_metrics,
            probabilities=probabilities,
            mito=mito,
            groupby=resolved_groupby,
            config=config,
        )
        warnings.extend(projection_warnings)

    return MitoMixtureResult(
        probabilities=probabilities,
        posterior=posterior,
        models=models,
        warnings=warnings,
        ceilings=ceilings,
    )


@dataclass(frozen=True)
class _GroupOutcome:
    """Store one group's result at one grouping level, before it is committed.

    Args:
        positions: Integer positions of the cells this outcome would score. Typed as an
            integer array rather than a bare ``ndarray`` because it indexes with ``.iloc``,
            where a float array would silently select nothing on some pandas versions.
        model: Model record, fitted or deferred.
        adjusted: Per-cell adjusted probabilities, or None when deferred.
        raw: Per-cell RAW posteriors, before the post-processing rules, or None
            when deferred. Kept separately so the model's own verdict stays
            recoverable after the policy rules have hardened it to 0.0 and 1.0.
        warnings: Warnings that only apply if the outcome is committed.
    """

    positions: npt.NDArray[np.intp]
    model: MitoMixtureModel
    adjusted: np.ndarray | None
    raw: np.ndarray | None = None
    warnings: tuple[str, ...] = ()


def _fit_groups_at_level(
    cell_metrics: pd.DataFrame,
    *,
    groupby: list[str],
    level: int,
    pending: np.ndarray,
    mito: pd.Series,
    complexity: pd.Series,
    config: QCMitoMixtureConfig,
) -> tuple[list[_GroupOutcome], list[_GroupOutcome]]:
    """Fit every group at one grouping level without applying any of the results.

    Args:
        cell_metrics: Cell-level QC metric table.
        groupby: Grouping columns for this level, possibly empty.
        level: Hierarchy level, 0 being the requested grouping.
        pending: Boolean mask of cells still awaiting a model.
        mito: Numeric mitochondrial metric aligned to ``cell_metrics``.
        complexity: Numeric complexity metric aligned to ``cell_metrics``.
        config: Mitochondrial mixture configuration.

    Returns:
        Outcomes that produced a fit, and outcomes that could not be fit. Groups
        with no pending cells appear in neither.
    """

    fitted: list[_GroupOutcome] = []
    unfittable: list[_GroupOutcome] = []

    mito_values = mito.to_numpy()
    complexity_values = complexity.to_numpy()

    for group_label, _values, positions in _iterate_groups(cell_metrics, groupby):
        assign_to = positions[pending[positions]]

        if assign_to.size == 0:
            continue

        usable = positions[np.isfinite(mito_values[positions])]
        usable = usable[np.isfinite(complexity_values[usable])]

        if usable.size < config.min_cells:
            unfittable.append(
                _GroupOutcome(
                    positions=assign_to,
                    model=_unfiltered_model(
                        group_label,
                        int(usable.size),
                        f"only {usable.size} usable cells, below min_cells={config.min_cells}",
                        level=level,
                        n_assigned=0,
                    ),
                    adjusted=None,
                )
            )
            continue

        fit = _fit_two_component_regression(
            complexity_values[usable],
            mito_values[usable],
            config=config,
        )

        if fit is None:
            unfittable.append(
                _GroupOutcome(
                    positions=assign_to,
                    model=_unfiltered_model(
                        group_label,
                        int(usable.size),
                        "expectation-maximisation did not produce a usable two-component fit",
                        level=level,
                        n_assigned=0,
                    ),
                    adjusted=None,
                )
            )
            continue

        # Identify which component describes the compromised cells. Following miQC, that
        # is the component with the higher intercept: at equal library complexity a
        # damaged cell carries the larger mitochondrial fraction.
        compromised = int(np.argmax(fit.coefficients[:, 0]))
        intact = 1 - compromised

        raw = fit.responsibilities[np.isin(usable, assign_to), compromised]
        hardening = _adjust_probabilities(
            raw=raw,
            mito=mito_values[assign_to],
            complexity=complexity_values[assign_to],
            intact_coefficients=fit.coefficients[intact],
            config=config,
        )
        adjusted = hardening.adjusted

        fit_warnings: list[str] = []

        if not fit.converged:
            fit_warnings.append(
                f"Mitochondrial mixture model for group '{group_label}' hit the "
                f"{config.max_iterations}-iteration cap without converging; its "
                "parameters were used as-is."
            )

        if fit.coefficients[compromised, 1] > 0:
            fit_warnings.append(
                f"Mitochondrial mixture model for group '{group_label}' assigned "
                "the compromised component a POSITIVE complexity slope "
                f"({fit.coefficients[compromised, 1]:.4g}), which inverts the "
                "expected damaged-cell relationship. Inspect this group before "
                "trusting its filtering."
            )

        fitted.append(
            _GroupOutcome(
                positions=assign_to,
                model=MitoMixtureModel(
                    group=group_label,
                    n_cells=int(usable.size),
                    converged=fit.converged,
                    n_iterations=fit.n_iterations,
                    log_likelihood=float(fit.log_likelihood),
                    compromised_weight=float(fit.weights[compromised]),
                    compromised_intercept=float(fit.coefficients[compromised, 0]),
                    compromised_slope=float(fit.coefficients[compromised, 1]),
                    intact_intercept=float(fit.coefficients[intact, 0]),
                    intact_slope=float(fit.coefficients[intact, 1]),
                    compromised_variance=float(fit.variances[compromised]),
                    intact_variance=float(fit.variances[intact]),
                    n_compromised=int((adjusted > config.posterior_cutoff).sum()),
                    n_raw_compromised=hardening.n_raw_compromised,
                    n_rescued_below_boundary=hardening.n_rescued_below_boundary,
                    n_swept_left_cutoff=hardening.n_swept_left_cutoff,
                    level=level,
                    n_assigned=int(assign_to.size),
                ),
                adjusted=adjusted,
                raw=raw,
                warnings=tuple(fit_warnings),
            )
        )

    return fitted, unfittable


def _describe_abandoned_level(
    *,
    groupby: list[str],
    n_groups: int,
    unfittable: list[_GroupOutcome],
    has_next_level: bool,
) -> str:
    """Explain why a whole grouping level was discarded rather than partly used.

    Args:
        groupby: Grouping columns of the abandoned level.
        n_groups: Groups that had cells to score at that level.
        unfittable: Outcomes for the groups that could not be fit.
        has_next_level: Whether a coarser fallback level remains to try.

    Returns:
        One warning message naming the level, the blocking groups, and the reason.
    """

    blockers = "; ".join(
        f"{outcome.model.group}: {outcome.model.fallback}" for outcome in unfittable[:5]
    )
    if len(unfittable) > 5:
        blockers += f"; and {len(unfittable) - 5} more"

    consequence = (
        "so the whole dataset dropped to the next fallback_groupby level"
        if has_next_level
        else (
            "and there is no coarser level left, so NO cell was filtered by the "
            "model. Add 'fallback_groupby: [[]]' -- a pooled level has one group, "
            "so it resolves for every cell or for none"
        )
    )

    return (
        f"Mitochondrial mixture level {groupby or 'pooled'} could not fit "
        f"{len(unfittable)} of {n_groups} group(s) ({blockers}), {consequence}. "
        "Every cell is judged at the SAME grouping level on purpose: mixing "
        "levels makes the threshold depend on group size, which usually tracks "
        "the study arm. Set level_policy='per_group' to allow mixing."
    )


def _project_to_mito_ceilings(
    cell_metrics: pd.DataFrame,
    probabilities: pd.Series,
    mito: pd.Series,
    groupby: list[str],
    config: QCMitoMixtureConfig,
) -> tuple[list[MitoCeiling], pd.Series, list[str]]:
    """Reduce each group's fitted posterior to a single mitochondrial ceiling.

    The posterior is a function of mitochondrial fraction *and* library complexity, which
    means it can discard a cell at 1.7% mitochondrial reads while keeping one at 2.5% -- it
    is then not a mitochondrial rule at all, but a depth rule wearing one's clothes, and
    depth is already filtered separately. Projecting removes that failure mode by
    construction: the rule becomes "discard above X% mitochondrial reads", monotone in the
    metric it names.

    The ceiling is the value that best reproduces the model's own calls, chosen by
    minimising disagreement with them, with ties broken towards the larger ceiling so the
    projection never removes more than the model asked for. The minimising value is found
    in one pass over the sorted metric: for a candidate ``t``, disagreement is
    ``2 * (removed at or below t) - (cells at or below t) + (kept overall)``.

    Args:
        cell_metrics: Cell-level QC metric table.
        probabilities: Adjusted per-cell compromised probabilities.
        mito: Numeric mitochondrial metric aligned to ``cell_metrics``.
        groupby: Grouping the accepted models were fit at, which is the grouping
            the ceilings are reported at. Projecting a pooled model onto per-group
            ceilings would put a different number on each group without any
            per-group parameter behind it.
        config: Mitochondrial mixture configuration.

    Returns:
        Ceiling records, hardened per-cell probabilities, and warnings.
    """

    mito_values = mito.to_numpy(dtype=float)
    removed = probabilities.to_numpy(dtype=float) > config.posterior_cutoff
    projected = np.zeros(len(cell_metrics), dtype=float)

    ceilings: list[MitoCeiling] = []
    warnings: list[str] = []

    for group_label, values, positions in _iterate_groups(cell_metrics, groupby):
        usable = positions[np.isfinite(mito_values[positions])]

        group_mito = mito_values[usable]
        group_removed = removed[usable]

        if usable.size == 0 or not group_removed.any():
            ceilings.append(
                MitoCeiling(
                    groupby_columns=tuple(groupby),
                    group_values=values,
                    group=group_label,
                    n_cells=int(usable.size),
                    ceiling=None,
                    n_removed=0,
                    disagreement=int(group_removed.sum()),
                    disagreement_fraction=(
                        float(group_removed.sum() / usable.size) if usable.size else 0.0
                    ),
                )
            )
            continue

        order = np.argsort(group_mito, kind="stable")
        sorted_mito = group_mito[order]
        sorted_removed = group_removed[order]

        cumulative_cells = np.arange(1, usable.size + 1, dtype=float)
        cumulative_removed = np.cumsum(sorted_removed, dtype=float)

        total_kept = float(usable.size) - float(sorted_removed.sum())
        disagreement = 2.0 * cumulative_removed - cumulative_cells + total_kept

        last_of_value = np.append(np.diff(sorted_mito) > 0, True)
        candidates = np.flatnonzero(last_of_value)

        reversed_scores = disagreement[candidates][::-1]
        best = candidates[len(candidates) - 1 - int(np.argmin(reversed_scores))]
        ceiling = float(sorted_mito[best])

        group_projected = group_mito > ceiling
        projected[usable] = group_projected.astype(float)

        n_disagreement = int((group_projected != group_removed).sum())

        ceilings.append(
            MitoCeiling(
                groupby_columns=tuple(groupby),
                group_values=values,
                group=group_label,
                n_cells=int(usable.size),
                ceiling=ceiling,
                n_removed=int(group_projected.sum()),
                disagreement=n_disagreement,
                disagreement_fraction=float(n_disagreement / usable.size),
            )
        )

        if n_disagreement / usable.size > PROJECTION_WARN_DISAGREEMENT:
            warnings.append(
                f"Mitochondrial ceiling {ceiling:.4g}% for group '{group_label}' "
                f"disagrees with its own mixture model on {n_disagreement} of "
                f"{usable.size} cells "
                f"({100 * n_disagreement / usable.size:.1f}%). The model separated "
                "this group substantially on library complexity rather than "
                "mitochondrial content, so treat its filtering as unreliable and "
                "inspect the group directly."
            )

    return (
        ceilings,
        pd.Series(projected, index=cell_metrics.index, dtype=float),
        warnings,
    )


def _iterate_groups(
    cell_metrics: pd.DataFrame, groupby: list[str]
) -> Iterator[tuple[str, tuple[str, ...], np.ndarray]]:
    """Iterate over fitting groups as (label, values, positional index) triples.

    Args:
        cell_metrics: Cell-level QC metric table.
        groupby: Grouping columns, possibly empty for a pooled fit.

    Yields:
        Group label, the group's column values, and the ascending positions of
        that group's cells.
    """

    if not groupby:
        yield "all", (), np.arange(len(cell_metrics))
        return

    grouped = cell_metrics.groupby(groupby, observed=True, sort=True)

    for key, positions in grouped.indices.items():
        values = tuple(str(part) for part in key) if isinstance(key, tuple) else (str(key),)

        yield "|".join(values), values, np.sort(np.asarray(positions, dtype=int))


def _unfiltered_model(
    group: str,
    n_cells: int,
    reason: str,
    *,
    level: int,
    n_assigned: int,
) -> MitoMixtureModel:
    """Build a model record for a group that could not be fit at this level.

    Args:
        group: Group label.
        n_cells: Usable cell count.
        reason: Why the group was not fit.
        level: Grouping-hierarchy level, 0 being the requested grouping.
        n_assigned: Cells this record assigned probabilities to.

    Returns:
        Model record with NaN parameters and the reason recorded.
    """

    return MitoMixtureModel(
        group=group,
        n_cells=n_cells,
        converged=False,
        n_iterations=0,
        log_likelihood=float("nan"),
        compromised_weight=float("nan"),
        compromised_intercept=float("nan"),
        compromised_slope=float("nan"),
        intact_intercept=float("nan"),
        intact_slope=float("nan"),
        n_compromised=0,
        level=level,
        n_assigned=n_assigned,
        fallback=reason,
    )


@dataclass(frozen=True)
class _HardeningOutcome:
    """Store the effect of the two post-processing rules on one group's posteriors.

    Args:
        adjusted: Adjusted probabilities, 0.0 forcing keep and 1.0 forcing removal.
        n_raw_compromised: Cells the raw posterior exceeded the cut-off for.
        n_rescued_below_boundary: Cells rule one forced back to keep.
        n_swept_left_cutoff: Cells rule two forced to discard.
    """

    adjusted: np.ndarray
    n_raw_compromised: int
    n_rescued_below_boundary: int
    n_swept_left_cutoff: int


def _adjust_probabilities(
    *,
    raw: np.ndarray,
    mito: np.ndarray,
    complexity: np.ndarray,
    intact_coefficients: np.ndarray,
    config: QCMitoMixtureConfig,
) -> _HardeningOutcome:
    """Apply the two miQC post-processing rules to raw posterior probabilities.

    Args:
        raw: Posterior probability of the compromised component, per cell.
        mito: Mitochondrial percentage, per cell.
        complexity: Library complexity, per cell.
        intact_coefficients: Intercept and slope of the intact component.
        config: Mitochondrial mixture configuration.

    Returns:
        _HardeningOutcome holding the adjusted probabilities and how many cells
        each rule moved.
    """

    adjusted = np.asarray(raw, dtype=float).copy()

    discarded_raw = adjusted > config.posterior_cutoff
    n_raw_compromised = int(discarded_raw.sum())
    n_rescued = 0
    n_swept = 0

    # Keep every cell sitting below the intact population's own trend line. Without this,
    # a cell can be assigned to the compromised component purely because it is far out
    # along the complexity axis, even though its mitochondrial fraction is lower than the
    # healthy model predicts for it. Nothing about such a cell is damaged.
    if config.keep_all_below_boundary:
        boundary = intact_coefficients[0] + intact_coefficients[1] * complexity
        below = mito < boundary
        n_rescued = int((discarded_raw & below).sum())
        adjusted[below] = 0.0

    # Enforce monotonicity in the discard region. Having decided to discard some cell, it
    # is incoherent to keep another cell that is BOTH shallower and higher in mitochondrial
    # fraction. This finds the lowest mitochondrial percentage among the discarded cells
    # and discards everything at or above it that is no more complex.
    if config.enforce_left_cutoff:
        discarded = adjusted > config.posterior_cutoff
        if bool(discarded.any()):
            # Find the lowest mitochondrial percentage that is being discarded.
            min_discarded_mito = float(mito[discarded].min())

            # Take the least complex of the cells tied at that percentage, which extends
            # the discard region as little as possible.
            tied = discarded & (mito == min_discarded_mito)
            complexity_cutoff = float(complexity[tied].min())

            swept = (complexity <= complexity_cutoff) & (mito >= min_discarded_mito)
            n_swept = int((swept & ~discarded).sum())
            adjusted[swept] = 1.0

    return _HardeningOutcome(
        adjusted=adjusted,
        n_raw_compromised=n_raw_compromised,
        n_rescued_below_boundary=n_rescued,
        n_swept_left_cutoff=n_swept,
    )


def _fit_two_component_regression(
    complexity: np.ndarray,
    mito: np.ndarray,
    *,
    config: QCMitoMixtureConfig,
) -> _ComponentFit | None:
    """Fit a two-component mixture of linear regressions by expectation-maximisation.

    Args:
        complexity: Library complexity per cell, the regression predictor.
        mito: Mitochondrial percentage per cell, the regression response.
        config: Mitochondrial mixture configuration.

    Returns:
        The best fit found, or None when no restart produced a usable one.
    """

    if not np.isfinite(np.var(mito)) or np.var(mito) <= 0:
        return None

    design = np.column_stack([np.ones_like(complexity), complexity])

    variance_floor = max(float(np.var(mito)) * 1e-8, np.finfo(float).tiny)

    best: _ComponentFit | None = None

    for restart in range(config.n_restarts):
        responsibilities = _initial_responsibilities(
            design, mito, restart=restart, random_state=config.random_state
        )

        fit = _run_em(
            design,
            mito,
            responsibilities,
            config=config,
            variance_floor=variance_floor,
        )

        if fit is None:
            continue

        if best is None or fit.log_likelihood > best.log_likelihood:
            best = fit

    return best


def _initial_responsibilities(
    design: np.ndarray,
    response: np.ndarray,
    *,
    restart: int,
    random_state: int,
) -> np.ndarray:
    """Build starting responsibilities for one expectation-maximisation restart.

    Args:
        design: Regression design matrix.
        response: Regression response.
        restart: Restart number; restart 0 is the deterministic start.
        random_state: Seed for the randomized restarts.

    Returns:
        Starting responsibility matrix of shape (n_cells, 2).
    """

    n_cells = response.shape[0]

    if restart == 0:
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        above = response > design @ coefficients
        responsibilities = np.where(above[:, None], [0.1, 0.9], [0.9, 0.1]).astype(float)
        return responsibilities

    rng = np.random.default_rng(random_state + restart)
    first = rng.uniform(0.2, 0.8, size=n_cells)
    return np.column_stack([first, 1.0 - first])


def _run_em(
    design: np.ndarray,
    response: np.ndarray,
    responsibilities: np.ndarray,
    *,
    config: QCMitoMixtureConfig,
    variance_floor: float,
) -> _ComponentFit | None:
    """Run expectation-maximisation for a two-component regression mixture.

    Args:
        design: Regression design matrix.
        response: Regression response.
        responsibilities: Starting responsibility matrix.
        config: Mitochondrial mixture configuration.
        variance_floor: Lower bound on each component's residual variance.

    Returns:
        The converged or iteration-capped fit, or None when a component collapsed.
    """

    n_cells = response.shape[0]

    coefficients = np.zeros((N_COMPONENTS, design.shape[1]), dtype=float)
    variances = np.zeros(N_COMPONENTS, dtype=float)
    weights = np.zeros(N_COMPONENTS, dtype=float)

    previous_log_likelihood = -np.inf
    log_likelihood = -np.inf
    converged = False
    iteration = 0

    for _ in range(config.max_iterations):
        iteration += 1

        for component in range(N_COMPONENTS):
            component_weights = responsibilities[:, component]

            if component_weights.sum() < config.min_component_weight * n_cells:
                return None

            root_weights = np.sqrt(component_weights)
            solution, *_ = np.linalg.lstsq(
                design * root_weights[:, None], response * root_weights, rcond=None
            )
            coefficients[component] = solution

            residuals = response - design @ solution
            variances[component] = max(
                float(np.average(residuals**2, weights=component_weights)),
                variance_floor,
            )

            weights[component] = float(component_weights.mean())

        log_density = np.empty((n_cells, N_COMPONENTS), dtype=float)
        for component in range(N_COMPONENTS):
            residuals = response - design @ coefficients[component]
            log_density[:, component] = (
                np.log(weights[component])
                - 0.5 * np.log(2.0 * np.pi * variances[component])
                - residuals**2 / (2.0 * variances[component])
            )

        shift = log_density.max(axis=1, keepdims=True)
        total = shift[:, 0] + np.log(np.exp(log_density - shift).sum(axis=1))

        log_likelihood = float(total.sum())
        responsibilities = np.exp(log_density - total[:, None])

        improvement = abs(log_likelihood - previous_log_likelihood)
        if improvement <= config.tolerance * (abs(log_likelihood) + 1.0):
            converged = True
            break

        previous_log_likelihood = log_likelihood

    if not np.isfinite(log_likelihood) or np.allclose(
        coefficients[0], coefficients[1], rtol=1e-8, atol=1e-12
    ):
        return None

    return _ComponentFit(
        coefficients=coefficients,
        variances=variances,
        weights=weights,
        responsibilities=responsibilities,
        log_likelihood=log_likelihood,
        n_iterations=iteration,
        converged=converged,
    )
