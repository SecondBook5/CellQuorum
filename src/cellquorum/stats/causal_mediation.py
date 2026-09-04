"""Causal mediation for donor-paired single-cell designs.

What this answers
-----------------
A contrast says disease changed a program. Mediation asks *through what*: of the
total change in an outcome program, how much travels through a candidate mediator
program, and how much does not. It decomposes one total effect into the indirect
path (treatment -> mediator -> outcome, the ACME) and the direct remainder, which
is the difference between "these two things both moved" and a mechanism.

The design is the Imai-Keele-Tingley linear decomposition, the same one R's
``mediation`` package fits: a mediator model ``M ~ T + Z`` gives path *a*, an
outcome model ``Y ~ T + M + Z`` gives path *b* and the direct effect *c'*, and
``ACME = a * b``, ``total = c' + a * b``. Nothing here is single-cell-specific;
what is specific is the four guards below, each of which is a way this analysis
goes wrong on data shaped like ours.

Guard 1: the unit is a sample, never a cell
-------------------------------------------
Program scores exist per cell, and a mediation fitted on cells has an n in the
thousands, which makes every path significant. The cells within a donor are not
independent observations of the treatment -- the treatment was applied to the
donor. So this module takes a per-cell frame and aggregates it to one row per
sample *itself*, rather than trusting a caller to have done it. There is
deliberately no cell-level entry point.

Guard 2: paired samples are not independent samples either
----------------------------------------------------------
After aggregation a nine-donor paired cohort has eighteen rows -- and standard
mediation inference, quasi-Bayesian or bootstrap, treats those eighteen as
eighteen independent draws. They are nine, measured twice.

So the default inference resamples **donors**, not rows: each bootstrap draw takes
whole donors with replacement, carrying both of a donor's conditions together, and
both models are refitted on the resample.

Which direction that moves an interval depends on the design, and both directions
are real. Where a donor contributes several libraries per condition, the
unclustered fit inflates n outright and its interval is too tight. Where the design
is strictly paired, the opposite happens: the donor offset cancels inside each
donor's case-minus-control contrast, the unclustered fit cannot see that and
charges between-donor spread to residual noise, so *its* interval is too wide and
respecting the pairing tightens it. The claim being made here is therefore not
"clustered intervals are wider" -- it is that clustered intervals are the ones that
answer the question the design asked.

The naive unclustered p-value is computed too and reported alongside, with a flag
when the two disagree about significance -- in either direction, since a path the
pairing rescues matters as much as one it removes, and neither should require a
reanalysis to notice.

Donor identity is handled in the resampling rather than as a fixed effect on
purpose: nine donor intercepts would consume half the residual degrees of freedom
of an eighteen-row fit, and the paired structure is a property of the sampling.

Guard 3: a mediated *fraction* needs a total effect to be a fraction of
----------------------------------------------------------------------
``proportion_mediated`` is ``ACME / total``. When the total effect cannot be
distinguished from zero, that denominator crosses zero, and the ratio is not a
small number with a wide interval -- it is undefined, and its interval spans
whatever the sampling happened to produce. A published table carrying
"proportion mediated -0.21, CI -9.83 to 6.15" is reporting arithmetic on a
denominator near zero, not a weakly-mediated effect. Those rows are returned as
NaN with the reason stated, and the ACME itself is left intact, since it remains
interpretable in its own units.

Guard 4: mediator and outcome must not be built from the same genes
-------------------------------------------------------------------
If the mediator score and the outcome score are computed from overlapping gene
sets, path *b* is partly definitional: the mediator predicts the outcome because
it partly *is* the outcome. This is easy to do by accident, because the natural
mediators are pathway leading edges and leading edges of related pathways share
genes. Pass the two feature lists and the overlap is measured and graded
(``disjoint`` / ``overlapping`` / ``nested``) in the output. It is graded rather
than refused: an overlap is a caveat on interpretation, and which mediation to
believe is the analyst's call, not this function's -- but it is recorded on every
row so the call cannot be made without seeing it.

The remedy, when a grade comes back other than ``disjoint``, is not to argue about
it: rescore the mediator on its own features *minus* the outcome's, rerun, and
report both. An ACME that survives the removal was never definitional; one that
vanishes was. That is a two-line change at the call site, so it is left there
rather than done implicitly here -- rescoring needs the expression matrix, which
this module deliberately never touches.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from cellquorum.stats.depth_confounding import MIN_PAIRED_BLOCKS
from cellquorum.stats.module_remodeling import bh_fdr

DEFAULT_SEED = 1337

# Bootstrap draws for the donor-clustered interval.
#
# The percentile interval's own Monte-Carlo error is what this buys down; 2000 is
# the conventional floor for a 95% percentile CI (the 2.5% tail is then estimated
# from ~50 draws). It is also what makes the p-value's resolution 1/2000.
DEFAULT_N_BOOT = 2000

# Draws for the unclustered quasi-Bayesian comparison, matching R ``mediation``'s
# default so the two analyses are comparable at the same Monte-Carlo resolution.
DEFAULT_N_SIMS = 1000

# The significance line used only to decide whether clustering CHANGED the call.
# Nothing in the estimates depends on it.
MEDIATION_ALPHA = 0.05

CI_LEVEL = 0.95

# A bootstrap resample can be unusable -- every selected donor from one condition,
# or a mediator with no variance in the draw. Individual failures are skipped, but
# if most draws fail the interval is not an interval, so the fit is reported as
# not estimable instead of summarising fifty survivors as if they were 2000.
_MIN_USABLE_BOOTSTRAP_FRACTION = 0.5

# Order matters: it is the order the terms are reported in, and it matches the
# reference layout (path a, path b, direct, indirect, total, proportion).
MEDIATION_TERMS: tuple[str, ...] = (
    "path_a",
    "path_b",
    "direct",
    "acme",
    "total",
    "proportion_mediated",
)

MEDIATION_COLUMNS: tuple[str, ...] = (
    "group",
    "mediator",
    "outcome",
    "term",
    "estimate",
    "ci_low",
    "ci_high",
    "p_value",
    "p_value_unclustered",
    "clustering_changes_the_call",
    "n_samples",
    "n_donors",
    "n_case",
    "n_control",
    "shared_features",
    "feature_overlap",
    "circularity",
    "method",
    "reason",
)


def _percentile_interval(draws: np.ndarray) -> tuple[float, float]:
    """Percentile CI at :data:`CI_LEVEL` from a draw distribution."""
    if draws.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - CI_LEVEL) / 2.0 * 100.0
    return float(np.percentile(draws, tail)), float(np.percentile(draws, 100.0 - tail))


def _two_sided_p(draws: np.ndarray) -> float:
    """Two-sided p-value from a draw distribution, floored at its resolution.

    The proportion of draws on the far side of zero, doubled. Floored at
    ``1 / n_draws`` because zero draws beyond zero means "smaller than this
    procedure can measure", not "zero" -- reporting 0.0 would claim a precision
    the number of draws does not support.
    """
    usable = draws[np.isfinite(draws)]
    if usable.size == 0:
        return float("nan")
    below = float(np.mean(usable <= 0.0))
    above = float(np.mean(usable >= 0.0))
    return float(min(1.0, max(2.0 * min(below, above), 1.0 / usable.size)))


def _design(columns: Sequence[np.ndarray], n: int) -> np.ndarray:
    """Stack an intercept and the given columns into a design matrix."""
    return np.column_stack([np.ones(n), *columns]) if columns else np.ones((n, 1))


def _ols(design: np.ndarray, response: np.ndarray) -> np.ndarray | None:
    """Least-squares coefficients, or None when the fit is not identified."""
    if design.shape[0] <= design.shape[1]:
        return None
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return coefficients


def _ols_with_cov(design: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Coefficients and their covariance, for the quasi-Bayesian draws."""
    coefficients = _ols(design, response)
    if coefficients is None:
        return None
    residual = response - design @ coefficients
    dof = design.shape[0] - design.shape[1]
    sigma2 = float(residual @ residual) / dof
    try:
        gram_inverse = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return None
    return coefficients, sigma2 * gram_inverse


def _decompose(
    treatment: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
    covariates: list[np.ndarray],
) -> dict[str, float] | None:
    """The linear mediation decomposition, or None when either model fails.

    Two fits: ``M ~ T + Z`` for path *a*, and ``Y ~ T + M + Z`` for path *b* and
    the direct effect. The total is reconstructed as ``c' + a*b`` rather than
    refitted as ``Y ~ T + Z``; with covariates the two coincide only up to the
    linear algebra of the same design, and the reconstruction is the one whose
    parts sum to the whole by construction.
    """
    n = treatment.size
    mediator_coefficients = _ols(_design([treatment, *covariates], n), mediator)
    outcome_coefficients = _ols(_design([treatment, mediator, *covariates], n), outcome)
    if mediator_coefficients is None or outcome_coefficients is None:
        return None
    return _terms_from_coefficients(mediator_coefficients, outcome_coefficients)


def _terms_from_coefficients(
    mediator_coefficients: np.ndarray, outcome_coefficients: np.ndarray
) -> dict[str, float]:
    """Assemble the six reported terms from the two models' coefficients."""
    path_a = float(mediator_coefficients[1])
    direct = float(outcome_coefficients[1])
    path_b = float(outcome_coefficients[2])
    acme = path_a * path_b
    total = direct + acme
    proportion = acme / total if total != 0.0 else float("nan")
    return {
        "path_a": path_a,
        "path_b": path_b,
        "direct": direct,
        "acme": acme,
        "total": total,
        "proportion_mediated": float(proportion),
    }


def _grade_overlap(
    mediator_features: Sequence[str] | None, outcome_features: Sequence[str] | None
) -> tuple[int | None, float | None, str | None]:
    """Measure and grade the mediator/outcome feature overlap (guard 4).

    ``nested`` is reserved for the case that makes path *b* least interpretable:
    most of the smaller set is inside the larger one, so the mediator largely *is*
    a piece of the outcome. Any smaller overlap is ``overlapping`` -- a caveat, not
    a disqualification.
    """
    if mediator_features is None or outcome_features is None:
        return None, None, None
    mediator_set = {str(feature) for feature in mediator_features}
    outcome_set = {str(feature) for feature in outcome_features}
    if not mediator_set or not outcome_set:
        return 0, 0.0, "disjoint"
    shared = mediator_set & outcome_set
    union = mediator_set | outcome_set
    jaccard = len(shared) / len(union)
    smaller = min(len(mediator_set), len(outcome_set))
    if not shared:
        grade = "disjoint"
    elif len(shared) / smaller >= 0.5:
        grade = "nested"
    else:
        grade = "overlapping"
    return len(shared), float(jaccard), grade


def _aggregate_to_samples(
    cells: pd.DataFrame,
    *,
    sample: str,
    donor: str,
    treatment: str,
    mediator: str,
    outcome: str,
    covariates: Sequence[str],
    group: str | None,
) -> pd.DataFrame:
    """Collapse a per-cell frame to one row per analysis unit (guard 1).

    The unit is the sample, or the **sample within a group** when ``group`` is given.
    That distinction matters because a group can live at either level. A cohort arm
    (site, batch, treatment protocol) is a property of the whole sample; a cell subtype
    or subcluster is a property of *cells*, so one sample contributes to several
    groups at once. Keying on the pair covers both: for a sample-level group it is
    the same partition as keying on the sample alone, and for a cell-level group it
    is the only key that does not average a subtype together with its neighbours.

    Numeric columns are averaged. Donor and treatment are taken as the unit's single
    value, and a unit carrying more than one is an error rather than a majority vote:
    a sample with two donor labels means the sample column does not identify a
    sample, and averaging across that would silently mix people.
    """
    numeric = [mediator, outcome, *covariates]
    keys = [sample] + ([group] if group else [])

    grouped = cells.groupby(keys, sort=True, observed=True)
    aggregated = grouped[numeric].mean()
    for column in (donor, treatment):
        unique = grouped[column].nunique(dropna=False)
        ambiguous = list(unique[unique > 1].index)
        if ambiguous:
            unit = "sample" if group is None else f"sample x {group}"
            raise ValueError(
                f"'{column}' is not constant within {unit} {ambiguous[:5]}: the sample "
                f"column '{sample}' does not identify a sample, so aggregating would mix "
                f"cells that received different treatments or came from different donors"
            )
        aggregated[column] = grouped[column].first()
    return aggregated.reset_index()


def _not_estimable(context: dict[str, object], reason: str, method: str) -> list[dict[str, object]]:
    """Every term, reported as not estimable, with the reason recorded.

    An empty frame would read as "no mediation here" and a raised exception would
    lose the other groups in the family, so a refused fit is reported as a fit
    that was refused.
    """
    return [
        {
            **context,
            "term": term,
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "p_value_unclustered": float("nan"),
            "clustering_changes_the_call": None,
            "method": method,
            "reason": reason,
        }
        for term in MEDIATION_TERMS
    ]


def _donor_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    donor: str,
    treatment_values: np.ndarray,
    mediator_values: np.ndarray,
    outcome_values: np.ndarray,
    covariate_values: list[np.ndarray],
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample whole donors with replacement and refit (guard 2)."""
    rng = np.random.default_rng(seed)
    donors = frame[donor].to_numpy()
    unique_donors = np.unique(donors)
    rows_by_donor = {value: np.flatnonzero(donors == value) for value in unique_donors}

    collected: dict[str, list[float]] = {term: [] for term in MEDIATION_TERMS}
    for _ in range(n_boot):
        picked = rng.choice(unique_donors, size=unique_donors.size, replace=True)
        rows = np.concatenate([rows_by_donor[value] for value in picked])
        terms = _decompose(
            treatment_values[rows],
            mediator_values[rows],
            outcome_values[rows],
            [values[rows] for values in covariate_values],
        )
        if terms is None:
            continue
        for term, value in terms.items():
            collected[term].append(value)
    return {term: np.asarray(values, dtype=float) for term, values in collected.items()}


def _quasi_bayesian_draws(
    *,
    treatment_values: np.ndarray,
    mediator_values: np.ndarray,
    outcome_values: np.ndarray,
    covariate_values: list[np.ndarray],
    n_sims: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """The unclustered comparison: draw coefficients from their sampling normal.

    This is R ``mediation``'s ``boot = FALSE`` procedure, reproduced so the
    clustered result has something to be compared against. It is deliberately
    *not* the reported inference -- it treats every sample as independent.
    """
    n = treatment_values.size
    mediator_fit = _ols_with_cov(_design([treatment_values, *covariate_values], n), mediator_values)
    outcome_fit = _ols_with_cov(
        _design([treatment_values, mediator_values, *covariate_values], n), outcome_values
    )
    if mediator_fit is None or outcome_fit is None:
        return {term: np.array([]) for term in MEDIATION_TERMS}

    rng = np.random.default_rng(seed)
    mediator_draws = rng.multivariate_normal(mediator_fit[0], mediator_fit[1], size=n_sims)
    outcome_draws = rng.multivariate_normal(outcome_fit[0], outcome_fit[1], size=n_sims)

    collected: dict[str, list[float]] = {term: [] for term in MEDIATION_TERMS}
    for mediator_coefficients, outcome_coefficients in zip(
        mediator_draws, outcome_draws, strict=True
    ):
        for term, value in _terms_from_coefficients(
            mediator_coefficients, outcome_coefficients
        ).items():
            collected[term].append(value)
    return {term: np.asarray(values, dtype=float) for term, values in collected.items()}


def mediation_effects(
    cells: pd.DataFrame,
    *,
    sample: str,
    donor: str,
    treatment: str,
    mediator: str,
    outcome: str,
    case: object,
    control: object,
    group: str | None = None,
    covariates: Sequence[str] = (),
    mediator_features: Sequence[str] | None = None,
    outcome_features: Sequence[str] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    n_sims: int = DEFAULT_N_SIMS,
    min_donors: int = MIN_PAIRED_BLOCKS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Decompose a treatment's effect on an outcome score into mediated and direct parts.

    Aggregates ``cells`` to one row per sample, then per group fits the mediation
    decomposition and reports six terms with donor-clustered percentile intervals.
    See the module docstring for the four guards this applies and why each exists.

    Args:
        cells: Per-cell frame carrying the score columns and the design columns. A
            frame that is already one row per sample also works: aggregation is a
            no-op when ``sample`` is unique.
        sample: Column identifying a donor-condition sample. With no ``group`` this is
            the analysis unit; with one the unit is the sample within the group.
        donor: Column identifying the person. The bootstrap resamples these.
        treatment: Two-level condition column.
        mediator: Numeric score the effect may travel through.
        outcome: Numeric score being explained.
        case: Value of ``treatment`` coded 1 (the disease/treated arm).
        control: Value of ``treatment`` coded 0.
        group: Optional column to fit separately within. May be a sample-level
            attribute (a cohort arm) or a cell-level one (a subtype or subcluster,
            where one sample contributes to several groups); the aggregation keys on
            sample-within-group either way. None fits one model over everything,
            reported as group ``"all"``.
        covariates: Extra numeric columns entered in both models.
        mediator_features: Features the mediator score was built from, for the
            circularity grade. Optional; omitting it leaves the grade blank rather
            than claiming disjointness.
        outcome_features: Features the outcome score was built from.
        n_boot: Donor-clustered bootstrap draws.
        n_sims: Quasi-Bayesian draws for the unclustered comparison.
        min_donors: Fewest donors that may be fitted at all.
        seed: Seed for both draw procedures.

    Returns:
        Long-format frame with :data:`MEDIATION_COLUMNS`, six rows per
        (group, mediator, outcome). Refused fits keep their rows with NaN
        estimates and a stated ``reason``.

    Raises:
        KeyError: A named column is absent.
        ValueError: ``treatment`` carries neither ``case`` nor ``control``, or a
            design column is not constant within a sample.
    """

    required = [sample, donor, treatment, mediator, outcome, *covariates]
    if group:
        required.append(group)
    missing = [column for column in required if column not in cells.columns]
    if missing:
        raise KeyError(f"columns absent from the frame: {missing}")

    samples = _aggregate_to_samples(
        cells,
        sample=sample,
        donor=donor,
        treatment=treatment,
        mediator=mediator,
        outcome=outcome,
        covariates=covariates,
        group=group,
    )

    coded = samples[treatment].map({control: 0.0, case: 1.0})
    if coded.isna().all():
        raise ValueError(
            f"'{treatment}' holds neither {control!r} nor {case!r}; found "
            f"{sorted(map(str, samples[treatment].unique()))[:6]}"
        )
    samples = samples.assign(_treatment_code=coded).dropna(subset=["_treatment_code"])

    shared, overlap, circularity = _grade_overlap(mediator_features, outcome_features)

    group_keys = (
        list(dict.fromkeys(samples[group])) if group else [None]  # preserve appearance order
    )
    records: list[dict[str, object]] = []
    for key in group_keys:
        block = samples if key is None else samples[samples[group] == key]
        context: dict[str, object] = {
            "group": "all" if key is None else str(key),
            "mediator": mediator,
            "outcome": outcome,
            "n_samples": int(len(block)),
            "n_donors": int(block[donor].nunique()),
            "n_case": int((block["_treatment_code"] == 1.0).sum()),
            "n_control": int((block["_treatment_code"] == 0.0).sum()),
            "shared_features": shared,
            "feature_overlap": overlap,
            "circularity": circularity,
        }
        records.extend(
            _fit_one_group(
                block,
                context=context,
                donor=donor,
                mediator=mediator,
                outcome=outcome,
                covariates=covariates,
                n_boot=n_boot,
                n_sims=n_sims,
                min_donors=min_donors,
                seed=seed,
            )
        )

    return pd.DataFrame.from_records(records, columns=list(MEDIATION_COLUMNS))


def _fit_one_group(
    block: pd.DataFrame,
    *,
    context: dict[str, object],
    donor: str,
    mediator: str,
    outcome: str,
    covariates: Sequence[str],
    n_boot: int,
    n_sims: int,
    min_donors: int,
    seed: int,
) -> list[dict[str, object]]:
    """Fit and summarise one group, or record why it could not be fitted."""
    method = "donor_bootstrap"
    n_donors = int(context["n_donors"])  # type: ignore[arg-type]

    if n_donors < min_donors:
        return _not_estimable(
            context,
            f"only {n_donors} donors contribute (floor is {min_donors}); a mediation "
            f"decomposition at this n reports the resampling, not the mechanism",
            method,
        )
    if int(context["n_case"]) < 2 or int(context["n_control"]) < 2:  # type: ignore[arg-type]
        return _not_estimable(
            context,
            f"one arm has fewer than 2 samples ({context['n_case']} case, "
            f"{context['n_control']} control), so no slope in the treatment is identified",
            method,
        )

    treatment_values = block["_treatment_code"].to_numpy(dtype=float)
    mediator_values = block[mediator].to_numpy(dtype=float)
    outcome_values = block[outcome].to_numpy(dtype=float)
    covariate_values = [block[column].to_numpy(dtype=float) for column in covariates]

    point = _decompose(treatment_values, mediator_values, outcome_values, covariate_values)
    if point is None:
        return _not_estimable(
            context,
            "the mediator or outcome model is not identified on these samples "
            "(too few samples for the terms, or a collinear design)",
            method,
        )

    clustered = _donor_bootstrap_draws(
        block,
        donor=donor,
        treatment_values=treatment_values,
        mediator_values=mediator_values,
        outcome_values=outcome_values,
        covariate_values=covariate_values,
        n_boot=n_boot,
        seed=seed,
    )
    usable = clustered["acme"].size
    if usable < _MIN_USABLE_BOOTSTRAP_FRACTION * n_boot:
        return _not_estimable(
            context,
            f"only {usable} of {n_boot} donor resamples could be fitted, so the interval "
            f"would describe the resamples that happened to work rather than the cohort",
            method,
        )

    unclustered = _quasi_bayesian_draws(
        treatment_values=treatment_values,
        mediator_values=mediator_values,
        outcome_values=outcome_values,
        covariate_values=covariate_values,
        n_sims=n_sims,
        seed=seed,
    )

    # Guard 3: is there a total effect for a fraction to be a fraction of?
    total_low, total_high = _percentile_interval(clustered["total"])
    total_spans_zero = not (
        np.isfinite(total_low) and np.isfinite(total_high) and total_low * total_high > 0
    )

    records: list[dict[str, object]] = []
    for term in MEDIATION_TERMS:
        draws = clustered[term]
        low, high = _percentile_interval(draws)
        p_clustered = _two_sided_p(draws)
        p_naive = _two_sided_p(unclustered[term])
        reason = ""

        if term == "proportion_mediated" and total_spans_zero:
            estimate = low = high = p_clustered = p_naive = float("nan")
            reason = (
                "the total effect's interval crosses zero, so the mediated fraction has a "
                "denominator that crosses zero and is not interpretable; read the ACME instead"
            )
            agrees = None
        else:
            estimate = float(point[term])
            agrees = bool(
                (p_clustered < MEDIATION_ALPHA) == (p_naive < MEDIATION_ALPHA)
                if np.isfinite(p_clustered) and np.isfinite(p_naive)
                else True
            )
            if agrees is False:
                reason = (
                    "treating the paired samples as independent changes this call: "
                    f"unclustered p={p_naive:.3g} against donor-clustered p={p_clustered:.3g}"
                )

        records.append(
            {
                **context,
                "term": term,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "p_value": p_clustered,
                "p_value_unclustered": p_naive,
                # Reported as "did clustering change the call", so True is the flag
                # worth looking at rather than the reassuring default.
                "clustering_changes_the_call": (None if agrees is None else not agrees),
                "method": method,
                "reason": reason,
            }
        )
    return records


def mediation_grid(
    cells: pd.DataFrame,
    *,
    mediators: Sequence[str],
    outcome: str,
    features: dict[str, Sequence[str]] | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Run :func:`mediation_effects` for several candidate mediators and stack the result.

    Args:
        cells: Per-cell frame, as for :func:`mediation_effects`.
        mediators: Candidate mediator score columns.
        outcome: The outcome score column, shared across candidates.
        features: Optional map from score column to the features it was built
            from, used for the circularity grade of each mediator against the
            outcome.
        **kwargs: Forwarded to :func:`mediation_effects`.

    Returns:
        The stacked long-format table, with a ``fdr`` column added across the ACME
        tests -- one family, since asking several candidate mediators of one
        outcome is exactly the multiple-comparison situation a mediation table is
        usually published without.
    """

    features = features or {}
    frames = [
        mediation_effects(
            cells,
            mediator=mediator,
            outcome=outcome,
            mediator_features=features.get(mediator),
            outcome_features=features.get(outcome),
            **kwargs,  # type: ignore[arg-type]
        )
        for mediator in mediators
    ]
    # Stacked from records rather than pd.concat: when a mediator's guard columns are
    # entirely absent (no feature lists given) concat warns about all-NA columns and
    # will one day change their dtype underneath us. Records fix the column order and
    # the dtypes are inferred once, over the whole table.
    stacked = pd.DataFrame(
        [row for frame in frames for row in frame.to_dict("records")],
        columns=list(MEDIATION_COLUMNS),
    )

    # FDR over the ACME rows only. The other five terms are components of the same
    # decomposition, not independent hypotheses, and correcting them as a family
    # would be correcting a number against itself.
    stacked["fdr"] = np.nan
    is_acme = stacked["term"] == "acme"
    if is_acme.any():
        stacked.loc[is_acme, "fdr"] = bh_fdr(stacked.loc[is_acme, "p_value"].to_numpy())
    return stacked


__all__ = [
    "CI_LEVEL",
    "DEFAULT_N_BOOT",
    "DEFAULT_N_SIMS",
    "MEDIATION_ALPHA",
    "MEDIATION_COLUMNS",
    "MEDIATION_TERMS",
    "mediation_effects",
    "mediation_grid",
]
