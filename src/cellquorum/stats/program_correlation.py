"""Do these programs move together? Correlation with the n, the covariate, and the overlap.

A program-by-program correlation matrix is the standard second table of any multi-program
analysis, usually printed as ``0.54**`` — a coefficient and a significance mark. It is one
of the easiest tables to produce and one of the easiest to mislead with, in four ways this
module refuses to reproduce.

**The n is the number of independent units, not the number of cells.** ``scores.corr()`` on
a per-cell score matrix will happily return r = 0.05 across 2,000 cells, and a p-value
computed from that n is around 0.02. But 2,000 cells drawn from nine donors are not 2,000
independent observations of anything; the donor is the unit that was sampled. So the unit is
named, not assumed: pass ``sample_col`` and the scores are averaged within each sample
before anything is correlated, and the returned frame records which unit was used and how
many there were. Passing no ``sample_col`` means "the rows I gave you *are* the units", and
the frame says so — there is no path through this function that quietly treats cells as
independent.

**Pooling across conditions manufactures correlation.** Two programs that are both raised in
disease will correlate across samples even when they are uncorrelated within either arm,
because the condition is a common cause of both. That is not co-regulation, it is the
contrast being read twice. So when ``condition_col`` is given, the condition-adjusted partial
correlation is reported beside the raw one, and a pair whose coefficient collapses under
adjustment is visible as such rather than reported as a relationship. The condition is only
the most obvious common cause; ``covariate_cols`` takes the continuous ones — sequencing
depth above all, which raises the detection rate of every gene and therefore every score
built from them — and the frame records what was actually removed rather than leaving a
reader to assume it was the condition alone.

**Two programs that share genes correlate by construction.** If seven of one program's eight
genes are in the other, their scores must track each other, and the resulting r is arithmetic
rather than biology. The shared-gene count is reported per pair when the gene lists are
supplied, so a definitional correlation cannot be read as an empirical one. This is the same
warning :mod:`cellquorum.stats.gene_set_overlap` gives from the other direction, and the two
tables are meant to be read together.

**Too few units is a refusal, not a small p-value.** A correlation over three donors is not
weak evidence, it is no evidence; such rows come back with the coefficient and a ``reason``
in place of a p-value.

The test is two-sided here, unlike the one-sided overlap test: two programs moving in
opposite directions is as interpretable a finding as two moving together, so both tails are
of interest.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cellquorum.stats.module_remodeling import bh_fdr

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

CORRELATION_COLUMNS: tuple[str, ...] = (
    "program_a",
    "program_b",
    "unit",
    "n_units",
    "method",
    "r",
    "p_value",
    "fdr",
    "r_adjusted",
    "p_adjusted",
    "fdr_adjusted",
    "adjusted_for",
    "n_units_adjusted",
    "shared_genes",
    "shares_genes",
    "reason",
)

#: Below this many units a coefficient is reported without a p-value. Four is not a
#: defensible sample size either; it is the point below which the t approximation has no
#: residual degrees of freedom left once a condition covariate is fitted.
MIN_UNITS = 4

_METHODS = ("spearman", "pearson")


def program_correlation_tests(
    scores: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    *,
    sample_col: str | None = None,
    condition_col: str | None = None,
    covariate_cols: Sequence[str] = (),
    method: str = "spearman",
    fdr_method: str = "fdr_bh",
    program_genes: Mapping[str, Iterable[str]] | None = None,
    min_units: int = MIN_UNITS,
) -> pd.DataFrame:
    """
    Correlate every pair of programs at a named unit, with significance and caveats.

    Args:
        scores: Programs in columns. Rows are cells when ``sample_col`` is given, and
            are the units themselves when it is not.
        metadata: Row-aligned with ``scores``; required when ``sample_col`` or
            ``condition_col`` is given.
        sample_col: Column of ``metadata`` naming the independent unit — the donor, the
            sample, the biological replicate. Scores are averaged within each unit
            before correlating. ``None`` declares that ``scores``' rows are already
            units.
        condition_col: Column of ``metadata`` naming the contrast. When given, the
            condition-adjusted partial correlation is reported alongside the raw one.
            Must be constant within each unit.
        covariate_cols: Continuous columns of ``metadata`` to remove alongside the
            condition — library depth, age, anything that could raise both programs at
            once. Averaged within each unit, like the scores. A categorical nuisance
            variable has to be dummy-coded by the caller, because guessing which of its
            levels is the reference would put a modelling choice inside this function.
        method: ``"spearman"`` (default, rank-based, robust to the skew score
            distributions usually have) or ``"pearson"``.
        fdr_method: Passed to :func:`cellquorum.stats.module_remodeling.bh_fdr`.
        program_genes: Optional gene lists, keyed as ``scores``' columns. Supplying them
            adds the shared-gene count per pair — the disclosure that separates a
            definitional correlation from an empirical one.
        min_units: Below this many units, coefficients are reported without p-values.

    Returns:
        One row per unordered pair. ``r``/``p_value``/``fdr`` are the correlation at the
        stated unit; ``r_adjusted``/``p_adjusted``/``fdr_adjusted`` are the same after
        removing the nuisance variables named in ``adjusted_for``, over the
        ``n_units_adjusted`` units that had a value for all of them; ``shared_genes`` and
        ``shares_genes`` are the overlap disclosure; ``reason`` explains any row without a
        p-value. The diagonal is not returned — a program correlates with itself at r = 1,
        and including those rows would inflate the family the FDR corrects over.

    Raises:
        ValueError: Fewer than two programs, an unknown ``method``, a missing
            ``metadata`` when a column was named, a named column absent from it, a
            non-numeric covariate, or a unit that straddles two conditions.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if scores.shape[1] < 2:
        raise ValueError(
            f"need at least 2 programs to correlate, got {scores.shape[1]}: "
            "a pairwise correlation table of one program is empty"
        )

    units, condition, unit_covariates = _to_units(
        scores,
        metadata,
        sample_col=sample_col,
        condition_col=condition_col,
        covariate_cols=covariate_cols,
    )
    unit_label = sample_col or "row"
    n_units = len(units)

    shared = _shared_gene_counts(list(scores.columns), program_genes)
    design, adjusted_names = _adjustment_design(
        condition, unit_covariates, condition_col=condition_col
    )
    # A unit missing a covariate cannot be adjusted, but its scores are still perfectly
    # good for the unadjusted coefficient — so the two coefficients are allowed different
    # n, and the frame says what each one used. Dropping the unit from both would let one
    # missing depth value silently shrink the headline correlation's sample size.
    design_ok = (
        np.isfinite(design).all(axis=1) if design is not None else np.ones(n_units, dtype=bool)
    )
    adjusted_for = ", ".join(adjusted_names)

    rows: list[dict[str, object]] = []
    for name_a, name_b in combinations(list(scores.columns), 2):
        row: dict[str, object] = {
            "program_a": name_a,
            "program_b": name_b,
            "unit": unit_label,
            "n_units": n_units,
            "method": method,
            "r": float("nan"),
            "p_value": float("nan"),
            "fdr": float("nan"),
            "r_adjusted": float("nan"),
            "p_adjusted": float("nan"),
            "fdr_adjusted": float("nan"),
            "adjusted_for": adjusted_for,
            "n_units_adjusted": 0,
            "shared_genes": shared.get((name_a, name_b), -1) if shared else -1,
            "shares_genes": bool(shared.get((name_a, name_b), 0)) if shared else False,
            "reason": "",
        }
        a = units[name_a].to_numpy(dtype=float)
        b = units[name_b].to_numpy(dtype=float)
        usable = np.isfinite(a) & np.isfinite(b)

        if usable.sum() < 3 or _is_constant(a[usable]) or _is_constant(b[usable]):
            row["reason"] = (
                f"not correlatable at the {unit_label} level: "
                f"{int(usable.sum())} unit(s) with a finite value in both programs"
                + (", and one program is constant across them" if usable.sum() >= 3 else "")
            )
            rows.append(row)
            continue

        row["r"], row["p_value"] = _correlate(a[usable], b[usable], method=method)
        if usable.sum() < min_units:
            row["p_value"] = float("nan")
            row["reason"] = (
                f"{int(usable.sum())} {unit_label}(s) is too few to test; the coefficient "
                f"is reported and the p-value withheld (needs {min_units})"
            )
        elif design is not None:
            adjustable = usable & design_ok
            row["n_units_adjusted"] = int(adjustable.sum())
            residual_df = int(adjustable.sum()) - 2 - design.shape[1]
            if residual_df >= 1:
                row["r_adjusted"], row["p_adjusted"] = _partial_correlate(
                    a[adjustable], b[adjustable], design[adjustable], method=method
                )
            else:
                row["reason"] = (
                    f"raw coefficient only: adjusting for {adjusted_for} leaves "
                    f"{residual_df} residual degree(s) of freedom at "
                    f"{int(adjustable.sum())} {unit_label}(s)"
                )
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(CORRELATION_COLUMNS))
    for p_column, fdr_column in (("p_value", "fdr"), ("p_adjusted", "fdr_adjusted")):
        testable = table[p_column].notna()
        if testable.any():
            table.loc[testable, fdr_column] = bh_fdr(
                table.loc[testable, p_column].to_numpy(dtype=float), method=fdr_method
            )
    return table


def _to_units(
    scores: pd.DataFrame,
    metadata: pd.DataFrame | None,
    *,
    sample_col: str | None,
    condition_col: str | None,
    covariate_cols: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame | None]:
    """Collapse to one row per independent unit, carrying the condition and covariates."""
    covariate_cols = [str(name) for name in covariate_cols]
    named = [name for name in (sample_col, condition_col, *covariate_cols) if name is not None]
    if not named:
        return scores.reset_index(drop=True), None, None

    if metadata is None:
        raise ValueError(f"metadata is required to resolve {', '.join(named)}")
    if len(metadata) != len(scores):
        raise ValueError(
            f"metadata has {len(metadata)} rows and scores has {len(scores)}; "
            "they must be row-aligned"
        )
    for name in named:
        if name not in metadata.columns:
            raise ValueError(f"{name!r} is not a column of metadata: {sorted(metadata.columns)}")

    covariates = _numeric_covariates(metadata, covariate_cols)
    if sample_col is None:
        condition = (
            None if condition_col is None else metadata[condition_col].reset_index(drop=True)
        )
        return scores.reset_index(drop=True), condition, covariates

    keys = metadata[sample_col].to_numpy()
    units = scores.set_axis(keys, axis=0).groupby(level=0, sort=True).mean()
    if covariates is not None:
        # The same aggregation the scores get: a per-cell nuisance variable becomes the
        # unit's mean nuisance, which is the level the coefficient is computed at.
        covariates = covariates.set_axis(keys, axis=0).groupby(level=0, sort=True).mean()
        covariates = covariates.reindex(units.index)
    if condition_col is None:
        return units, None, covariates

    per_unit = (
        pd.DataFrame({sample_col: keys, condition_col: metadata[condition_col].to_numpy()})
        .groupby(sample_col, sort=True)[condition_col]
        .agg(lambda values: pd.unique(values.dropna()))
    )
    straddling = [str(unit) for unit, values in per_unit.items() if len(values) > 1]
    if straddling:
        # A donor appearing in both arms is a design error, and averaging over it would
        # hide the very contrast the adjustment is for.
        raise ValueError(
            f"{condition_col!r} is not constant within {sample_col!r} for "
            f"{len(straddling)} unit(s) ({', '.join(straddling[:5])}); "
            "a unit that straddles two conditions cannot be averaged over"
        )
    condition = per_unit.map(lambda values: values[0] if len(values) else None)
    return units, condition.reindex(units.index), covariates


def _numeric_covariates(
    metadata: pd.DataFrame, covariate_cols: Sequence[str]
) -> pd.DataFrame | None:
    """The covariate columns as floats, refusing anything that is not already numeric."""
    if not covariate_cols:
        return None
    columns: dict[str, np.ndarray] = {}
    for name in covariate_cols:
        values = pd.to_numeric(metadata[name], errors="coerce")
        original = metadata[name]
        unconvertible = values.isna() & original.notna()
        if unconvertible.any():
            raise ValueError(
                f"covariate {name!r} is not numeric ({int(unconvertible.sum())} of "
                f"{len(values)} value(s) could not be read as a number); a categorical "
                "nuisance variable has to be dummy-coded by the caller, because choosing "
                "its reference level is a modelling decision"
            )
        columns[str(name)] = values.to_numpy(dtype=float)
    return pd.DataFrame(columns, index=metadata.index).reset_index(drop=True)


def _adjustment_design(
    condition: pd.Series | None,
    covariates: pd.DataFrame | None,
    *,
    condition_col: str | None,
) -> tuple[np.ndarray | None, list[str]]:
    """The nuisance design and the names of what it actually removes.

    A variable that turns out to be constant across the units is dropped rather than
    fitted: it explains nothing and would spend a residual degree of freedom, which at
    nine donors is a quarter of what there is. It is left out of the returned names too,
    so ``adjusted_for`` says what was removed rather than what was requested.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []
    if condition is not None:
        levels = pd.Index(pd.unique(condition.dropna()))
        if len(levels) >= 2:
            # Reference-coded, first level dropped: the intercept is added by the
            # residualiser. A unit whose condition is missing gets all-zero dummies,
            # which is why such units are excluded from the adjusted coefficient rather
            # than quietly counted as the reference arm.
            columns.extend(
                np.where(condition.isna().to_numpy(), np.nan, condition.to_numpy() == level).astype(
                    float
                )
                for level in levels[1:]
            )
            names.append(condition_col or "condition")
    if covariates is not None:
        for name in covariates.columns:
            values = covariates[name].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0 or np.ptp(finite) == 0:
                continue
            columns.append(values)
            names.append(str(name))
    if not columns:
        return None, []
    return np.column_stack(columns), names


def _is_constant(values: np.ndarray) -> bool:
    return bool(np.ptp(values) == 0)


def _correlate(a: np.ndarray, b: np.ndarray, *, method: str) -> tuple[float, float]:
    from scipy.stats import pearsonr, spearmanr

    if method == "spearman":
        result = spearmanr(a, b)
        return float(result.statistic), float(result.pvalue)
    result = pearsonr(a, b)
    return float(result.statistic), float(result.pvalue)


def _partial_correlate(
    a: np.ndarray, b: np.ndarray, design: np.ndarray, *, method: str
) -> tuple[float, float]:
    """Correlate the two vectors after regressing the covariates out of both.

    For Spearman the ranks are taken first and the residualisation is done on them, which
    is the usual construction: the partial of a monotone association, not the association
    of a partialled-out monotone transform.
    """
    from scipy.stats import rankdata, t

    if method == "spearman":
        a, b = rankdata(a), rankdata(b)
    covariates = np.column_stack([np.ones(len(a)), design])
    residual_a = a - covariates @ np.linalg.lstsq(covariates, a, rcond=None)[0]
    residual_b = b - covariates @ np.linalg.lstsq(covariates, b, rcond=None)[0]
    if _is_constant(residual_a) or _is_constant(residual_b):
        return float("nan"), float("nan")

    r = float(np.corrcoef(residual_a, residual_b)[0, 1])
    df = len(a) - 2 - design.shape[1]
    if df < 1 or not np.isfinite(r) or abs(r) >= 1.0:
        return r, float("nan")
    statistic = r * np.sqrt(df / (1.0 - r**2))
    return r, float(2.0 * t.sf(abs(statistic), df))


def _shared_gene_counts(
    programs: list[str], program_genes: Mapping[str, Iterable[str]] | None
) -> dict[tuple[str, str], int]:
    if program_genes is None:
        return {}
    members = {
        str(name): {str(gene) for gene in genes}
        for name, genes in program_genes.items()
        if str(name) in set(programs)
    }
    return {
        (a, b): len(members[a] & members[b])
        for a, b in combinations(programs, 2)
        if a in members and b in members
    }


__all__ = ["CORRELATION_COLUMNS", "MIN_UNITS", "program_correlation_tests"]
