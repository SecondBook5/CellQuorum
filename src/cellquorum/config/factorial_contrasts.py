"""Structured factorial contrasts, built from design rows rather than coefficient names.

A user states a biological estimand in factor-level terms ("genotype A vs B at tissue
T"). This module turns it into a weight vector over the fixed-effect design columns by
encoding *hypothetical specimens* with the exact coding the observed design matrix uses::

    simple effect:            L = X(numerator, at) - X(denominator, at)
    difference of differences L = X(na, nb) - X(na, db) - X(da, nb) + X(da, db)

so no backend-generated coefficient name (``genotypepi3k_ko``) is ever parsed or
guessed. The invariant that makes this safe is one coding for three things: the
observed design matrix, the hypothetical design rows, and the contrast weights. It is
carried by :class:`FactorialDesignEncoder`, whose output is checked column-for-column
against :func:`cellquorum.config.design.build_design_matrix`.

Only one-degree-of-freedom, level-vs-level estimands are supported (a three-level factor
supports any chosen pair of its levels); omnibus multi-coefficient tests are deferred.
Everything here is pure Python and independent of any statistical backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cellquorum.config.design import (
    DifferenceOfDifferencesContrast,
    FactorialContrast,
    FactorialDesignConfig,
    SimpleEffectContrast,
    build_design_matrix,
)
from cellquorum.core.exceptions import CellQuorumConfigError

# Numerical tolerance for the estimability (row-space) check.
_ESTIMABLE_ATOL = 1e-8


class ContrastNotEstimable(CellQuorumConfigError):
    """A syntactically valid contrast the observed design cannot estimate.

    Distinct from an invalid declaration (plain ``CellQuorumConfigError``): globally it
    is fatal, but for one cell type whose support shrank it is an ordinary inferential
    failure that callers may turn into a recorded skip.
    """


@dataclass(frozen=True)
class FactorialDesignEncoder:
    """The fixed-effect coding of a factorial design: treatment contrasts vs. references.

    Remembers every factor's full level vocabulary and declared reference, so a one-row
    hypothetical specimen is coded exactly as it would be inside the observed matrix
    (``pd.get_dummies`` on a single row cannot know the other levels).

    Args:
        factors: Main-effect factors, in declaration order.
        levels: Observed levels per factor (sorted).
        references: Reference level per factor (dropped from the coding).
        interactions: Two-way interaction terms.
        columns: Fixed-effect column names, ``Intercept`` first; identical to
            ``build_design_matrix(..., references=...)``.
    """

    factors: tuple[str, ...]
    levels: Mapping[str, tuple[str, ...]]
    references: Mapping[str, str]
    interactions: tuple[tuple[str, str], ...]
    columns: tuple[str, ...]

    @classmethod
    def fit(
        cls, sample_meta: pd.DataFrame, design: FactorialDesignConfig
    ) -> FactorialDesignEncoder:
        """Learn the level vocabulary from specimen metadata.

        Args:
            sample_meta: One row per specimen holding every declared factor.
            design: The factorial design declaration.

        Returns:
            The fitted encoder.

        Raises:
            CellQuorumConfigError: If a factor column is absent or a reference level is
                not an observed level.
        """

        factors = tuple(design.factors)
        absent = [f for f in factors if f not in sample_meta.columns]
        if absent:
            raise CellQuorumConfigError(
                f"Factor column(s) {absent} are absent from specimen metadata."
            )
        levels = {f: tuple(sorted(sample_meta[f].astype(str).unique())) for f in factors}
        references = {f: spec.reference for f, spec in design.factors.items()}
        for f in factors:
            if references[f] not in levels[f]:
                raise CellQuorumConfigError(
                    f"Reference level '{references[f]}' of factor '{f}' is not observed; "
                    f"observed levels: {list(levels[f])}."
                )
        interactions = tuple((a, b) for a, b in design.interactions)

        columns = ["Intercept"]
        for f in factors:
            columns.extend(_dummy_names(f, levels[f], references[f]))
        for a, b in interactions:
            for ca in _dummy_names(a, levels[a], references[a]):
                for cb in _dummy_names(b, levels[b], references[b]):
                    columns.append(f"{ca}:{cb}")
        return cls(
            factors=factors,
            levels=levels,
            references=references,
            interactions=interactions,
            columns=tuple(columns),
        )

    def complete(self, assignment: Mapping[str, str]) -> dict[str, str]:
        """Return a full factor assignment, filling unspecified factors with their reference.

        Reference completion is only meaningful for factors that cancel from a contrast;
        the contrast builder refuses to rely on it for any factor that interacts with a
        compared one.

        Raises:
            CellQuorumConfigError: If a factor is undeclared or a level unobserved.
        """

        full = dict(self.references)
        for factor, level in assignment.items():
            if factor not in self.levels:
                raise CellQuorumConfigError(
                    f"Unknown factor '{factor}'; declared factors: {list(self.factors)}."
                )
            if str(level) not in self.levels[factor]:
                raise CellQuorumConfigError(
                    f"Factor '{factor}' has no level '{level}'; observed levels: "
                    f"{list(self.levels[factor])}."
                )
            full[factor] = str(level)
        return full

    def encode(self, assignment: Mapping[str, str]) -> np.ndarray:
        """Return the fixed-effect design row of a hypothetical specimen.

        Args:
            assignment: Factor -> level; unspecified factors take their reference level.

        Returns:
            A vector aligned to :attr:`columns`.
        """

        full = self.complete(assignment)
        row = dict.fromkeys(self.columns, 0.0)
        row["Intercept"] = 1.0
        for f in self.factors:
            if full[f] != self.references[f]:
                row[f"{f}_{full[f]}"] = 1.0
        for a, b in self.interactions:
            if full[a] != self.references[a] and full[b] != self.references[b]:
                row[f"{a}_{full[a]}:{b}_{full[b]}"] = 1.0
        return np.array([row[c] for c in self.columns], dtype=float)

    def matrix(self, sample_meta: pd.DataFrame) -> pd.DataFrame:
        """Return the observed fixed-effect design matrix for ``sample_meta``.

        Delegates to :func:`build_design_matrix` and verifies it reproduces
        :attr:`columns` exactly, so the observed matrix and the hypothetical rows can
        never drift onto different codings.

        Raises:
            CellQuorumConfigError: If the observed coding disagrees with the encoder's.
        """

        matrix = build_design_matrix(
            sample_meta,
            list(self.factors),
            list(self.interactions),
            references=dict(self.references),
        )
        if tuple(matrix.columns) != self.columns:
            raise CellQuorumConfigError(
                "Internal coding mismatch between the design encoder and the design matrix: "
                f"{list(matrix.columns)} vs {list(self.columns)}."
            )
        return matrix


def _dummy_names(factor: str, levels: Sequence[str], reference: str) -> list[str]:
    """Column names of a factor's treatment dummies (reference dropped, levels sorted)."""
    return [f"{factor}_{lv}" for lv in levels if lv != reference]


@dataclass(frozen=True)
class ContrastComponent:
    """One signed hypothetical specimen entering a contrast.

    Args:
        sign: ``+1`` or ``-1``.
        assignment: The explicit factor levels (compared factors plus ``at``).
        design_row: Its fixed-effect design row, ``column -> value``.
    """

    sign: int
    assignment: dict[str, str]
    design_row: dict[str, float]


@dataclass(frozen=True)
class FactorialContrastResult:
    """A built factorial contrast.

    Args:
        name: Contrast name.
        kind: ``simple_effect`` or ``difference_of_differences``.
        columns: Fixed-effect columns the weights are aligned to.
        weights: One weight per column.
        tested_effect: Human-readable statement of the estimand.
        components: The signed hypothetical design rows summed into ``weights``.
        factors: The compared factor(s).
        levels: Compared levels, e.g. ``{"numerator": ..., "denominator": ...}`` (or the
            ``_a``/``_b`` forms for a difference of differences).
        at: Levels other factors are held at.
    """

    name: str
    kind: str
    columns: tuple[str, ...]
    weights: tuple[float, ...]
    tested_effect: str
    components: tuple[ContrastComponent, ...]
    factors: tuple[str, ...]
    levels: dict[str, str]
    at: dict[str, str] = field(default_factory=dict)

    def support(self, sample_meta: pd.DataFrame, subject_col: str | None) -> dict[str, object]:
        """Describe the biological replication behind this estimand (counts, no test).

        Args:
            sample_meta: One row per (included) specimen.
            subject_col: Subject column, if any.

        Returns:
            Per-component specimen/subject counts and the subjects contributing every
            component (the complete pairs / blocks of the contrast).
        """

        comps: list[dict[str, object]] = []
        subject_sets: list[set[str]] = []
        for comp in self.components:
            mask = np.ones(len(sample_meta), dtype=bool)
            for factor, level in comp.assignment.items():
                mask &= (sample_meta[factor].astype(str) == level).to_numpy()
            rows = sample_meta[mask]
            subjects = set(rows[subject_col].astype(str)) if subject_col is not None else set()
            subject_sets.append(subjects)
            comps.append(
                {
                    "sign": comp.sign,
                    "assignment": dict(comp.assignment),
                    "n_specimens": int(len(rows)),
                    "n_subjects": len(subjects) if subject_col is not None else None,
                }
            )
        complete = (
            sorted(set.intersection(*subject_sets))
            if subject_col is not None and subject_sets
            else []
        )
        return {
            "components": comps,
            "subjects_in_every_component": complete if subject_col is not None else None,
            "n_subjects_in_every_component": len(complete) if subject_col is not None else None,
        }

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "kind": self.kind,
            "columns": list(self.columns),
            "weights": [float(w) for w in self.weights],
            "tested_effect": self.tested_effect,
            "factors": list(self.factors),
            "levels": dict(self.levels),
            "at": dict(self.at),
            "components": [
                {
                    "sign": c.sign,
                    "assignment": dict(c.assignment),
                    "design_row": {k: float(v) for k, v in c.design_row.items()},
                }
                for c in self.components
            ],
        }


@dataclass(frozen=True)
class FactorialContrastSet:
    """An ordered collection of contrasts sharing one fixed-effect column order.

    Args:
        columns: Fixed-effect columns (rows of :meth:`matrix`).
        contrasts: The built contrasts, in declaration order.
    """

    columns: tuple[str, ...]
    contrasts: tuple[FactorialContrastResult, ...]

    @property
    def names(self) -> list[str]:
        """Contrast names, in order."""
        return [c.name for c in self.contrasts]

    def matrix(self) -> pd.DataFrame:
        """Return the ``fixed coefficients x contrasts`` weight matrix (the ``L`` for R)."""
        data = {c.name: list(c.weights) for c in self.contrasts}
        return pd.DataFrame(data, index=list(self.columns), columns=self.names)

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-safe dictionary."""
        return {
            "columns": list(self.columns),
            "contrasts": [c.to_dict() for c in self.contrasts],
        }


def _label(level_assignment: Mapping[str, str], factors: Sequence[str]) -> str:
    """Render the levels of ``factors`` as written, space-separated."""
    return " ".join(level_assignment[f] for f in factors)


def _at_suffix(at: Mapping[str, str]) -> str:
    """Render the ``at`` clause of a tested effect (empty when none)."""
    return "" if not at else " at " + ", ".join(f"{k}={v}" for k, v in at.items())


def _interacting_partners(design: FactorialDesignConfig, factor: str) -> set[str]:
    """Return the factors that interact with ``factor`` in the fixed-effect model."""
    partners: set[str] = set()
    for a, b in design.interactions:
        if a == factor:
            partners.add(b)
        elif b == factor:
            partners.add(a)
    return partners


def _check_at_and_ambiguity(
    contrast: FactorialContrast,
    design: FactorialDesignConfig,
    focal: Sequence[str],
) -> None:
    """Validate the ``at`` clause and refuse an effect that depends on an unstated level."""

    for factor in contrast.at:
        if factor not in design.factors:
            raise CellQuorumConfigError(
                f"Contrast '{contrast.name}': 'at' references unknown factor '{factor}'; "
                f"declared factors: {list(design.factors)}."
            )
    for factor in focal:
        if factor not in design.factors:
            raise CellQuorumConfigError(
                f"Contrast '{contrast.name}': unknown factor '{factor}'; declared factors: "
                f"{list(design.factors)}."
            )
    for factor in focal:
        missing = sorted(
            p
            for p in _interacting_partners(design, factor)
            if p not in focal and p not in contrast.at
        )
        if missing:
            raise CellQuorumConfigError(
                f"Contrast '{contrast.name}' is ambiguous: the effect of '{factor}' depends on "
                f"{missing} (they interact in the model). State the intended level(s) with "
                f"`at`, e.g. at: {{{missing[0]}: <level>}}."
            )


def _build_one(
    contrast: FactorialContrast, encoder: FactorialDesignEncoder, design: FactorialDesignConfig
) -> FactorialContrastResult:
    """Build one contrast's weights, components and tested-effect text."""

    if isinstance(contrast, SimpleEffectContrast):
        focal: tuple[str, ...] = (contrast.factor,)
        _check_at_and_ambiguity(contrast, design, focal)
        specs = [
            (+1, {contrast.factor: contrast.numerator, **contrast.at}),
            (-1, {contrast.factor: contrast.denominator, **contrast.at}),
        ]
        tested = f"{contrast.factor}: {contrast.numerator} vs {contrast.denominator}" + _at_suffix(
            contrast.at
        )
        levels = {"numerator": contrast.numerator, "denominator": contrast.denominator}
    elif isinstance(contrast, DifferenceOfDifferencesContrast):
        a, b = contrast.factor_a, contrast.factor_b
        focal = (a, b)
        _check_at_and_ambiguity(contrast, design, focal)
        na, da, nb, db = (
            contrast.numerator_a,
            contrast.denominator_a,
            contrast.numerator_b,
            contrast.denominator_b,
        )
        specs = [
            (+1, {a: na, b: nb, **contrast.at}),
            (-1, {a: na, b: db, **contrast.at}),
            (-1, {a: da, b: nb, **contrast.at}),
            (+1, {a: da, b: db, **contrast.at}),
        ]
        tested = f"{a} x {b}: ({na} {nb} - {na} {db}) - ({da} {nb} - {da} {db})" + _at_suffix(
            contrast.at
        )
        levels = {
            "numerator_a": na,
            "denominator_a": da,
            "numerator_b": nb,
            "denominator_b": db,
        }
    else:  # pragma: no cover - the discriminated union is exhaustive
        raise CellQuorumConfigError(f"Unsupported contrast type {type(contrast).__name__}.")

    weights = np.zeros(len(encoder.columns), dtype=float)
    components: list[ContrastComponent] = []
    for sign, assignment in specs:
        row = encoder.encode(assignment)
        weights += sign * row
        components.append(
            ContrastComponent(
                sign=sign,
                assignment={k: str(v) for k, v in assignment.items()},
                design_row=dict(zip(encoder.columns, row.tolist(), strict=True)),
            )
        )

    if not np.any(np.abs(weights) > _ESTIMABLE_ATOL):
        raise CellQuorumConfigError(
            f"Contrast '{contrast.name}' is identically zero under the declared model: "
            "its factors do not interact (a difference of differences needs the "
            "interaction in `design.factorial.interactions`)."
        )

    return FactorialContrastResult(
        name=contrast.name,
        kind=contrast.kind,
        columns=encoder.columns,
        weights=tuple(float(w) for w in weights),
        tested_effect=tested,
        components=tuple(components),
        factors=tuple(focal),
        levels=levels,
        at=dict(contrast.at),
    )


def build_factorial_contrasts(
    contrasts: Sequence[FactorialContrast],
    *,
    design: FactorialDesignConfig,
    sample_meta: pd.DataFrame,
) -> FactorialContrastSet:
    """
    Build every requested factorial contrast against a specimen table.

    Contrast weights come from encoding hypothetical specimens with the design's own
    reference-aware coding (see the module docstring), so they align exactly with
    ``FactorialDesignReport.fixed_effect_columns`` and survive a change of reference
    levels. Each contrast is additionally checked for estimability in the observed design
    (its weight vector must lie in the row space of the fixed-effect matrix).

    Args:
        contrasts: Structured contrasts, in the order results should be produced.
        design: The factorial design declaration.
        sample_meta: One row per specimen holding every factor; defines the level
            vocabulary and the observed design.

    Returns:
        A :class:`FactorialContrastSet` with one weight vector per contrast.

    Raises:
        CellQuorumConfigError: Duplicate names, unknown factors/levels, an ambiguous
            simple effect (a factor interacting with the compared one is not fixed by
            ``at``), or a contrast identically zero under the model.
        ContrastNotEstimable: A valid contrast the observed design cannot estimate.
    """

    names = [c.name for c in contrasts]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise CellQuorumConfigError(f"Duplicate contrast name(s): {duplicates}.")

    encoder = FactorialDesignEncoder.fit(sample_meta, design)
    observed = encoder.matrix(sample_meta).to_numpy(dtype=float)
    projector = np.linalg.pinv(observed) @ observed

    built: list[FactorialContrastResult] = []
    for contrast in contrasts:
        result = _build_one(contrast, encoder, design)
        weights = np.asarray(result.weights)
        if not np.allclose(weights @ projector, weights, atol=_ESTIMABLE_ATOL):
            raise ContrastNotEstimable(
                f"Contrast '{contrast.name}' ({result.tested_effect}) is not estimable in the "
                "observed design (its weight vector is outside the row space of the fixed-effect "
                "matrix; an empty or aliased factorial cell)."
            )
        built.append(result)
    return FactorialContrastSet(columns=encoder.columns, contrasts=tuple(built))


__all__ = [
    "ContrastComponent",
    "ContrastNotEstimable",
    "FactorialContrastResult",
    "FactorialContrastSet",
    "FactorialDesignEncoder",
    "build_factorial_contrasts",
]
