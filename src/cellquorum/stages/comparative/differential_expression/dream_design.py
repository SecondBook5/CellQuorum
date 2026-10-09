"""Pure helpers for the ``pseudobulk_dream`` method: formula, machine coding, gates.

Nothing here touches R. The model is expressed twice and both forms are recorded:

* the **human formula**, in the user's own factor and column names, e.g.
  ``~ arm * site + (1 | subject)``;
* the **machine formula**, over safe internal column names, e.g.
  ``~ cq_fx1 + cq_fx2 + cq_fx3 + (1 | cq_re1)``.

The machine formula's fixed-effect columns are CellQuorum's own treatment-coded design
columns (so R never re-levels a factor), which makes one coding the single source of truth
for the design matrix, the contrast weights and the fit. :class:`MachineDesign` carries the
mapping back to the human-readable column names.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pandas as pd

from cellquorum.config.design import FactorialDesignConfig
from cellquorum.config.factorial_contrasts import FactorialContrastSet, FactorialDesignEncoder
from cellquorum.config.factorial_design import FactorialDesignReport
from cellquorum.stages.comparative.differential_expression.config import PseudobulkSupportConfig

# The intercept's name in the R coefficient vector (and the L matrix row index).
R_INTERCEPT = "(Intercept)"


def slugify(name: str) -> str:
    """Return a filesystem-safe, deterministic slug (lowercase ``[a-z0-9._-]``)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_.-").lower()
    return slug or "unnamed"


def unique_slugs(names: list[str]) -> dict[str, str]:
    """Map each name to a slug, disambiguating collisions deterministically.

    Two names that slugify identically (``"B cell"`` / ``"B_cell"``) must never share a
    path. Every member of a colliding group gets a short stable hash of its original
    name appended, so the result is independent of input order.

    Args:
        names: Original names (cell types or contrasts).

    Returns:
        ``{name: slug}`` with all slugs distinct.
    """

    base = {n: slugify(n) for n in names}
    counts: dict[str, int] = {}
    for slug in base.values():
        counts[slug] = counts.get(slug, 0) + 1
    out: dict[str, str] = {}
    for name, slug in base.items():
        if counts[slug] > 1:
            digest = hashlib.sha1(str(name).encode()).hexdigest()[:8]
            slug = f"{slug}-{digest}"
        out[name] = slug
    return out


def human_formula(design: FactorialDesignConfig) -> str:
    """Render the requested model in the user's own names, e.g. ``~ a * b + (1 | s)``."""

    factors = list(design.factors)
    pairs = [tuple(i) for i in design.interactions]
    if len(factors) == 2 and len(pairs) == 1 and set(pairs[0]) == set(factors):
        fixed = f"{factors[0]} * {factors[1]}"
    else:
        fixed = " + ".join([*factors, *(f"{a}:{b}" for a, b in pairs)])
    random = "".join(f" + (1 | {col})" for col in design.random_effects)
    return f"~ {fixed}{random}"


@dataclass(frozen=True)
class MachineDesign:
    """Everything R needs, plus the mapping back to human names.

    Args:
        meta: Specimen-indexed frame of numeric ``cq_fx*`` columns and ``cq_re*``
            grouping columns.
        formula: Machine formula over the internal names.
        human_formula: The formula in the user's names.
        fixed_columns: ``{design column -> cq_fx name}`` (Intercept excluded).
        random_columns: ``{grouping column -> cq_re name}``.
        contrast_ids: ``{cq_c id -> contrast name}``.
        L: ``coefficients x contrasts`` matrix: rows ``(Intercept)`` then ``cq_fx*``,
            columns the ``cq_c`` ids.
    """

    meta: pd.DataFrame
    formula: str
    human_formula: str
    fixed_columns: dict[str, str]
    random_columns: dict[str, str]
    contrast_ids: dict[str, str]
    L: pd.DataFrame


def build_machine_design(
    *,
    encoder: FactorialDesignEncoder,
    specimens: pd.DataFrame,
    design: FactorialDesignConfig,
    contrasts: FactorialContrastSet,
    sample_col: str,
) -> MachineDesign:
    """
    Translate a validated factorial design into the numeric inputs of the R fit.

    Args:
        encoder: Fitted design encoder (defines the coding and column order).
        specimens: Included specimens, indexed by specimen id, holding every factor and
            random-effect grouping column.
        design: The factorial design declaration.
        contrasts: Built contrasts, aligned to ``encoder.columns``.
        sample_col: The specimen column (a random effect may refer to it; it is then the
            index).

    Returns:
        The :class:`MachineDesign`.
    """

    matrix = encoder.matrix(specimens)
    fixed_names = [c for c in encoder.columns if c != "Intercept"]
    fixed_map = {col: f"cq_fx{i}" for i, col in enumerate(fixed_names, start=1)}

    meta = matrix[fixed_names].rename(columns=fixed_map).astype(float)
    random_map: dict[str, str] = {}
    for j, column in enumerate(design.random_effects, start=1):
        internal = f"cq_re{j}"
        random_map[column] = internal
        groups = (
            pd.Series(specimens.index.astype(str), index=specimens.index)
            if column == sample_col
            else specimens[column].astype(str)
        )
        meta[internal] = groups.loc[meta.index].to_numpy()

    terms = [*fixed_map.values(), *(f"(1 | {name})" for name in random_map.values())]
    formula = "~ " + " + ".join(terms)

    ids = {f"cq_c{k}": name for k, name in enumerate(contrasts.names, start=1)}
    weights = contrasts.matrix()
    rows = {R_INTERCEPT if c == "Intercept" else fixed_map[c]: c for c in contrasts.columns}
    L = pd.DataFrame(
        {cid: weights[name].to_numpy() for cid, name in ids.items()},
        index=list(rows),
    )
    return MachineDesign(
        meta=meta,
        formula=formula,
        human_formula=human_formula(design),
        fixed_columns=fixed_map,
        random_columns=random_map,
        contrast_ids=ids,
        L=L,
    )


def support_gate_failures(
    report: FactorialDesignReport,
    design: FactorialDesignConfig,
    gates: PseudobulkSupportConfig,
) -> list[str]:
    """
    List the inferential support gates a cell type's included specimens fail.

    Replication thresholds are method requirements, not design validity, so they live
    here rather than in the design analyzer. An empty list means every gate passed.

    Args:
        report: Factorial design report computed on the INCLUDED specimens.
        design: The factorial design declaration.
        gates: Configured support thresholds.

    Returns:
        One human-readable reason per failed gate.
    """

    failures: list[str] = []

    # Specimens per factorial cell: the interaction grid, or each factor level when no
    # interaction is modelled.
    if report.interactions:
        for inter in report.interactions:
            thin = [
                f"{c['levels']}={c['n_samples']}"
                for c in inter.cells
                if int(c["n_samples"]) < gates.min_samples_per_factorial_cell  # type: ignore[call-overload]
            ]
            if thin:
                failures.append(
                    f"fewer than {gates.min_samples_per_factorial_cell} specimens in "
                    f"{inter.factors[0]} x {inter.factors[1]} cell(s): {', '.join(thin)}"
                )
    else:
        for factor in report.factors:
            thin = [
                f"{lv}={n}"
                for lv, n in factor.n_samples_by_level.items()
                if n < gates.min_samples_per_factorial_cell
            ]
            if thin:
                failures.append(
                    f"fewer than {gates.min_samples_per_factorial_cell} specimens in "
                    f"'{factor.name}' level(s): {', '.join(thin)}"
                )

    # DISTINCT subjects per factorial cell: specimens from one subject are not replicates.
    if report.subject_col is not None:
        if report.interactions:
            for inter in report.interactions:
                thin = [
                    f"{c['levels']}={c['n_subjects']}"
                    for c in inter.cells
                    if c["n_subjects"] is not None
                    and int(c["n_subjects"]) < gates.min_subjects_per_factorial_cell  # type: ignore[call-overload]
                ]
                if thin:
                    failures.append(
                        f"fewer than {gates.min_subjects_per_factorial_cell} distinct subjects in "
                        f"{inter.factors[0]} x {inter.factors[1]} cell(s): {', '.join(thin)}"
                    )
        else:
            for factor in report.factors:
                thin = [
                    f"{lv}={n}"
                    for lv, n in (factor.n_subjects_by_level or {}).items()
                    if n < gates.min_subjects_per_factorial_cell
                ]
                if thin:
                    failures.append(
                        f"fewer than {gates.min_subjects_per_factorial_cell} distinct subjects in "
                        f"'{factor.name}' level(s): {', '.join(thin)}"
                    )

    # Independent subjects per level of every between-subject factor.
    for factor in report.factors:
        if design.factors[factor.name].scope != "between_subject":
            continue
        if factor.n_subjects_by_level is None:
            continue
        thin = [
            f"{lv}={n}"
            for lv, n in factor.n_subjects_by_level.items()
            if n < gates.min_subjects_per_between_level
        ]
        if thin:
            failures.append(
                f"fewer than {gates.min_subjects_per_between_level} independent subjects in "
                f"between-subject factor '{factor.name}' level(s): {', '.join(thin)}"
            )
    return failures


__all__ = [
    "R_INTERCEPT",
    "MachineDesign",
    "build_machine_design",
    "human_formula",
    "slugify",
    "support_gate_failures",
    "unique_slugs",
]
