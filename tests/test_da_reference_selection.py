"""Tests for compositional reference selection.

The fixtures build cohorts whose correct answer is known by construction, so a
failure points at the selector rather than at a judgement call about real data.
The central case is the one that motivated the module: a rare population that
scCODA's criterion prefers and that must not be chosen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.comparative.differential_abundance.reference_selection import (
    REFERENCE_CRITERION_COLUMNS,
    select_compositional_reference,
    split_reference_fits,
)


def build_counts(
    spec: dict[str, tuple[float, float]],
    *,
    n_samples: int = 12,
    total: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a sample × cell-type count matrix with controlled abundance and stability.

    Args:
        spec: Maps cell type to (mean relative abundance, log-normal noise sigma).
            Sigma is approximately the coefficient of variation of that cell
            type's share, so abundance and stability are set independently --
            which is exactly the separation scCODA's var/mean criterion conflates.
        n_samples: Number of samples to generate.
        total: Approximate cells per sample.
        seed: Seed for the generator.

    Returns:
        Integer count DataFrame, samples × cell types.
    """

    rng = np.random.default_rng(seed)
    columns = {}
    for name, (abundance, sigma) in spec.items():
        # Multiplicative noise keeps the CV at roughly sigma regardless of how
        # abundant the population is, which is what lets a test hold stability
        # fixed while varying abundance.
        #
        # The -sigma**2/2 offset is load-bearing, not decoration. A plain
        # log-normal has mean exp(sigma**2/2), so without it the noisiest cell
        # types come out the MOST abundant -- which couples the two properties
        # this fixture exists to separate, and silently turned an "everything is
        # rare" cohort into one where only the noisy types cleared the abundance
        # floor.
        noise = rng.normal(-(sigma**2) / 2, sigma, size=n_samples)
        columns[name] = np.maximum(1, np.round(total * abundance * np.exp(noise))).astype(int)
    index = [f"S{i:02d}" for i in range(n_samples)]
    return pd.DataFrame(columns, index=index)


def test_a_rare_population_is_not_chosen_however_stable_it_looks():
    """The rare cell type scCODA's criterion prefers must be rejected outright."""

    counts = build_counts(
        {
            "Abundant_stable": (0.40, 0.12),
            "Abundant_noisy": (0.30, 0.60),
            "Midsize": (0.29, 0.35),
            "Rare_noisy": (0.003, 0.70),
        }
    )

    choice = select_compositional_reference(counts)
    table = choice.criterion.set_index("cell_type")

    # scCODA's criterion genuinely prefers the rare type -- if it did not, this
    # fixture would not be testing anything.
    assert table["sccoda_dispersion"].idxmin() == "Rare_noisy"

    assert choice.cell_type == "Abundant_stable"
    assert not table.loc["Rare_noisy", "eligible"]
    assert "Rare_noisy" in choice.reason, "the rejected scCODA pick should be named"


def test_the_steadiest_abundant_type_wins():
    """Among comparably abundant types, the lowest-variance share is selected."""

    counts = build_counts(
        {
            "Steady": (0.33, 0.10),
            "Wobbly": (0.33, 0.55),
            "Wildest": (0.34, 0.90),
        }
    )

    choice = select_compositional_reference(counts)

    assert choice.cell_type == "Steady"
    assert not choice.relaxed
    table = choice.criterion.set_index("cell_type")
    assert table.loc["Steady", "clr_variance"] < table.loc["Wobbly", "clr_variance"]
    assert table.loc["Wobbly", "clr_variance"] < table.loc["Wildest", "clr_variance"]


def test_the_criterion_is_scale_free_where_sccodas_is_not():
    """Shrinking a cell type's abundance must not make it look more stable.

    This pins the mathematical claim the module rests on. ``var(p)/mean(p)``
    equals ``cv**2 * mean``, so scaling a population down by 100x divides that
    criterion by about 100 while the population is exactly as steady as before.
    The centred-log-ratio variance is the quantity that does not move.
    """

    common = {"Filler_a": (0.45, 0.30), "Filler_b": (0.45, 0.30)}
    big = build_counts({**common, "Probe": (0.10, 0.40)}, seed=7)
    small = build_counts({**common, "Probe": (0.001, 0.40)}, seed=7)

    def probe_row(counts: pd.DataFrame) -> pd.Series:
        return select_compositional_reference(counts).criterion.set_index("cell_type").loc["Probe"]

    probe_big = probe_row(big)
    probe_small = probe_row(small)

    # scCODA's criterion collapses with abundance, which is the defect.
    ratio = probe_big["sccoda_dispersion"] / probe_small["sccoda_dispersion"]
    assert ratio > 20, f"expected sccoda_dispersion to track abundance, ratio was {ratio:.1f}"

    # The CLR variance reflects stability, which did not change. It is not exactly
    # equal -- the pseudocount matters more at low counts, and shrinking one
    # population slightly re-scales the others -- so this is a tolerance, not an
    # identity.
    assert probe_big["clr_variance"] == pytest.approx(probe_small["clr_variance"], rel=0.35)


def test_a_sometimes_absent_type_is_ineligible():
    """A cell type missing from some samples cannot serve as a denominator."""

    counts = build_counts({"Steady": (0.30, 0.15), "Patchy": (0.35, 0.20), "Other": (0.35, 0.25)})
    # Zero out the would-be reference in a quarter of samples.
    counts.loc[counts.index[:3], "Patchy"] = 0

    choice = select_compositional_reference(counts)
    table = choice.criterion.set_index("cell_type")

    assert not table.loc["Patchy", "eligible"]
    assert choice.cell_type != "Patchy"


def test_an_all_rare_cohort_relaxes_the_floor_and_says_so():
    """With no abundant population the floor is dropped, not silently ignored."""

    # Relative abundances necessarily sum to one, so "every population is rare"
    # only exists above 1/floor cell types. With 25 types the mean share is 4%,
    # under the 5% floor, and no reference can satisfy it.
    counts = build_counts(
        {f"Type{i:02d}": (0.04, 0.20 + 0.02 * i) for i in range(25)},
        total=200000,
    )

    choice = select_compositional_reference(counts)

    assert choice.cell_type == "Type00", "the steadiest type should still win"
    assert choice.relaxed
    assert "floor" in choice.reason
    assert "counting noise" in choice.reason, "the cost of relaxing should be stated"


def test_no_reference_when_nothing_is_present_often_enough():
    """When every cell type is patchy the decision is handed back, with a reason."""

    counts = build_counts({"A": (0.5, 0.2), "B": (0.5, 0.2)}, n_samples=8)
    counts.loc[counts.index[:4], "A"] = 0
    counts.loc[counts.index[4:], "B"] = 0

    choice = select_compositional_reference(counts)

    assert choice.cell_type is None
    assert "fitting backend" in choice.reason
    assert set(REFERENCE_CRITERION_COLUMNS) <= set(choice.criterion.columns)


def test_degenerate_inputs_return_a_reason_not_an_exception():
    """Empty and all-zero matrices are declined cleanly."""

    empty = select_compositional_reference(pd.DataFrame())
    assert empty.cell_type is None
    assert empty.reason
    assert list(empty.criterion.columns) == list(REFERENCE_CRITERION_COLUMNS)

    zeros = select_compositional_reference(pd.DataFrame({"A": [0, 0], "B": [0, 0]}))
    assert zeros.cell_type is None
    assert "zero cell total" in zeros.reason


def test_the_criterion_table_is_complete_and_marks_one_winner():
    """Every cell type appears once, and exactly one row is the selection."""

    counts = build_counts({"A": (0.4, 0.1), "B": (0.3, 0.3), "C": (0.3, 0.5)})

    choice = select_compositional_reference(counts)

    assert set(REFERENCE_CRITERION_COLUMNS) <= set(choice.criterion.columns)
    assert sorted(choice.criterion["cell_type"]) == ["A", "B", "C"]
    assert int(choice.criterion["selected"].sum()) == 1
    assert choice.criterion.loc[choice.criterion["selected"], "cell_type"].iloc[0] == (
        choice.cell_type
    )
    # Ordered by the criterion, so the table reads as the ranking it is.
    variances = choice.criterion["clr_variance"].to_numpy()
    assert np.all(np.diff(variances) >= 0)


def _stacked_fits() -> pd.DataFrame:
    """Two scCODA fits stacked as the helper returns them, told apart by reference."""

    rows = []
    for reference, credible in (("auto", [True, False, False]), ("B", [True, True, False])):
        for cell_type, is_credible in zip(["A", "B", "C"], credible, strict=True):
            rows.append(
                {
                    "cell_type": cell_type,
                    "log2_fold_change": 1.0,
                    "credible_effect": is_credible,
                    "reference": reference,
                }
            )
    return pd.DataFrame(rows)


def test_the_reported_fit_is_separated_from_the_sensitivity_fit():
    """A stacked table holds every cell type twice; counting it whole double-counts.

    This is not hypothetical: reading the stacked frame reported 6 credible effects on
    a cohort with 3, on every scCODA run the engine had produced.
    """

    primary, sensitivity = split_reference_fits(_stacked_fits(), "B")

    assert list(primary["reference"].unique()) == ["B"]
    assert list(sensitivity["reference"].unique()) == ["auto"]
    assert int(primary["credible_effect"].sum()) == 2
    assert int(sensitivity["credible_effect"].sum()) == 1
    # Every row is accounted for exactly once.
    assert len(primary) + len(sensitivity) == 6


def test_a_single_fit_is_the_reported_one():
    """With reference selection disabled only one fit runs, and it is the result."""

    single = _stacked_fits().query("reference == 'auto'").reset_index(drop=True)

    primary, sensitivity = split_reference_fits(single, None)

    assert len(primary) == 3
    assert sensitivity.empty
    # The schema is the input's; no marker column is invented for a table of one fit.
    assert list(primary.columns) == list(single.columns)


def test_an_absent_reference_falls_back_rather_than_returning_nothing():
    """A caller asking for a block that is not there must still get a usable frame."""

    primary, sensitivity = split_reference_fits(_stacked_fits(), "does_not_exist")

    assert list(primary["reference"].unique()) == ["auto"]
    assert list(sensitivity["reference"].unique()) == ["B"]


def test_a_table_without_a_reference_column_is_one_fit():
    """Not every producer of this table labels its fits."""

    unlabelled = pd.DataFrame({"cell_type": ["A"], "credible_effect": [True]})

    primary, sensitivity = split_reference_fits(unlabelled, "B")

    assert len(primary) == 1
    assert sensitivity.empty
