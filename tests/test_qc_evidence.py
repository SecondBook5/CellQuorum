"""Tests for the graded QC evidence model and initial adjudication.

Ordered to match the module: evidence semantics first, then the adjudication built on it.

Every assertion corresponds to a way the previous QC system destroyed real biology, so
each should fail loudly if someone later "simplifies" the logic. The numbers in
:func:`_policy` are test fixtures, **not** calibrated defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.qc.evidence import (
    AdjudicationPolicy,
    AdjudicationReason,
    AxisEvidence,
    Direction,
    EvidenceAvailability,
    EvidenceFamily,
    EvidenceTable,
    QCAdjudicationError,
    QCEvidenceError,
    QCStateInitial,
    adjudicate_initial,
    build_axis,
)

CELLS = pd.Index([f"cell_{i}" for i in range(5)])


def _axis(
    name: str,
    family: EvidenceFamily,
    values: list[float] | float,
    availability: EvidenceAvailability = EvidenceAvailability.AVAILABLE_VALID,
    weight: float = 1.0,
) -> AxisEvidence:
    """Build one axis over the shared cell index; a scalar broadcasts to all cells."""
    if not isinstance(values, list):
        values = [values] * len(CELLS)
    return build_axis(
        name=name,
        family=family,
        direction=Direction.UPPER_TAIL,
        severity=pd.Series(values, index=CELLS, dtype=float),
        availability=availability,
        weight=weight,
    )


def _table(*axes: AxisEvidence) -> EvidenceTable:
    """Assemble axes into a table over the shared cell index."""
    return EvidenceTable(axes=axes, obs_names=CELLS)


def _policy(**overrides: float) -> AdjudicationPolicy:
    """A policy for testing only. These numbers are NOT calibrated defaults."""
    bars: dict[str, float] = {
        "concern_severity": 0.5,
        "severe_severity": 0.8,
        "min_concordant_families": 2,
        "uninformative_capture_severity": 0.98,
        "min_coverage_for_quarantine": 0.5,
        "multiplet_severity": 0.9,
    }
    bars.update(overrides)
    return AdjudicationPolicy(**bars)  # type: ignore[arg-type]


# ═══ 1. Availability semantics ══════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("state", "usable"),
    [
        (EvidenceAvailability.AVAILABLE_VALID, True),
        (EvidenceAvailability.MODEL_UNSTABLE, True),
        (EvidenceAvailability.UNAVAILABLE_INPUT, False),
        (EvidenceAvailability.NOT_APPLICABLE, False),
        (EvidenceAvailability.COMPUTATION_FAILED, False),
    ],
)
def test_only_measured_states_are_usable(state, usable) -> None:
    """Measured states contribute; the three 'did not measure' states never do."""
    assert state.is_usable is usable


def test_unavailable_axis_severity_is_blanked_not_zero() -> None:
    """An unmeasurable axis must not present a low severity.

    The central invariant: a producer passing zeros for an axis it could not compute
    would otherwise make that axis look reassuring.
    """
    axis = _axis(
        "intronic_fraction",
        EvidenceFamily.NUCLEAR_INTEGRITY,
        0.0,
        availability=EvidenceAvailability.UNAVAILABLE_INPUT,
    )

    assert axis.severity.isna().all()
    assert not axis.usable_mask().any()
    assert axis.effective_severity().isna().all()


def test_not_applicable_and_computation_failed_stay_distinct() -> None:
    """The two most-often-conflated states must remain separable in the obs frame.

    'Expected for this assay' and 'should have worked and did not' demand different
    responses; collapsing them hides a real evidence gap.
    """
    frame = _table(
        _axis(
            "intronic_fraction",
            EvidenceFamily.NUCLEAR_INTEGRITY,
            0.5,
            availability=EvidenceAvailability.NOT_APPLICABLE,
        ),
        _axis(
            "malat1_fraction",
            EvidenceFamily.NUCLEAR_INTEGRITY,
            0.5,
            availability=EvidenceAvailability.COMPUTATION_FAILED,
        ),
    ).to_obs_frame()

    assert set(frame["qc_ev_intronic_fraction_availability"]) == {"not_applicable"}
    assert set(frame["qc_ev_malat1_fraction_availability"]) == {"computation_failed"}


# ═══ 2. Construction guards ═════════════════════════════════════════════════════════


def test_usable_cell_with_nan_severity_is_rejected() -> None:
    """A cell marked usable but carrying NaN is the contradiction the model forbids."""
    with pytest.raises(QCEvidenceError, match="marked usable but have NaN"):
        AxisEvidence(
            name="pct_counts_mito",
            family=EvidenceFamily.METABOLIC_STRESS,
            direction=Direction.UPPER_TAIL,
            severity=pd.Series([0.1, np.nan, 0.3, 0.4, 0.5], index=CELLS),
            availability=pd.Series(["available_valid"] * 5, index=CELLS),
        )


def test_out_of_range_severity_is_rejected() -> None:
    """Severity is a normalised scale; unnormalised input fails rather than clipping."""
    with pytest.raises(QCEvidenceError, match=r"severity must lie in \[0, 1\]"):
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, [0.1, 0.2, 7.5, 0.4, 0.5])


def test_misaligned_axis_is_rejected() -> None:
    """Axes describing different cells cannot be aggregated safely."""
    other = pd.Index([f"other_{i}" for i in range(5)])
    axis = build_axis(
        name="x",
        family=EvidenceFamily.CAPTURE_COMPLEXITY,
        direction=Direction.LOWER_TAIL,
        severity=pd.Series([0.1] * 5, index=other),
        availability=EvidenceAvailability.AVAILABLE_VALID,
    )
    with pytest.raises(QCEvidenceError, match="not aligned"):
        EvidenceTable(axes=(axis,), obs_names=CELLS)


def test_empty_table_is_rejected() -> None:
    """Adjudicating on no evidence must not be possible."""
    with pytest.raises(QCEvidenceError, match="at least one axis"):
        EvidenceTable(axes=(), obs_names=CELLS)


def test_non_positive_weight_is_rejected() -> None:
    """A zero weight would silently erase an axis from every aggregate."""
    with pytest.raises(QCEvidenceError, match="weight must be positive"):
        _axis("x", EvidenceFamily.METABOLIC_STRESS, 0.5, weight=0.0)


# ═══ 3. Family aggregation: the anti-double-counting invariant ══════════════════════


def test_correlated_axes_in_one_family_count_as_a_single_hit() -> None:
    """Low UMI and low genes are one mechanism, so they must corroborate once.

    This is the failure that makes a small quiescent cell look damaged twice for being
    small once.
    """
    table = _table(
        _axis("total_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis("log1p_total_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
    )
    assert (table.concordant_family_count(min_severity=0.5) == 1).all()


def test_distinct_families_do_corroborate() -> None:
    """Severity in genuinely independent families counts separately."""
    table = _table(
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.9),
        _axis("malat1_fraction", EvidenceFamily.NUCLEAR_INTEGRITY, 0.9),
    )
    assert (table.concordant_family_count(min_severity=0.5) == 3).all()


def test_family_severity_is_max_not_mean() -> None:
    """One healthy correlated metric must not dilute a genuine signal from another."""
    table = _table(
        _axis("malat1_fraction", EvidenceFamily.NUCLEAR_INTEGRITY, 0.0),
        _axis("intronic_fraction", EvidenceFamily.NUCLEAR_INTEGRITY, 0.8),
    )
    # A mean would report 0.4 and read as mild.
    assert np.allclose(table.family_severity()["nuclear_integrity"].to_numpy(), 0.8)


def test_family_severity_is_nan_only_when_no_axis_was_usable() -> None:
    """An unmeasurable family reports NaN, not a comfortable zero."""
    table = _table(
        _axis(
            "intronic_fraction",
            EvidenceFamily.NUCLEAR_INTEGRITY,
            0.5,
            availability=EvidenceAvailability.UNAVAILABLE_INPUT,
        ),
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.7),
    )
    severity = table.family_severity()

    assert severity["nuclear_integrity"].isna().all()
    assert np.allclose(severity["metabolic_stress"].to_numpy(), 0.7)
    # An unmeasured family never counts toward concordance either.
    assert (table.concordant_family_count(min_severity=0.1) == 1).all()


def test_weight_reduces_influence_without_discarding_the_axis() -> None:
    """A shaky fit should count for less, not for nothing."""
    table = _table(_axis("mito_mixture", EvidenceFamily.METABOLIC_STRESS, 0.8, weight=0.5))
    assert np.allclose(table.family_severity()["metabolic_stress"].to_numpy(), 0.4)


def test_doublet_evidence_is_excluded_from_damage_severity() -> None:
    """A doublet is not a damaged cell; pooling them would misclassify good libraries."""
    table = _table(
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.1),
        _axis("scdblfinder_score", EvidenceFamily.MULTIPLET, 0.95),
        _axis("cell_probability", EvidenceFamily.CELL_CALLING, 0.9),
    )
    assert list(table.damage_family_severity().columns) == ["metabolic_stress"]


# ═══ 4. Coverage ════════════════════════════════════════════════════════════════════


def test_coverage_reflects_how_much_was_measurable() -> None:
    """A cell judged on half its families must report half coverage."""
    table = _table(
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.2),
        _axis(
            "intronic_fraction",
            EvidenceFamily.NUCLEAR_INTEGRITY,
            0.2,
            availability=EvidenceAvailability.NOT_APPLICABLE,
        ),
    )
    assert np.allclose(table.evidence_coverage().to_numpy(), 0.5)


def test_coverage_is_per_cell_not_per_dataset() -> None:
    """Availability that varies by cell must produce coverage that varies by cell."""
    varying = build_axis(
        name="mito_mixture_posterior",
        family=EvidenceFamily.METABOLIC_STRESS,
        direction=Direction.UPPER_TAIL,
        severity=pd.Series([0.3] * 5, index=CELLS),
        availability=pd.Series(
            [
                "available_valid",
                "available_valid",
                "model_unstable",
                "computation_failed",
                "unavailable_input",
            ],
            index=CELLS,
        ),
    )
    table = _table(varying, _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.1))

    # Cells 0-2 have both families; cells 3-4 have only capture.
    assert np.allclose(table.evidence_coverage().to_numpy(), [1.0, 1.0, 1.0, 0.5, 0.5])


# ═══ 5. Ordering and output shape ═══════════════════════════════════════════════════


def test_families_present_follows_canonical_order() -> None:
    """Stable column order keeps tables, figures, and CSVs comparable across runs."""
    table = _table(
        _axis("scdblfinder_score", EvidenceFamily.MULTIPLET, 0.1),
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.1),
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.1),
    )
    assert table.families_present() == (
        EvidenceFamily.CAPTURE_COMPLEXITY,
        EvidenceFamily.METABOLIC_STRESS,
        EvidenceFamily.MULTIPLET,
    )


def test_obs_frame_pairs_every_severity_with_its_availability() -> None:
    """Severity must never be readable without the availability that qualifies it."""
    table = _table(
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.3),
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.4),
    )
    frame = table.to_obs_frame()

    for axis in table.axes:
        assert f"qc_ev_{axis.name}_severity" in frame.columns
        assert f"qc_ev_{axis.name}_availability" in frame.columns
    assert "qc_evidence_coverage" in frame.columns
    assert frame.index.equals(CELLS)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_concordance_rejects_a_bar_outside_the_severity_scale(bad) -> None:
    """A bar outside [0, 1] can only be a caller error and would silently degenerate."""
    table = _table(_axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.5))
    with pytest.raises(QCEvidenceError, match=r"min_severity must lie in \[0, 1\]"):
        table.concordant_family_count(min_severity=bad)


def test_no_default_severity_bar_exists() -> None:
    """Choosing a cut-off is calibration; a default here would become silent policy."""
    table = _table(_axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.5))
    with pytest.raises(TypeError):
        table.concordant_family_count()  # type: ignore[call-arg]


# ═══ 6. Adjudication: one model may not condemn a cell ═════════════════════════════


def test_extreme_single_family_severity_is_borderline_not_quarantine() -> None:
    """A miQC posterior of 0.96, alone, must not quarantine.

    A posterior describes a fitted distribution, not a membrane. This is exactly what the
    previous mitochondrial ceiling got wrong: it accounted for essentially all removals,
    and the cells it removed had normal complexity everywhere else.
    """
    table = _table(
        _axis("mito_mixture_posterior", EvidenceFamily.METABOLIC_STRESS, 0.96),
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.05),
        _axis("malat1_fraction", EvidenceFamily.NUCLEAR_INTEGRITY, 0.05),
    )
    result = adjudicate_initial(table, _policy())

    assert (result.state == str(QCStateInitial.BORDERLINE)).all()
    assert (result.reason == str(AdjudicationReason.SUPPORTING_EVIDENCE_ONLY)).all()


def test_severe_stress_plus_severe_mito_alone_cannot_quarantine() -> None:
    """Two supporting axes are still only supporting evidence.

    In inflamed tissue both are genuinely biology, so their agreement is not independent
    corroboration of damage.
    """
    table = _table(
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.95),
        _axis("dissociation_stress", EvidenceFamily.METABOLIC_STRESS, 0.95),
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.1),
    )
    assert (adjudicate_initial(table, _policy()).state == str(QCStateInitial.BORDERLINE)).all()


def test_correlated_axes_cannot_fake_concordance() -> None:
    """Three severe capture metrics are one family hit, so they must not quarantine."""
    table = _table(
        _axis("total_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis("log1p_total_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis("malat1_fraction", EvidenceFamily.NUCLEAR_INTEGRITY, 0.05),
    )
    result = adjudicate_initial(table, _policy())

    assert (result.severe_families == 1).all()
    assert (result.state == str(QCStateInitial.BORDERLINE)).all()


# ═══ 7. Adjudication: the two legitimate routes to quarantine ══════════════════════


def test_uninformative_barcode_quarantines_on_capture_alone() -> None:
    """Near-total capture failure is the one single-family route, and it is justified.

    The claim is not "this cell is damaged" but "this barcode carries no usable
    information about any cell".
    """
    table = _table(
        _axis("capture_collapse", EvidenceFamily.CAPTURE_COMPLEXITY, 0.99),
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.1),
    )
    result = adjudicate_initial(table, _policy())

    assert (result.state == str(QCStateInitial.QUARANTINE)).all()
    assert (result.reason == str(AdjudicationReason.UNINFORMATIVE_BARCODE)).all()


def test_concordant_severe_damage_across_independent_families_quarantines() -> None:
    """Independent families agreeing severely is real corroboration."""
    table = _table(
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis("intronic_fraction", EvidenceFamily.NUCLEAR_INTEGRITY, 0.9),
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.9),
    )
    result = adjudicate_initial(table, _policy())

    assert (result.state == str(QCStateInitial.QUARANTINE)).all()
    assert (result.reason == str(AdjudicationReason.CONCORDANT_SEVERE_DAMAGE)).all()


# ═══ 8. Adjudication: coverage makes it more conservative ══════════════════════════


def test_quarantine_is_withheld_when_coverage_is_too_low() -> None:
    """Condemning a cell on evidence we mostly could not collect is the worst error."""
    table = _table(
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis("intronic_fraction", EvidenceFamily.NUCLEAR_INTEGRITY, 0.9),
        # Two of four families unmeasurable drops coverage to 0.5...
        _axis(
            "ambient_burden",
            EvidenceFamily.AMBIENT_BACKGROUND,
            0.5,
            availability=EvidenceAvailability.UNAVAILABLE_INPUT,
        ),
        _axis(
            "pct_counts_mito",
            EvidenceFamily.METABOLIC_STRESS,
            0.5,
            availability=EvidenceAvailability.COMPUTATION_FAILED,
        ),
    )
    # ...so a floor above that must withhold quarantine.
    result = adjudicate_initial(table, _policy(min_coverage_for_quarantine=0.75))

    assert (result.state == str(QCStateInitial.BORDERLINE)).all()
    assert (result.reason == str(AdjudicationReason.WITHHELD_LOW_COVERAGE)).all()


def test_same_cells_quarantine_once_coverage_is_sufficient() -> None:
    """The coverage gate must be the only thing that changed the verdict."""
    table = _table(
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis("intronic_fraction", EvidenceFamily.NUCLEAR_INTEGRITY, 0.9),
    )
    result = adjudicate_initial(table, _policy(min_coverage_for_quarantine=0.75))

    assert (result.state == str(QCStateInitial.QUARANTINE)).all()


# ═══ 9. Adjudication: multiplet stays out of the damage verdict ════════════════════


def test_high_multiplet_severity_does_not_quarantine_a_healthy_library() -> None:
    """A doublet can be an excellent library that is simply not one cell."""
    table = _table(
        _axis("scdblfinder_score", EvidenceFamily.MULTIPLET, 0.99),
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.05),
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.05),
    )
    result = adjudicate_initial(table, _policy())

    assert (result.state == str(QCStateInitial.CORE)).all()
    assert result.probable_multiplet.all()
    assert (result.reason == str(AdjudicationReason.PROBABLE_MULTIPLET)).all()


def test_cell_calling_evidence_does_not_contribute_to_damage_concordance() -> None:
    """'May be an empty droplet' is a different claim from 'this cell is dying'."""
    table = _table(
        _axis("cell_probability", EvidenceFamily.CELL_CALLING, 0.95),
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.95),
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.05),
    )
    result = adjudicate_initial(table, _policy())

    assert (result.severe_families == 1).all()
    assert (result.state == str(QCStateInitial.BORDERLINE)).all()


# ═══ 10. Adjudication: core, drivers, confidence, reporting ════════════════════════


def test_clean_cells_are_core() -> None:
    """Nothing concerning means eligible to define the biological reference."""
    table = _table(
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.05),
        _axis("pct_counts_mito", EvidenceFamily.METABOLIC_STRESS, 0.1),
    )
    result = adjudicate_initial(table, _policy())

    assert (result.state == str(QCStateInitial.CORE)).all()
    assert (result.reason == str(AdjudicationReason.NO_CONCERN)).all()
    assert (result.primary_driver == "").all()


def test_primary_driver_is_emitted_not_inferred() -> None:
    """Overlapping evidence means only the adjudicator may name a primary cause."""
    table = _table(
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.6),
        _axis("intronic_fraction", EvidenceFamily.NUCLEAR_INTEGRITY, 0.9),
    )
    result = adjudicate_initial(table, _policy())

    assert (result.primary_driver == str(EvidenceFamily.NUCLEAR_INTEGRITY)).all()


def test_confidence_is_lower_when_coverage_is_lower() -> None:
    """A cell judged on less evidence must not look as certain as one judged on more."""
    full = _table(
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis("intronic_fraction", EvidenceFamily.NUCLEAR_INTEGRITY, 0.9),
    )
    partial = _table(
        _axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9),
        _axis(
            "intronic_fraction",
            EvidenceFamily.NUCLEAR_INTEGRITY,
            0.9,
            availability=EvidenceAvailability.UNAVAILABLE_INPUT,
        ),
    )
    policy = _policy()
    lower = adjudicate_initial(partial, policy).confidence
    higher = adjudicate_initial(full, policy).confidence

    assert (lower < higher).all()


def test_counts_include_states_with_zero_cells() -> None:
    """Summary tables need stable columns across runs."""
    table = _table(_axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.05))
    counts = adjudicate_initial(table, _policy()).counts()

    assert set(counts) == {str(state) for state in QCStateInitial}
    assert counts["core"] == len(CELLS)
    assert counts["quarantine"] == 0


def test_obs_frame_carries_state_reason_and_coverage() -> None:
    """Everything a figure or audit needs must land in obs, aligned."""
    table = _table(_axis("n_genes_by_counts", EvidenceFamily.CAPTURE_COMPLEXITY, 0.9))
    frame = adjudicate_initial(table, _policy()).to_obs_frame()

    for column in (
        "qc_state_initial",
        "qc_state_reason",
        "qc_concerning_families",
        "qc_severe_families",
        "qc_primary_driver",
        "qc_probable_multiplet",
        "qc_evidence_coverage",
        "qc_confidence",
    ):
        assert column in frame.columns
    assert frame.index.equals(CELLS)


def test_mixed_population_gets_distinct_states() -> None:
    """A realistic mix must separate rather than collapse to one verdict."""
    table = _table(
        # clean | stress only | concordant severe | uninformative | clean
        _axis(
            "n_genes_by_counts",
            EvidenceFamily.CAPTURE_COMPLEXITY,
            [0.05, 0.05, 0.90, 0.99, 0.05],
        ),
        _axis(
            "intronic_fraction",
            EvidenceFamily.NUCLEAR_INTEGRITY,
            [0.05, 0.05, 0.90, 0.10, 0.05],
        ),
        _axis(
            "pct_counts_mito",
            EvidenceFamily.METABOLIC_STRESS,
            [0.10, 0.95, 0.90, 0.10, 0.10],
        ),
    )
    result = adjudicate_initial(table, _policy())

    assert list(result.state) == [
        str(QCStateInitial.CORE),
        str(QCStateInitial.BORDERLINE),
        str(QCStateInitial.QUARANTINE),
        str(QCStateInitial.QUARANTINE),
        str(QCStateInitial.CORE),
    ]
    assert result.reason.iloc[1] == str(AdjudicationReason.SUPPORTING_EVIDENCE_ONLY)
    assert result.reason.iloc[3] == str(AdjudicationReason.UNINFORMATIVE_BARCODE)
    assert np.isfinite(result.confidence.to_numpy()).all()


# ═══ 11. Policy validation ══════════════════════════════════════════════════════════


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_policy_rejects_bars_outside_the_severity_scale(bad) -> None:
    """Bars are compared against normalised severity, so out-of-range is a caller error."""
    with pytest.raises(QCAdjudicationError, match=r"must lie in \[0, 1\]"):
        _policy(concern_severity=bad)


def test_policy_rejects_severe_bar_below_concern_bar() -> None:
    """'Severe' must not be weaker than 'concerning'."""
    with pytest.raises(QCAdjudicationError, match="must be >= concern_severity"):
        _policy(concern_severity=0.8, severe_severity=0.5)


def test_policy_rejects_single_family_quarantine() -> None:
    """Allowing one family defeats the design, so the policy refuses to express it."""
    with pytest.raises(QCAdjudicationError, match="min_concordant_families must be >= 2"):
        _policy(min_concordant_families=1)


def test_policy_has_no_defaults() -> None:
    """Calibration must be explicit; a default bar would become invisible policy."""
    with pytest.raises(TypeError):
        AdjudicationPolicy()  # type: ignore[call-arg]
