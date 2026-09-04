"""Tests for the depth-confounding audit.

The fixtures are built so the ground truth is known by construction rather than
asserted from a real dataset: one metric is pure biology (independent of depth),
one is pure depth (a deterministic function of it plus noise), and depth itself is
either confounded with condition or balanced across it. An audit that cannot tell
those apart is worse than no audit, because it would license exactly the claim it
is supposed to catch.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from cellquorum.stats import depth_confound_audit, depth_stratified_abundance

N_DONORS = 9
CELLS_PER_SAMPLE = 60
CASE = "Lymphedema"
CONTROL = "Normal"


def build_cohort(
    *,
    depth_confounded: bool,
    seed: int = 1337,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a paired cohort with a known depth structure.

    Args:
        depth_confounded: When True, case samples get systematically deeper cells,
            reproducing the situation that makes depth adjustment necessary. When
            False the arms are depth-balanced.
        seed: Seed for the deterministic fixture.

    Returns:
        ``(metrics, metadata)``, aligned on a shared index. ``metrics`` carries
        ``biology`` (a real condition effect, independent of depth) and
        ``depth_readout`` (a function of depth alone, with no condition term).
    """
    rng = np.random.default_rng(seed)
    records = []
    for donor in range(N_DONORS):
        for condition in (CONTROL, CASE):
            base_depth = 2200.0 + 120.0 * donor
            if depth_confounded and condition == CASE:
                base_depth += 700.0
            depth = rng.normal(base_depth, 300.0, CELLS_PER_SAMPLE).clip(200.0)
            # A real biological effect, deliberately uncorrelated with depth.
            biology = rng.normal(1.0 if condition == CASE else 0.0, 0.5, CELLS_PER_SAMPLE)
            # A metric that is nothing but depth. It has NO condition term: any
            # condition effect it shows is inherited from depth.
            depth_readout = -0.0004 * depth + rng.normal(0.0, 0.05, CELLS_PER_SAMPLE)
            for index in range(CELLS_PER_SAMPLE):
                records.append(
                    {
                        "cell": f"d{donor}_{condition}_{index}",
                        "donor_id": f"D{donor}",
                        "condition": condition,
                        "n_genes_by_counts": depth[index],
                        "biology": biology[index],
                        "depth_readout": depth_readout[index],
                    }
                )
    frame = pd.DataFrame(records).set_index("cell")
    metrics = frame[["biology", "depth_readout"]]
    metadata = frame[["donor_id", "condition", "n_genes_by_counts"]]
    return metrics, metadata


def audit(metrics: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Run the audit with the fixture's column names, indexed by metric."""
    return depth_confound_audit(
        metrics,
        metadata,
        donor_col="donor_id",
        condition_col="condition",
        case=CASE,
        control=CONTROL,
        depth_col="n_genes_by_counts",
    ).set_index("metric")


def test_a_depth_driven_metric_is_caught_when_the_arms_differ_in_depth() -> None:
    """
    Verify the motivating failure is flagged.

    ``depth_readout`` has no condition term at all, so its unadjusted condition
    effect is entirely inherited from the depth imbalance. That effect is real in
    the raw test and must not survive the audit.
    """

    table = audit(*build_cohort(depth_confounded=True))
    row = table.loc["depth_readout"]

    assert (
        row["raw_t_p"] < 0.05
    ), "the fixture must produce a raw effect for there to be one to kill"
    assert row["verdict"] == "depth_driven"
    assert "library-size" in row["reason"] or "reverses sign" in row["reason"]


def test_a_real_effect_survives_adjustment_in_the_same_cohort() -> None:
    """
    Verify the audit does not simply condemn everything in a confounded cohort.

    ``biology`` carries a genuine +1.0 shift that is independent of depth. If the
    audit flagged it, it would be discarding true findings on cohorts that happen
    to be depth-imbalanced -- the failure mode that would get it switched off.
    """

    table = audit(*build_cohort(depth_confounded=True))
    row = table.loc["biology"]

    assert row["depth_is_confounded"]
    assert row["verdict"] == "robust"
    assert row["reason"] == ""
    # Nearly all of it, not merely most: the within-sample slope cannot absorb the
    # condition effect, so a depth-independent finding comes through intact. A
    # pooled-slope adjustment removed 40% of this same known-true effect.
    assert row["delta_retained_fraction"] > 0.95
    # And the pooled correlation is the trap: it is inflated purely by the depth
    # imbalance acting through condition, while the within-sample one is ~0.
    assert row["spearman_rho_vs_depth"] > 0.3
    assert abs(row["within_sample_rho_vs_depth"]) < 0.1


def test_a_depth_coupled_metric_is_not_condemned_when_the_arms_are_balanced() -> None:
    """
    Verify the gate: confounding needs depth to differ by condition.

    ``depth_readout`` is just as depth-coupled here as in the confounded cohort,
    but with balanced arms there is nothing for depth to confound. Flagging on
    coupling alone would manufacture alarm on well-balanced datasets, which is
    most of them.
    """

    table = audit(*build_cohort(depth_confounded=False))
    row = table.loc["depth_readout"]

    assert not row["depth_is_confounded"]
    assert row["verdict"] == "depth_balanced"
    assert abs(row["spearman_rho_vs_depth"]) > 0.5


def test_strong_coupling_is_still_named_in_a_balanced_cohort() -> None:
    """
    Verify a safe-for-now metric is recorded as fragile.

    A metric that is half depth by rank is safe in this cohort and dangerous in
    the next one. Passing it silently would let the same claim be made on an
    imbalanced dataset with no warning in the record.
    """

    table = audit(*build_cohort(depth_confounded=False))

    assert "within-sample coupling" in table.loc["depth_readout", "reason"]
    # The gate's own explanation goes on every row, but the fragility warning must
    # not: attaching it to a depth-independent metric would make it noise.
    assert "within-sample coupling" not in table.loc["biology", "reason"]


def test_the_depth_gate_is_reported_identically_on_every_row() -> None:
    """
    Verify the cohort-level leg is evaluated once, not per metric.

    ``depth_is_confounded`` is a property of the design; a table where it varied
    by metric would mean the gate was being re-derived from each metric's own
    subset of cells.
    """

    table = audit(*build_cohort(depth_confounded=True))

    assert table["depth_is_confounded"].nunique() == 1
    assert table["depth_delta"].nunique() == 1
    assert table["depth_n_donors_positive"].iloc[0] == N_DONORS


def test_too_few_pairs_declines_to_reach_a_verdict() -> None:
    """
    Verify a small cohort is not told its effect failed adjustment.

    Below the paired floor a non-significant adjusted p reports the sample size.
    Calling that ``depth_driven`` would be a false accusation.
    """

    metrics, metadata = build_cohort(depth_confounded=True)
    keep = metadata["donor_id"].isin(["D0", "D1", "D2"])

    table = depth_confound_audit(
        metrics[keep],
        metadata[keep],
        donor_col="donor_id",
        condition_col="condition",
        case=CASE,
        control=CONTROL,
        depth_col="n_genes_by_counts",
    ).set_index("metric")

    assert set(table["verdict"]) == {"insufficient_pairs"}
    assert "below min_pairs" in table.loc["biology", "reason"]


def test_a_metric_with_no_raw_effect_is_not_audited() -> None:
    """
    Verify the audit only speaks about claims that exist.

    A metric that shows nothing before adjustment cannot have been rescued or
    destroyed by it, and labelling it ``robust`` would imply a finding.
    """

    metrics, metadata = build_cohort(depth_confounded=True)
    metrics = metrics.assign(flat=1.0 + np.arange(len(metrics)) % 2 * 1e-9)

    table = audit(metrics, metadata)

    assert table.loc["flat", "verdict"] == "no_raw_effect"


def test_an_effect_depth_was_hiding_comes_back_as_a_lead() -> None:
    """
    Verify the audit reports what removing depth *reveals*, not only what it kills.

    Confounding has a direction. A metric that rises with depth in a cohort whose
    case arm is deeper has its case mean pushed up, which cancels part of a genuine
    fall -- so the unadjusted test sees nothing and the adjusted one sees the whole
    effect. Filing that under ``no_raw_effect`` would hide the one row the audit
    found rather than protected, and a caller filtering on the verdict column would
    never see it.
    """

    metrics, metadata = build_cohort(depth_confounded=True)
    depth = metadata["n_genes_by_counts"].to_numpy(dtype=float)
    is_case = (metadata["condition"] == CASE).to_numpy()
    rng = np.random.default_rng(11)

    # Linear in log1p(depth), which is the scale the audit residualises on, so the
    # depth component is exactly removable and the ground truth is exact. The
    # condition coefficient is *derived* as the case arm's own depth advantage on
    # that scale, so the two cancel by construction rather than by tuning.
    lifted = 1.9 * np.log1p(depth)
    hidden = float(lifted[is_case].mean() - lifted[~is_case].mean())
    masked = lifted - hidden * is_case + rng.normal(0.0, 0.02, len(metrics))

    table = audit(metrics.assign(masked=masked), metadata)
    row = table.loc["masked"]

    assert row["raw_t_p"] >= 0.05, "the fixture must hide the effect before adjustment"
    assert row["adjusted_t_p"] < 0.05
    assert row["verdict"] == "depth_masked"
    # The whole fall, recovered: the raw contrast is ~0 and the adjusted one is the
    # coefficient that was cancelled.
    assert abs(row["raw_delta"]) < 0.1 * hidden
    assert row["adjusted_delta"] == pytest.approx(-hidden, abs=0.05)
    # And it is labelled a lead, because the unadjusted test is the one a declared
    # family would have been corrected over.
    assert "lead" in row["reason"]


def test_the_retained_fraction_is_blank_where_there_was_nothing_to_retain() -> None:
    """
    Verify a ratio to a non-effect is not printed as a percentage.

    ``delta_retained_fraction`` is ``|adjusted| / |raw|``, which only means "fraction of
    the effect that survived" if there was an effect. Where the raw contrast is
    indistinguishable from zero the ratio blows up, and it does so in the direction that
    reads as good news: a metric going from −0.0019 to −0.0116 prints as 620% retained.
    Someone will quote that. The verdict already says there was no unadjusted claim, so
    the fraction is withheld rather than left to be distrusted.
    """

    metrics, metadata = build_cohort(depth_confounded=True)
    depth = metadata["n_genes_by_counts"].to_numpy(dtype=float)
    is_case = (metadata["condition"] == CASE).to_numpy()
    rng = np.random.default_rng(11)
    lifted = 1.9 * np.log1p(depth)
    hidden = float(lifted[is_case].mean() - lifted[~is_case].mean())

    table = audit(
        metrics.assign(
            flat=1.0 + np.arange(len(metrics)) % 2 * 1e-9,
            masked=lifted - hidden * is_case + rng.normal(0.0, 0.02, len(metrics)),
        ),
        metadata,
    )

    assert table.loc["masked", "verdict"] == "depth_masked"
    assert np.isnan(table.loc["masked", "delta_retained_fraction"])
    assert np.isnan(table.loc["flat", "delta_retained_fraction"])
    # And it is still there where it means something: the audited rows keep it.
    assert np.isfinite(table.loc["biology", "delta_retained_fraction"])
    assert np.isfinite(table.loc["depth_readout", "delta_retained_fraction"])


def test_a_constant_metric_reports_no_correlation_without_warning() -> None:
    """
    Verify an all-zero metric column is answered, not warned about.

    The audit is used on genes as well as scores, and a gene undetected in one
    group's cells is a constant column. ``spearmanr`` has no coefficient to
    return there and warns; one warning per such column would bury the run's
    real output, and the honest value is simply undefined.
    """

    metrics, metadata = build_cohort(depth_confounded=True)
    metrics = metrics.assign(undetected=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = audit(metrics, metadata)

    row = table.loc["undetected"]
    assert np.isnan(row["spearman_rho_vs_depth"])
    assert np.isnan(row["within_sample_rho_vs_depth"])
    assert row["verdict"] == "no_raw_effect"


def test_misaligned_inputs_are_rejected() -> None:
    """
    Verify a silent row mismatch cannot produce a plausible table.

    Pairing metrics to the wrong cells' metadata yields numbers with no error, so
    the alignment has to be checked rather than assumed.
    """

    metrics, metadata = build_cohort(depth_confounded=True)

    with pytest.raises(ValueError, match="share an index"):
        depth_confound_audit(
            metrics,
            metadata.iloc[::-1].reset_index(drop=True),
            donor_col="donor_id",
            condition_col="condition",
            case=CASE,
            control=CONTROL,
            depth_col="n_genes_by_counts",
        )


def test_the_same_barcodes_in_two_index_dtypes_are_accepted() -> None:
    """
    Verify the alignment guard tests alignment and not dtype.

    AnnData holds ``obs_names`` as a pandas ``string`` index; a per-cell frame built from
    those same names — through a scoring library, a parquet round trip, or a plain
    ``DataFrame`` constructor — comes back as ``object``. The barcodes match element for
    element and the pair is perfectly aligned, so rejecting it sends the caller to fix
    something that is not broken, and the message it gets ("align them before calling") is
    advice they cannot act on.
    """

    metrics, metadata = build_cohort(depth_confounded=True)
    metadata = metadata.set_axis(pd.Index(metadata.index.astype(str), dtype="string"))

    table = depth_confound_audit(
        metrics,
        metadata,
        donor_col="donor_id",
        condition_col="condition",
        case=CASE,
        control=CONTROL,
        depth_col="n_genes_by_counts",
    )

    assert list(table["metric"]) == list(metrics.columns)


def test_a_missing_metadata_column_is_named() -> None:
    """Verify the error says which column, rather than raising deep in a groupby."""

    metrics, metadata = build_cohort(depth_confounded=True)

    with pytest.raises(KeyError, match="total_counts"):
        depth_confound_audit(
            metrics,
            metadata,
            donor_col="donor_id",
            condition_col="condition",
            case=CASE,
            control=CONTROL,
            depth_col="total_counts",
        )


# --------------------------------------------------------------------------- #
# depth_stratified_abundance
# --------------------------------------------------------------------------- #


def build_labelled_cohort(
    *, label_is_depth_stratum: bool, seed: int = 7
) -> tuple[pd.Series, pd.DataFrame]:
    """Build a paired cohort whose cluster labels have a known depth relationship.

    Args:
        label_is_depth_stratum: When True, ``target`` membership is decided purely
            by a cell's depth, so the case arm gains it only because its cells are
            deeper. When False, membership is decided by condition, independently
            of depth.
        seed: Seed for the deterministic fixture.

    Returns:
        ``(labels, metadata)`` aligned on a shared index.
    """
    rng = np.random.default_rng(seed)
    records = []
    for donor in range(N_DONORS):
        for condition in (CONTROL, CASE):
            # Donor medians are close and the within-sample spread is wide, so every
            # donor-sample has cells in every global depth quantile. With donor
            # medians spread as far apart as the arms are, the global quantiles
            # become donor strata and no stratum retains a full set of pairs --
            # which measures the fixture rather than the function.
            base_depth = 2400.0 + 40.0 * donor + (700.0 if condition == CASE else 0.0)
            depth = rng.normal(base_depth, 700.0, CELLS_PER_SAMPLE).clip(200.0)
            for index in range(CELLS_PER_SAMPLE):
                if label_is_depth_stratum:
                    is_target = depth[index] > 3000.0
                else:
                    is_target = rng.random() < (0.45 if condition == CASE else 0.10)
                records.append(
                    {
                        "cell": f"d{donor}_{condition}_{index}",
                        "donor_id": f"D{donor}",
                        "condition": condition,
                        "n_genes_by_counts": depth[index],
                        "label": "target" if is_target else "other",
                    }
                )
    frame = pd.DataFrame(records).set_index("cell")
    return frame["label"], frame[["donor_id", "condition", "n_genes_by_counts"]]


def stratified(labels: pd.Series, metadata: pd.DataFrame) -> pd.DataFrame:
    """Run the stratified abundance audit with the fixture's column names."""
    return depth_stratified_abundance(
        labels,
        metadata,
        donor_col="donor_id",
        condition_col="condition",
        case=CASE,
        control=CONTROL,
        depth_col="n_genes_by_counts",
    )


def test_a_cluster_that_is_a_depth_stratum_is_caught() -> None:
    """
    Verify the compositional version of the same failure is flagged.

    ``target`` membership here is a depth threshold and nothing else, so the case
    arm "gains" it purely by having deeper cells. Holding depth fixed inside bins
    must dissolve the shift.
    """

    table = stratified(*build_labelled_cohort(label_is_depth_stratum=True))
    pooled = table[(table["label"] == "target") & (table["stratum"] == "all")].iloc[0]

    assert (
        pooled["t_p"] < 0.05
    ), "the fixture must produce a pooled shift for there to be one to kill"
    assert pooled["verdict"] == "depth_driven"
    assert "depth strata" in pooled["reason"]


def test_a_real_compositional_shift_holds_inside_depth_strata() -> None:
    """
    Verify a genuine abundance shift is not condemned by stratification.

    Membership is driven by condition and is independent of depth, so every
    stratum should show the same direction -- which is precisely the evidence
    that distinguishes it from the case above.
    """

    table = stratified(*build_labelled_cohort(label_is_depth_stratum=False))
    pooled = table[(table["label"] == "target") & (table["stratum"] == "all")].iloc[0]

    assert pooled["verdict"] == "robust"
    assert pooled["n_strata_same_sign"] == pooled["n_strata"]


def test_stratum_rows_are_present_for_inspection() -> None:
    """
    Verify the per-stratum evidence is returned, not just the verdict.

    The verdict is a summary of the stratum rows; a reviewer asking "in which
    depth bins does this hold" has to be answerable from the table.
    """

    table = stratified(*build_labelled_cohort(label_is_depth_stratum=False))

    strata = set(table["stratum"])
    assert "all" in strata
    assert {"q1", "q2", "q3", "q4", "q5"} <= strata
    target = table[table["label"] == "target"]
    assert target[target["stratum"] != "all"]["n_pairs"].min() == N_DONORS


def test_summary_columns_are_only_on_the_pooled_row() -> None:
    """
    Verify a per-stratum row does not carry a verdict.

    A verdict on ``q3`` alone would invite quoting the one stratum that agrees
    with the author, which is the opposite of what stratification is for.
    """

    table = stratified(*build_labelled_cohort(label_is_depth_stratum=False))
    per_stratum = table[table["stratum"] != "all"]

    assert per_stratum["verdict"].isna().all()
    assert per_stratum["n_strata_same_sign"].isna().all()


def test_proportions_are_taken_within_the_stratum() -> None:
    """
    Verify a stratum's proportions use that stratum's cells as the denominator.

    Dividing a stratum's count by the whole sample's size would make every
    stratum proportion shrink by roughly the bin fraction and turn the comparison
    across bins into a comparison of bin sizes.
    """

    labels, metadata = build_labelled_cohort(label_is_depth_stratum=False)
    table = stratified(labels, metadata)

    for stratum in ("all", "q1", "q5"):
        block = table[table["stratum"] == stratum]
        assert np.isclose(block["mean_case"].sum(), 1.0, atol=1e-6)
        assert np.isclose(block["mean_control"].sum(), 1.0, atol=1e-6)
