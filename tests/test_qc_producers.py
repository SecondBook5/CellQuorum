"""Tests for QC evidence producers: comparability, invariance, degeneracy, ground truth.

These are the tests that make "it works every time" a claim rather than a hope. Four
groups, each targeting a way a per-sample QC system silently misbehaves:

    1. Comparability   one bar must mean one thing on every axis
    2. Invariance      subsetting, duplicating, reordering must not change a verdict
    3. Degeneracy      tiny, uniform, or empty inputs must degrade conservatively
    4. Ground truth    synthetic damage must be caught; healthy cells must not be

Group 4 is the only one that can say the system detects damage rather than merely flagging
things, so it uses the mechanism miQC models: membrane failure leaks cytoplasmic mRNA, so
complexity falls while mitochondrial fraction rises.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.qc.evidence import (
    AdjudicationPolicy,
    Direction,
    EvidenceAvailability,
    EvidenceFamily,
    QCStateInitial,
    adjudicate_initial,
)
from cellquorum.stages.qc.producers import (
    DEFAULT_HALF_SEVERITY_Z,
    MIN_CELLS_FOR_NULL,
    build_evidence_table,
    fit_robust_null,
    multiplet_agreement_severity,
    tail_severity,
)

RNG_SEED = 0


def _policy(**overrides: float) -> AdjudicationPolicy:
    """Test-bench policy. NOT calibrated defaults."""
    bars: dict[str, float] = {
        "concern_severity": 0.50,
        "severe_severity": 0.70,
        "min_concordant_families": 2,
        "uninformative_capture_severity": 0.90,
        "min_coverage_for_quarantine": 0.50,
        "multiplet_severity": 0.70,
    }
    bars.update(overrides)
    return AdjudicationPolicy(**bars)  # type: ignore[arg-type]


def _cohort(
    n_per_sample: int = 400,
    n_samples: int = 3,
    n_genes: int = 200,
    *,
    damaged_fraction: float = 0.0,
    depth_scale: dict[str, float] | None = None,
    seed: int = RNG_SEED,
) -> ad.AnnData:
    """Build a synthetic cohort with an optional known-damaged subset.

    Damage follows the mechanism a real dying cell shows: the membrane fails, cytoplasmic
    mRNA leaks out, so detected genes and total counts fall while the mitochondrial
    fraction rises. ``obs["is_damaged"]`` is the ground truth.

    Args:
        n_per_sample: Cells per sample.
        n_samples: Number of samples.
        n_genes: Genes, including MT- and stress genes.
        damaged_fraction: Fraction of each sample that is genuinely damaged.
        depth_scale: Per-sample multiplier on library size, for depth-heterogeneity tests.
        seed: RNG seed; every call is deterministic.
    """
    rng = np.random.default_rng(seed)
    mito = [f"MT-{i}" for i in range(8)]
    stress = ["FOS", "JUN", "HSPA1A", "HSPA1B", "EGR1"]
    other = [f"G{i}" for i in range(n_genes - len(mito) - len(stress) - 1)]
    genes = [*mito, *stress, "MALAT1", *other]

    blocks, obs_rows = [], []
    for sample_index in range(n_samples):
        sample = f"S{sample_index + 1}"
        scale = (depth_scale or {}).get(sample, 1.0)
        n_damaged = int(round(n_per_sample * damaged_fraction))

        # Healthy cells: moderate depth, low mitochondrial share.
        healthy = rng.poisson(6.0 * scale, size=(n_per_sample - n_damaged, len(genes)))
        healthy[:, : len(mito)] = rng.poisson(0.5 * scale, size=(len(healthy), len(mito)))

        # Damaged cells: complexity collapses, mitochondrial share rises.
        damaged = rng.poisson(0.6 * scale, size=(n_damaged, len(genes)))
        damaged[:, : len(mito)] = rng.poisson(9.0 * scale, size=(n_damaged, len(mito)))

        block = np.vstack([healthy, damaged]).astype(np.float32)
        blocks.append(block)
        obs_rows.extend(
            {"sample_id": sample, "is_damaged": index >= len(healthy)}
            for index in range(len(block))
        )

    matrix = np.vstack(blocks)
    obs = pd.DataFrame(obs_rows, index=[f"cell_{i}" for i in range(len(matrix))])
    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = matrix.copy()
    return adata


def _metrics(adata: ad.AnnData) -> pd.DataFrame:
    """Minimal QC metric frame, computed the way the QC stage would."""
    counts = adata.layers["counts"]
    mito_columns = [i for i, name in enumerate(adata.var_names) if str(name).startswith("MT-")]
    total = counts.sum(axis=1)
    return pd.DataFrame(
        {
            "total_counts": total,
            "n_genes_by_counts": (counts > 0).sum(axis=1),
            "pct_counts_mito": 100.0 * counts[:, mito_columns].sum(axis=1) / np.maximum(total, 1),
        },
        index=adata.obs_names,
    )


def _table(adata: ad.AnnData, **kwargs):
    """Build the evidence table for a synthetic cohort."""
    return build_evidence_table(
        adata, _metrics(adata), group_key="sample_id", layer="counts", **kwargs
    )


# ═══ 1. Comparability: one bar means one thing on every axis ════════════════════════


def test_healthy_cohort_severity_stays_near_zero_on_every_axis() -> None:
    """A clean cohort must not manufacture concern on any axis.

    This is the failure percentile-based severity cannot avoid: forcing a uniform
    distribution means the worst few per cent always score near 1, however clean the data.
    """
    table = _table(_cohort())
    severity = table.family_severity()

    for family in severity.columns:
        assert (
            severity[family].quantile(0.98) < 0.5
        ), f"{family}: a healthy cohort put its 98th percentile above the concern bar"


def test_axes_share_a_stringency_at_a_common_bar() -> None:
    """The same bar must flag a comparable share of a healthy cohort on each axis.

    The predecessor mapping flagged 13% on complexity and 1.5% on stress at one nominal
    0.50 — a 9x stringency spread produced by the scaling, not by any decision.
    """
    table = _table(_cohort(n_per_sample=600))
    severity = table.family_severity()

    flagged = {family: float((severity[family] >= 0.5).mean()) for family in severity.columns}
    assert max(flagged.values()) < 0.10, f"a healthy cohort over-flags somewhere: {flagged}"


def test_severity_is_monotone_in_the_metric() -> None:
    """A worse metric value must never produce a lower severity."""
    values = pd.Series(np.linspace(0.0, 50.0, 200))
    groups = pd.Series(["A"] * 200)

    upper = tail_severity(values, groups, direction=Direction.UPPER_TAIL)
    assert (upper.diff().dropna() >= -1e-12).all()

    lower = tail_severity(values, groups, direction=Direction.LOWER_TAIL)
    assert (lower.diff().dropna() <= 1e-12).all()


def test_half_severity_z_lands_severity_at_one_half() -> None:
    """Severity must be 0.5 at exactly the configured robust z — the scale's anchor."""
    # A tight normal group gives a predictable robust sigma.
    rng = np.random.default_rng(RNG_SEED)
    values = pd.Series(rng.normal(100.0, 10.0, size=4000))
    groups = pd.Series(["A"] * len(values))

    null = fit_robust_null(values, groups)
    location, scale = float(null.location.iloc[0]), float(null.scale.iloc[0])

    at_k = pd.Series([location + DEFAULT_HALF_SEVERITY_Z * scale])
    combined = pd.concat([values, at_k], ignore_index=True)
    severity = tail_severity(
        combined, pd.Series(["A"] * len(combined)), direction=Direction.UPPER_TAIL
    )
    assert severity.iloc[-1] == pytest.approx(0.5, abs=0.02)


# ═══ 2. Invariance: verdicts must not depend on incidental structure ════════════════


def test_reordering_cells_gives_identical_severity() -> None:
    """Output must not depend on row order."""
    adata = _cohort()
    original = _table(adata).family_severity()

    order = np.random.default_rng(1).permutation(adata.n_obs)
    shuffled = _table(adata[order].copy()).family_severity()

    pd.testing.assert_frame_equal(
        original.sort_index(), shuffled.sort_index(), check_like=True, atol=1e-12
    )


def test_adding_a_clean_sample_does_not_change_other_samples() -> None:
    """Nulls are per sample, so an unrelated library must not shift existing verdicts.

    A pooled null would fail this, and the failure would look like a batch effect.
    """
    base = _cohort(n_samples=2, damaged_fraction=0.05)
    before = _table(base).family_severity()

    extra = _cohort(n_samples=1, seed=99)
    extra.obs["sample_id"] = "S_extra"
    extra.obs_names = [f"extra_{i}" for i in range(extra.n_obs)]
    combined = ad.concat([base, extra])
    combined.layers["counts"] = combined.X.copy()

    after = _table(combined).family_severity().loc[before.index]
    pd.testing.assert_frame_equal(before, after, check_like=True, atol=1e-12)


def test_duplicating_a_sample_does_not_change_its_verdicts() -> None:
    """Doubling a library's cells leaves its distribution — and so its nulls — unchanged."""
    base = _cohort(n_samples=1, damaged_fraction=0.08)
    before = _table(base).family_severity()

    twin = base.copy()
    twin.obs_names = [f"dup_{name}" for name in twin.obs_names]
    doubled = ad.concat([base, twin])
    doubled.layers["counts"] = doubled.X.copy()

    after = _table(doubled).family_severity().loc[before.index]
    pd.testing.assert_frame_equal(before, after, check_like=True, atol=1e-10)


def test_sample_depth_does_not_drive_severity() -> None:
    """A uniformly shallower library must not be condemned for its depth.

    Per-sample nulls are the whole defence against differential removal between arms.
    """
    adata = _cohort(n_samples=2, depth_scale={"S2": 0.35})
    severity = _table(adata).family_severity()
    capture = severity[str(EvidenceFamily.CAPTURE_COMPLEXITY)]

    shallow = capture[adata.obs["sample_id"] == "S2"].mean()
    deep = capture[adata.obs["sample_id"] == "S1"].mean()
    assert (
        abs(shallow - deep) < 0.10
    ), f"depth leaked into severity: shallow {shallow:.3f} vs deep {deep:.3f}"


def test_repeated_runs_are_deterministic() -> None:
    """Identical input must give byte-identical output."""
    adata = _cohort()
    first = _table(adata).family_severity()
    second = _table(adata).family_severity()
    pd.testing.assert_frame_equal(first, second)


# ═══ 3. Degeneracy: unusual inputs must degrade conservatively ══════════════════════


def test_group_below_the_minimum_yields_no_severity() -> None:
    """Too few cells cannot support a null, so the axis must abstain, not guess."""
    values = pd.Series(np.arange(MIN_CELLS_FOR_NULL - 1, dtype=float))
    groups = pd.Series(["tiny"] * len(values))
    assert tail_severity(values, groups, direction=Direction.UPPER_TAIL).isna().all()


def test_zero_variance_group_yields_no_severity() -> None:
    """Identical cells have no healthy mode to speak of."""
    values = pd.Series([5.0] * 200)
    groups = pd.Series(["flat"] * 200)
    assert tail_severity(values, groups, direction=Direction.UPPER_TAIL).isna().all()


def test_zero_inflated_metric_still_produces_severity() -> None:
    """A median and MAD of exactly zero must fall back, not abstain.

    MALAT1 and stress fractions are zero for most cells; abandoning those axes would drop
    two evidence families on every dataset.
    """
    values = pd.Series([0.0] * 300 + [0.02] * 60 + [0.4] * 12)
    groups = pd.Series(["zi"] * len(values))
    severity = tail_severity(values, groups, direction=Direction.UPPER_TAIL)

    assert severity.notna().all()
    assert severity.iloc[0] == 0.0
    assert severity.iloc[-1] > 0.5


def test_unscorable_cells_become_model_unstable_not_zero() -> None:
    """An abstaining axis must lower coverage, never look reassuring."""
    adata = _cohort(n_per_sample=10, n_samples=1)  # below MIN_CELLS_FOR_NULL
    table = _table(adata)

    for axis in table.axes:
        if axis.name == "mito_mixture_posterior":
            continue
        assert set(axis.availability.unique()) <= {str(EvidenceAvailability.COMPUTATION_FAILED)}
    assert (table.evidence_coverage() == 0.0).all()


def test_single_sample_cohort_works() -> None:
    """One library is a legitimate dataset, not an edge case to crash on."""
    table = _table(_cohort(n_samples=1, damaged_fraction=0.05))
    assert table.family_severity().notna().any().any()


def test_snrna_drops_the_nuclear_axis_rather_than_misreading_it() -> None:
    """High nuclear-retained signal is expected in single-nucleus data."""
    adata = _cohort()
    table = _table(adata, nuclear_axis_applicable=False)
    assert EvidenceFamily.NUCLEAR_INTEGRITY not in table.families_present()


# ═══ 4. Ground truth: synthetic damage must be caught ══════════════════════════════


def test_synthetic_damage_is_separated_from_healthy_cells() -> None:
    """Damaged cells must score far higher than healthy ones on the damage families.

    The only test here that says the system detects damage rather than flagging things.
    """
    adata = _cohort(n_per_sample=500, damaged_fraction=0.10)
    damage = _table(adata).damage_family_severity().max(axis=1)
    truth = adata.obs["is_damaged"].to_numpy(dtype=bool)

    assert damage[truth].median() > 0.7, "damaged cells were not scored as damaged"
    # The property that matters is that healthy cells stay clear of the concern bar. A
    # max-over-families rollup is noise-elevated by construction, so its median is not the
    # thing to bound.
    assert damage[~truth].quantile(0.95) < 0.5, "healthy cells reached the concern bar"
    assert damage[truth].median() - damage[~truth].median() > 0.4, "poor separation"


def test_damaged_cells_are_quarantined_and_healthy_cells_are_not() -> None:
    """End to end: ground truth in, recall and precision out."""
    adata = _cohort(n_per_sample=500, damaged_fraction=0.10)
    result = adjudicate_initial(_table(adata), _policy())

    truth = adata.obs["is_damaged"].to_numpy(dtype=bool)
    condemned = (result.state == str(QCStateInitial.QUARANTINE)).to_numpy()

    recall = condemned[truth].mean()
    precision = condemned[condemned].size and condemned[truth].sum() / max(condemned.sum(), 1)
    assert recall > 0.75, f"recall on known damage only {recall:.2f}"
    assert precision > 0.90, f"precision only {precision:.2f}"


def test_no_healthy_cell_is_quarantined_in_a_clean_cohort() -> None:
    """With nothing wrong, nothing may be condemned.

    The single most important property: a clean dataset must survive QC intact.
    """
    result = adjudicate_initial(_table(_cohort(n_per_sample=500)), _policy())
    n_quarantined = int((result.state == str(QCStateInitial.QUARANTINE)).sum())
    assert n_quarantined == 0, f"{n_quarantined} healthy cells were quarantined"


def test_damage_needs_concordance_so_one_axis_cannot_condemn() -> None:
    """Raising mitochondrial fraction alone must not quarantine.

    Elevated mitochondrial content without complexity collapse is a metabolically unusual
    but intact cell — exactly what the old fixed ceiling deleted.
    """
    adata = _cohort(n_per_sample=400, n_samples=1)
    mito_columns = [i for i, n in enumerate(adata.var_names) if str(n).startswith("MT-")]
    hot = np.arange(20)

    counts = adata.layers["counts"].copy()
    counts[np.ix_(hot, mito_columns)] *= 20.0  # mito way up, complexity untouched
    adata.layers["counts"] = counts
    adata.X = counts

    result = adjudicate_initial(_table(adata), _policy())
    assert (result.state.iloc[hot] != str(QCStateInitial.QUARANTINE)).all()


# ═══ 5. Multiplet agreement ════════════════════════════════════════════════════════


def test_detector_scale_difference_no_longer_silences_the_family() -> None:
    """Normalising before combining is what makes agreement mean agreement.

    On the real cohort scDblFinder ran at median 0.110 and Scrublet at 0.031, so a minimum
    of the raw scores returned Scrublet verbatim and flagged nothing in 201,923 cells.
    """
    rng = np.random.default_rng(RNG_SEED)
    n = 500
    obs = pd.DataFrame(
        {
            "sample_id": ["A"] * n,
            # Same cells extreme in both detectors, on deliberately different scales.
            "doublet_score_scdblfinder": rng.normal(0.110, 0.02, n),
            "doublet_score_scrublet": rng.normal(0.031, 0.006, n),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    agreed = np.arange(10)
    obs.loc[obs.index[agreed], "doublet_score_scdblfinder"] = 0.95
    obs.loc[obs.index[agreed], "doublet_score_scrublet"] = 0.60

    severity = multiplet_agreement_severity(obs, obs["sample_id"])
    assert severity is not None
    assert (severity.iloc[agreed] > 0.7).all(), "agreed doublets were not flagged"
    assert severity.drop(severity.index[agreed]).quantile(0.98) < 0.5


def test_one_detector_alone_does_not_reach_agreement() -> None:
    """A cell extreme in only one detector must stay low — that is the requirement."""
    rng = np.random.default_rng(RNG_SEED)
    n = 500
    obs = pd.DataFrame(
        {
            "sample_id": ["A"] * n,
            "doublet_score_scdblfinder": rng.normal(0.110, 0.02, n),
            "doublet_score_scrublet": rng.normal(0.031, 0.006, n),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    lone = np.arange(10)
    obs.loc[obs.index[lone], "doublet_score_scdblfinder"] = 0.95  # scrublet left normal

    severity = multiplet_agreement_severity(obs, obs["sample_id"])
    assert severity is not None
    assert (severity.iloc[lone] < 0.5).all()


def test_redundant_doublet_score_column_is_not_double_counted() -> None:
    """`doublet_score` copies a detector, so it must not satisfy agreement twice."""
    rng = np.random.default_rng(RNG_SEED)
    n = 300
    scdbl = rng.normal(0.110, 0.02, n)
    obs = pd.DataFrame(
        {
            "sample_id": ["A"] * n,
            "doublet_score_scdblfinder": scdbl,
            "doublet_score": scdbl.copy(),
            "doublet_score_scrublet": rng.normal(0.031, 0.006, n),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    lone = np.arange(8)
    obs.loc[obs.index[lone], ["doublet_score_scdblfinder", "doublet_score"]] = 0.95

    severity = multiplet_agreement_severity(obs, obs["sample_id"])
    assert severity is not None
    # Scrublet still disagrees, so agreement must remain low despite two matching columns.
    assert (severity.iloc[lone] < 0.5).all()
