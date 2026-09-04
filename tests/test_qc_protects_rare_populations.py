"""A healthy rare cell type must survive QC. This is the defect that motivated lineages.

The measurement that forced this module into existence: a cohort of 950 ordinary cells plus 50
perfectly healthy cells whose *constitutive* biology is low-complexity and high-mitochondrial —
the neutrophil / erythrocyte / plasma-cell profile — quarantined **50 of 50** rare cells and 0
of 950 ordinary ones.

Two independent routes fired, so it was not a single tunable mistake:

    concordance     capture 0.946 + metabolic 0.974 = two severe damage families
    uninformative   capture >= 0.90 alone declares the barcode devoid of information

Both statements about the data were true. Neither meant the cell was damaged. Against a
sample-wide null a rare cell type and a dying cell are *geometrically identical*, so no
threshold separates them — which is why the fix is a change of reference class, not of number.

These tests pin both directions, because a fix that merely stops quarantining things would be
worthless:

    1. The healthy rare population survives                (no false deletion)
    2. Genuinely damaged cells are still caught            (no loss of power)
    3. A uniformly damaged lineage is still surfaced       (the new failure mode)
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.qc.evidence import AdjudicationPolicy, QCStateInitial, adjudicate_initial
from cellquorum.stages.qc.lineage import (
    UNASSIGNED,
    audit_lineages,
    provisional_lineages,
    resolve_null_groups,
)
from cellquorum.stages.qc.producers import build_evidence_table

#: The production-calibrated policy from QCGradedConfig, not a test-bench one. Using the real
#: bars is the point: the defect was reachable with the shipped configuration.
POLICY = AdjudicationPolicy(
    concern_severity=0.50,
    severe_severity=0.667,
    min_concordant_families=2,
    uninformative_capture_severity=0.90,
    min_coverage_for_quarantine=0.50,
    multiplet_severity=0.60,
)

MITO = [f"MT-{i}" for i in range(8)]


def _cohort(
    n_ordinary: int = 950,
    n_rare: int = 50,
    n_damaged: int = 0,
    n_genes: int = 200,
    *,
    seed: int = 0,
) -> ad.AnnData:
    """Ordinary cells, a healthy rare population, and optionally real damage.

    The rare population is healthy and *coherent*: every one of its cells expresses the same
    private marker block, which is what makes it a population rather than a smear. The damaged
    cells are deliberately incoherent — each is degraded toward a different random subset —
    because that is the real difference between damage and identity, and a fixture that made
    damage coherent would let the lineage grouping cheat.
    """
    rng = np.random.default_rng(seed)
    markers = [f"RARE{i}" for i in range(15)]
    genes = [*MITO, "MALAT1", *markers, *[f"G{i}" for i in range(n_genes - 24)]]
    n_g = len(genes)

    blocks, kinds = [], []

    ordinary = rng.poisson(6.0, size=(n_ordinary, n_g))
    ordinary[:, :8] = rng.poisson(0.5, size=(n_ordinary, 8))
    blocks.append(ordinary)
    kinds += ["ordinary"] * n_ordinary

    # Healthy but constitutively low-RNA and high-mito, plus its own marker programme.
    rare = rng.poisson(1.0, size=(n_rare, n_g))
    rare[:, :8] = rng.poisson(5.0, size=(n_rare, 8))
    rare[:, 9:24] += rng.poisson(15.0, size=(n_rare, 15))
    blocks.append(rare)
    kinds += ["rare"] * n_rare

    if n_damaged:
        # Incoherent degradation: each cell keeps a different random handful of genes.
        damaged = np.zeros((n_damaged, n_g), dtype=int)
        for row in range(n_damaged):
            kept = rng.choice(n_g, size=rng.integers(15, 40), replace=False)
            damaged[row, kept] = rng.poisson(2.0, size=len(kept))
        damaged[:, :8] = rng.poisson(12.0, size=(n_damaged, 8))
        blocks.append(damaged)
        kinds += ["damaged"] * n_damaged

    matrix = np.vstack(blocks).astype(np.float32)
    obs = pd.DataFrame(
        {"sample_id": "S1", "kind": kinds},
        index=[f"cell_{i}" for i in range(len(matrix))],
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = matrix.copy()
    return adata


def _metrics(adata: ad.AnnData) -> pd.DataFrame:
    """The QC metric frame, as the stage computes it."""
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


def _adjudicate(adata: ad.AnnData, *, lineage_conditional: bool):
    """Run the graded path with or without lineage conditioning."""
    metrics = _metrics(adata)
    grouping = None
    lineage = None
    if lineage_conditional:
        lineage = provisional_lineages(adata, layer="counts", resolution=0.5)
        grouping = resolve_null_groups(
            adata.obs, sample_key="sample_id", lineage=lineage, min_cells=25
        )

    evidence = build_evidence_table(
        adata,
        metrics,
        group_key="sample_id",
        layer="counts",
        grouping=grouping,
        lineage_conditional=grouping is not None,
    )
    return adjudicate_initial(evidence, POLICY), evidence, lineage


def _fraction(result, adata: ad.AnnData, kind: str, state: QCStateInitial) -> float:
    """Fraction of one cell kind assigned one state."""
    mask = (adata.obs["kind"] == kind).to_numpy()
    return float((result.state[mask] == str(state)).mean())


# ═══ 1. The rare population survives ═══════════════════════════════════════════════


def test_a_healthy_rare_population_is_not_quarantined() -> None:
    """The defect, as a regression test. 50/50 deleted before lineage conditioning."""
    adata = _cohort()
    result, _, _ = _adjudicate(adata, lineage_conditional=True)

    quarantined = _fraction(result, adata, "rare", QCStateInitial.QUARANTINE)
    assert quarantined == 0.0, (
        f"{quarantined:.0%} of a healthy rare population was quarantined for having unusual "
        f"baseline biology"
    )


def test_the_rare_population_can_still_fit_the_reference() -> None:
    """Surviving is not enough: barred from fitting, a population is undiscoverable.

    With core-only fitting downstream, a rare population stuck in borderline never shapes the
    manifold and never forms its own cluster — it gets absorbed into the nearest core cluster
    by label transfer. Not being quarantined is worth little if it cannot fit.
    """
    adata = _cohort()
    result, _, _ = _adjudicate(adata, lineage_conditional=True)

    core = _fraction(result, adata, "rare", QCStateInitial.CORE)
    assert core > 0.5, f"only {core:.0%} of the rare population may fit the reference"


def test_the_control_the_rare_population_really_is_deleted_without_lineages() -> None:
    """Without lineage conditioning the deletion is total, so the fix is doing the work."""
    adata = _cohort()
    result, _, _ = _adjudicate(adata, lineage_conditional=False)

    quarantined = _fraction(result, adata, "rare", QCStateInitial.QUARANTINE)
    assert quarantined > 0.5, (
        f"only {quarantined:.0%} of the rare population was quarantined without lineage "
        f"conditioning, so the fixture no longer demonstrates the defect the fix addresses"
    )


def test_ordinary_cells_are_unaffected() -> None:
    """The fix must not change the common case."""
    adata = _cohort()
    result, _, _ = _adjudicate(adata, lineage_conditional=True)

    assert _fraction(result, adata, "ordinary", QCStateInitial.QUARANTINE) == 0.0
    assert _fraction(result, adata, "ordinary", QCStateInitial.CORE) > 0.9


# ═══ 2. Real damage is still caught ════════════════════════════════════════════════


def test_genuine_damage_is_still_detected_alongside_a_rare_population() -> None:
    """The hard case: both present at once, and they must be separated.

    A fix that stopped quarantining anything would pass every test above and be useless. This
    is the test that makes those meaningful.
    """
    adata = _cohort(n_damaged=60)
    result, _, _ = _adjudicate(adata, lineage_conditional=True)

    damaged_flagged = 1.0 - _fraction(result, adata, "damaged", QCStateInitial.CORE)
    rare_quarantined = _fraction(result, adata, "rare", QCStateInitial.QUARANTINE)

    assert damaged_flagged > 0.75, f"only {damaged_flagged:.0%} of real damage was flagged"
    assert rare_quarantined == 0.0, "the rare population was condemned alongside real damage"


def test_damage_and_rarity_end_up_in_different_states() -> None:
    """Separation, stated as the comparison that matters."""
    adata = _cohort(n_damaged=60)
    result, _, _ = _adjudicate(adata, lineage_conditional=True)

    rare_core = _fraction(result, adata, "rare", QCStateInitial.CORE)
    damaged_core = _fraction(result, adata, "damaged", QCStateInitial.CORE)
    assert rare_core > damaged_core, (
        f"rare cells ({rare_core:.0%} core) are not being distinguished from damaged cells "
        f"({damaged_core:.0%} core)"
    )


# ═══ 3. The new failure mode: a lineage that is uniformly damaged ══════════════════


def test_a_uniformly_damaged_lineage_is_flagged_as_suspect() -> None:
    """Lineage conditioning exonerates a debris cluster by its own uniformity.

    Every cell in a debris group looks ordinary next to neighbours that are also debris, so no
    per-cell verdict can catch it. It has to be surfaced as a group-level judgement instead,
    which is what the audit is for.
    """
    adata = _cohort(n_damaged=80)
    result, evidence, lineage = _adjudicate(adata, lineage_conditional=True)

    # The audit reads absolute severity: "this whole group is bad" is not expressible on a
    # within-lineage scale.
    absolute = build_evidence_table(
        adata, _metrics(adata), group_key="sample_id", layer="counts"
    ).damage_family_severity()
    excluded = result.state != str(QCStateInitial.CORE)

    audit = audit_lineages(lineage, absolute, excluded)
    assert audit[
        "suspect"
    ].any(), "no lineage was flagged suspect even though 80 incoherent damaged cells are present"


def test_the_audit_reports_a_row_per_lineage_with_the_expected_columns() -> None:
    """The audit is a user-facing artifact, so its shape is part of the contract."""
    adata = _cohort(n_damaged=40)
    result, _, lineage = _adjudicate(adata, lineage_conditional=True)
    absolute = build_evidence_table(
        adata, _metrics(adata), group_key="sample_id", layer="counts"
    ).damage_family_severity()

    audit = audit_lineages(lineage, absolute, result.state != str(QCStateInitial.CORE))

    assert set(audit.columns) == {
        "n_cells",
        "median_absolute_severity",
        "excluded_fraction",
        "damage_excluded_fraction",
        "multiplet_fraction",
        "suspect",
        "vulnerable",
    }
    assert audit["n_cells"].sum() == adata.n_obs


def test_a_doublet_cluster_is_not_reported_as_a_lost_population() -> None:
    """Multiplet-driven exclusion must not read as rare-population loss.

    Found on the validation cohort, not imagined: a 2,111-cell lineage was flagged vulnerable at
    83% excluded while carrying the *lowest* absolute severity of any lineage. It was a doublet
    cluster — 50.5% called doublets against 1.9% cohort-wide, scDblFinder 0.693 vs 0.110, 3,668
    genes vs 2,051, mitochondrial content below average. Excellent libraries that are simply not
    one cell each. Separating the two exclusion causes dropped its damage-driven rate to 28% and
    the false alarm with it.
    """
    adata = _cohort(n_ordinary=300, n_rare=0)
    result, _, lineage = _adjudicate(adata, lineage_conditional=True)
    absolute = build_evidence_table(
        adata, _metrics(adata), group_key="sample_id", layer="counts"
    ).damage_family_severity()

    # Everything excluded, but every exclusion is a multiplet call.
    everyone = pd.Series(True, index=adata.obs_names)
    audit = audit_lineages(lineage, absolute, everyone, everyone)

    assert (audit["excluded_fraction"] == 1.0).all()
    assert (audit["multiplet_fraction"] == 1.0).all()
    assert not audit["vulnerable"].any(), (
        "a lineage excluded entirely because its cells are doublets was reported as a real "
        "population being lost"
    )


def test_damage_driven_exclusion_still_raises_the_alarm() -> None:
    """The counterpart: with no multiplets, exclusion must still flag as vulnerable."""
    adata = _cohort(n_ordinary=300, n_rare=0)
    result, _, lineage = _adjudicate(adata, lineage_conditional=True)
    absolute = build_evidence_table(
        adata, _metrics(adata), group_key="sample_id", layer="counts"
    ).damage_family_severity()

    everyone = pd.Series(True, index=adata.obs_names)
    nobody = pd.Series(False, index=adata.obs_names)
    audit = audit_lineages(lineage, absolute, everyone, nobody)

    assert audit["vulnerable"].all()


# ═══ 4. The grouping and null hierarchy themselves ═════════════════════════════════


def test_the_rare_population_forms_its_own_provisional_lineage() -> None:
    """Everything rests on the grouping actually separating them.

    Leiden needs density, so a 50-cell population is exactly what it might merge away. If this
    fails, the rest of the module is resting on nothing.
    """
    adata = _cohort()
    lineage = provisional_lineages(adata, layer="counts", resolution=0.5)

    rare = (adata.obs["kind"] == "rare").to_numpy()
    rare_labels = set(lineage[rare]) - {UNASSIGNED}
    ordinary_labels = set(lineage[~rare]) - {UNASSIGNED}

    assert rare_labels, "the rare population was not grouped at all"
    assert not (rare_labels & ordinary_labels), (
        f"the rare population shares lineage {rare_labels & ordinary_labels} with ordinary "
        f"cells, so a within-lineage null cannot protect it"
    )


def test_cells_below_the_gene_floor_are_left_unassigned() -> None:
    """An empty barcode cannot be a rare cell type, so it must not anchor a group."""
    adata = _cohort(n_ordinary=200, n_rare=30)
    adata.layers["counts"][:10, :] = 0.0
    adata.X = adata.layers["counts"]

    lineage = provisional_lineages(adata, layer="counts", min_genes=50)
    assert (lineage.iloc[:10] == UNASSIGNED).all()


def test_a_small_lineage_borrows_a_coarser_null_rather_than_none() -> None:
    """Falling back widens the null and lowers severity — the conservative direction."""
    obs = pd.DataFrame(
        {"sample_id": ["S1"] * 60 + ["S2"] * 60},
        index=[f"c{i}" for i in range(120)],
    )
    # One lineage is abundant per sample; the other has 5 cells per sample but 10 overall.
    lineage = pd.Series(["L0"] * 55 + ["L1"] * 5 + ["L0"] * 55 + ["L1"] * 5, index=obs.index)

    grouping = resolve_null_groups(obs, sample_key="sample_id", lineage=lineage, min_cells=25)
    levels = grouping.level

    assert (levels[lineage == "L0"] == "sample_x_lineage").all()
    # L1 cannot support a per-sample null (5 cells), so it falls to a coarser level. Nothing is
    # left without a reference class.
    assert levels.notna().all()
    assert set(levels[lineage == "L1"]) <= {"lineage", "sample", "pooled"}

    # The nesting that matters: the level L1 fell back to must be estimated over every cell
    # carrying that key, not only over the cells that fell back to it.
    fallback = levels[lineage == "L1"].iloc[0]
    keys = grouping.keys[fallback]
    assert int((keys == keys[lineage == "L1"].iloc[0]).sum()) > int((lineage == "L1").sum())


def test_unassigned_cells_still_receive_a_null() -> None:
    """No cell may be left without a reference class; that would read as absent evidence."""
    obs = pd.DataFrame({"sample_id": ["S1"] * 60}, index=[f"c{i}" for i in range(60)])
    lineage = pd.Series([UNASSIGNED] * 10 + ["L0"] * 50, index=obs.index)

    grouping = resolve_null_groups(obs, sample_key="sample_id", lineage=lineage, min_cells=25)

    assert grouping.level.notna().all()
    assert (grouping.level.iloc[:10] == "sample").all()
    # And that sample-level null must be estimated over all 60 cells, not just these 10.
    assert int(grouping.keys["sample"].nunique()) == 1


def test_lineage_conditioning_can_be_switched_off() -> None:
    """The absolute scale must remain reachable for cohorts with no usable lineages."""
    adata = _cohort(n_ordinary=300, n_rare=0)
    result, _, lineage = _adjudicate(adata, lineage_conditional=False)

    assert lineage is None
    assert (result.state == str(QCStateInitial.CORE)).mean() > 0.9


@pytest.mark.parametrize("n_rare", [30, 60, 120])
def test_protection_holds_across_rare_population_sizes(n_rare: int) -> None:
    """A population that is rarer must not be more likely to be deleted.

    Rarity itself must never be the thing that condemns a cell, or the tool systematically
    destroys exactly the populations a study is looking for.
    """
    adata = _cohort(n_rare=n_rare)
    result, _, _ = _adjudicate(adata, lineage_conditional=True)

    assert _fraction(result, adata, "rare", QCStateInitial.QUARANTINE) == 0.0
