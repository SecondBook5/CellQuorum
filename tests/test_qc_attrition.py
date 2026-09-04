"""Tests for the QC differential-attrition audit.

QC that removes cells at different rates in the arms of the study is not a
filter, it is a covariate. Nothing in the pipeline used to check for it, so it
was found by hand or not at all -- which on any dataset but the one in front of
you means not at all. These tests pin the checking down.
"""

from __future__ import annotations

# Import Path for the tmp_path fixture annotation.
from pathlib import Path

# Import SimpleNamespace to stand in for cohort and design config blocks.
from types import SimpleNamespace

# Import AnnData for the end-to-end stage test.
import anndata as ad

# Import NumPy for synthetic cohort construction.
import numpy as np

# Import pandas for label and decision series.
import pandas as pd

# Import pytest for parametrisation.
import pytest

# Import SciPy's own Fisher test, used to prove the stratified result differs.
from scipy.stats import fisher_exact

# Import the top-level config for the end-to-end stage test.
from cellquorum.config.models import CellQuorumConfig

# Import pipeline context contracts for the end-to-end stage test.
from cellquorum.core.context import PipelineContext, PipelinePaths

# Import the audit under test.
from cellquorum.stages.qc.attrition import (
    ATTRITION_ALPHA,
    ATTRITION_COLUMNS,
    ATTRITION_MIN_RATE_DIFFERENCE,
    audit_differential_attrition,
    audit_qc_design_leaks,
    audit_qc_stage_attrition,
)

# Import the QC config for the stage-level resolution tests.
from cellquorum.stages.qc.config import QCConfig

# Import the QC stage for the end-to-end test.
from cellquorum.stages.qc.stage import QCStage


def make_cohort(
    *,
    plan: dict[tuple[str, str], tuple[int, float]],
    seed: int = 0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Build a synthetic cohort with a prescribed removal rate per cell group.

    Args:
        plan: Mapping from (donor, condition) to (n_cells, removal_fraction).
        seed: Seed for the (deterministic) removal draw.

    Returns:
        The keep decision, the condition labels, and the donor labels, all
        indexed by the same cell names.
    """

    # Draw reproducibly.
    generator = np.random.default_rng(seed)

    # Accumulate one block of cells per plan entry.
    keep_parts: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    donors: list[np.ndarray] = []
    names: list[str] = []

    # Build each (donor, condition) block.
    for (donor, condition), (n_cells, removal) in plan.items():
        # Remove an exact count rather than a random one, so the rates under test
        # are the rates written in the plan.
        n_removed = int(round(n_cells * removal))
        block_keep = np.ones(n_cells, dtype=bool)
        block_keep[generator.permutation(n_cells)[:n_removed]] = False

        # Record the block.
        keep_parts.append(block_keep)
        conditions.append(np.full(n_cells, condition))
        donors.append(np.full(n_cells, donor))
        names.extend(f"{donor}_{condition}_{position}" for position in range(n_cells))

    # Assemble the aligned series.
    index = pd.Index(names, name="cell")
    keep = pd.Series(np.concatenate(keep_parts), index=index, name="keep")
    condition = pd.Series(np.concatenate(conditions), index=index, name="condition")
    donor = pd.Series(np.concatenate(donors), index=index, name="donor")
    return keep, condition, donor


def make_annotated_cohort(
    *,
    plan: dict[tuple[str, str, str], tuple[int, float]],
    seed: int = 0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Build a synthetic cohort whose removal rate is prescribed per cell type.

    The same construction as :func:`make_cohort` with one more level, which is
    what the subset pass exists for: an imbalance confined to one lineage.

    Args:
        plan: Mapping from (donor, condition, cell_type) to (n_cells, removal).
        seed: Seed for the removal draw.

    Returns:
        The keep decision, condition labels, donor labels and cell-type labels,
        all indexed by the same cell names.
    """

    generator = np.random.default_rng(seed)
    keep_parts: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    donors: list[np.ndarray] = []
    cell_types: list[np.ndarray] = []
    names: list[str] = []

    for (donor, condition, cell_type), (n_cells, removal) in plan.items():
        # Remove an exact count so the rates under test are the planned rates.
        n_removed = int(round(n_cells * removal))
        block_keep = np.ones(n_cells, dtype=bool)
        block_keep[generator.permutation(n_cells)[:n_removed]] = False

        keep_parts.append(block_keep)
        conditions.append(np.full(n_cells, condition))
        donors.append(np.full(n_cells, donor))
        cell_types.append(np.full(n_cells, cell_type))
        names.extend(f"{donor}_{condition}_{cell_type}_{position}" for position in range(n_cells))

    index = pd.Index(names, name="cell")
    return (
        pd.Series(np.concatenate(keep_parts), index=index, name="keep"),
        pd.Series(np.concatenate(conditions), index=index, name="condition"),
        pd.Series(np.concatenate(donors), index=index, name="donor"),
        pd.Series(np.concatenate(cell_types), index=index, name="cell_type"),
    )


def make_one_bad_lineage(
    *,
    n_donors: int = 8,
    bulk_cells: int = 900,
    lineage_cells: int = 100,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Build the case the subset pass exists for, and the cohort pass cannot see.

    One large balanced lineage and one small lineage that loses ten points more of
    its diseased cells. Pooled, the gap is a single point and falls under the
    effect-size gate; within the small lineage it is ten. This is the measured
    shape of the skin atlas, where the mitochondrial ceiling cost mast cells far
    more than any other lineage while the cohort rate barely moved.

    Args:
        n_donors: Donors, paired across both conditions. Must be at least
            :data:`MIN_PAIRED_BLOCKS` for the paired tests to run.
        bulk_cells: Cells per donor per arm in the balanced lineage.
        lineage_cells: Cells per donor per arm in the affected lineage.

    Returns:
        ``(keep, condition, donor, cell_type)``.
    """

    plan: dict[tuple[str, str, str], tuple[int, float]] = {}
    for index in range(n_donors):
        donor = f"D{index + 1}"
        # The bulk lineage: a hair more removal in the diseased arm, enough to
        # produce a p-value but far under the two-point gate.
        plan[(donor, "Normal", "Fibroblast")] = (bulk_cells, 0.02)
        plan[(donor, "Disease", "Fibroblast")] = (bulk_cells, 0.021)
        # The affected lineage: the same direction, ten points wide.
        plan[(donor, "Normal", "Mast")] = (lineage_cells, 0.02)
        plan[(donor, "Disease", "Mast")] = (lineage_cells, 0.12)
    return make_annotated_cohort(plan=plan)


def find_test(audit: object, *, factor: str, unit: str, subset: str | None = None) -> object:
    """
    Pull one record out of an audit by factor, unit of analysis and subset.

    Args:
        audit: Audit result.
        factor: Design factor name.
        unit: Unit of analysis the record must use.
        subset: Subset the record must cover. None selects the whole-cohort
            record, which is the default because it is the pre-specified test.

    Returns:
        The single matching record.
    """

    # Select the matching records.
    matches = [
        record
        for record in audit.tests  # type: ignore[attr-defined]
        if record.factor == factor and record.unit == unit and record.subset == subset
    ]

    # Confirm the selection is unambiguous, then return it.
    assert (
        len(matches) == 1
    ), f"expected one {factor}/{unit} record for subset {subset!r}, got {len(matches)}"
    return matches[0]


def test_balanced_removal_produces_no_warning() -> None:
    """
    Verify equal removal in both arms is reported as a clean result.

    The audit runs on every dataset, so a false alarm is as expensive as a miss:
    an engine that warns about balanced attrition teaches its users to ignore the
    warning.
    """

    # Build two arms with identical removal rates.
    keep, condition, _donor = make_cohort(
        plan={("D1", "Normal"): (1000, 0.1), ("D1", "Disease"): (1000, 0.1)}
    )

    # Audit the condition factor.
    audit = audit_differential_attrition(keep=keep, factors={"condition": condition})

    # Confirm nothing was flagged.
    assert audit.warnings == []

    # Confirm the test still ran and recorded the equal rates.
    record = find_test(audit, factor="condition", unit="cell")
    assert record.p_value > ATTRITION_ALPHA
    assert record.rate_difference == pytest.approx(0.0, abs=1e-9)


def test_condition_biased_removal_is_detected_and_quantified() -> None:
    """
    Verify a real imbalance is caught, with the numbers needed to act on it.

    A warning that says only "attrition differs" sends the reader back to the
    data. The rates, the gap, the direction and the test belong in the message.
    """

    # Build one arm losing four times as many cells as the other.
    keep, condition, _donor = make_cohort(
        plan={("D1", "Normal"): (2000, 0.05), ("D1", "Disease"): (2000, 0.20)}
    )

    # Audit the condition factor.
    audit = audit_differential_attrition(keep=keep, factors={"condition": condition})

    # Confirm the cell-level test found it.
    record = find_test(audit, factor="condition", unit="cell")
    assert record.test == "fisher_exact"
    assert record.p_value < 1e-20
    assert record.rate_difference == pytest.approx(0.15, abs=1e-9)

    # Confirm the per-level rates were recorded against their own labels.
    rates = dict(zip(record.levels, record.removal_rate, strict=True))
    assert rates["Disease"] == pytest.approx(0.20, abs=1e-9)
    assert rates["Normal"] == pytest.approx(0.05, abs=1e-9)

    # Confirm one warning, naming the factor, both rates and the worse arm.
    assert len(audit.warnings) == 1
    message = audit.warnings[0]
    assert "condition" in message
    assert "Disease" in message
    assert "20.0%" in message
    assert "5.0%" in message
    assert "Differential attrition" in message


def test_a_tiny_difference_is_recorded_but_not_warned_about() -> None:
    """
    Verify significance alone does not trigger the warning.

    Cells are counted in the tens of thousands, so a half-point difference in
    removal rate is significant at any alpha while meaning nothing. Gating the
    warning on effect size as well is what keeps it usable on an atlas.
    """

    # Build a large cohort with a half-point difference in removal.
    keep, condition, _donor = make_cohort(
        plan={("D1", "Normal"): (60000, 0.100), ("D1", "Disease"): (60000, 0.105)}
    )

    # Audit the condition factor.
    audit = audit_differential_attrition(keep=keep, factors={"condition": condition})

    # Confirm the difference is real, small, and below the warning threshold.
    record = find_test(audit, factor="condition", unit="cell")
    assert record.p_value < ATTRITION_ALPHA
    assert record.rate_difference < ATTRITION_MIN_RATE_DIFFERENCE

    # Confirm no warning was raised.
    assert audit.warnings == []


def test_the_donor_stratified_test_refuses_a_confounded_pooled_result() -> None:
    """
    Verify blocking on donor is what the cell-level test actually does.

    This cohort is Simpson's paradox as a QC artefact: within every donor the two
    arms lose cells at exactly the same rate, but the donors differ in both
    quality and arm composition, so the POOLED table shows a large, overwhelmingly
    significant difference that no donor exhibits. A pooled test would report the
    dataset as biased; the stratified one reports it as clean. Paired designs are
    the normal case in this pipeline, so the stratified answer is the honest one.
    """

    # Build two donors of very different quality and opposite arm composition.
    keep, condition, donor = make_cohort(
        plan={
            ("D_dirty", "Disease"): (1000, 0.30),
            ("D_dirty", "Normal"): (100, 0.30),
            ("D_clean", "Disease"): (100, 0.05),
            ("D_clean", "Normal"): (1000, 0.05),
        }
    )

    # Confirm the pooled table really is misleading, so the test has teeth.
    removed = ~keep
    table = [
        [int((removed & (condition == level)).sum()), int((keep & (condition == level)).sum())]
        for level in ("Disease", "Normal")
    ]
    _pooled_or, pooled_p = fisher_exact(table)
    assert pooled_p < 1e-20

    # Audit with the donor as the blocking factor.
    audit = audit_differential_attrition(keep=keep, factors={"condition": condition}, block=donor)

    # Confirm the cell-level test stratified on donor and found nothing.
    record = find_test(audit, factor="condition", unit="cell")
    assert record.test == "cochran_mantel_haenszel"
    assert record.n_strata == 2
    assert record.p_value > ATTRITION_ALPHA
    assert record.odds_ratio == pytest.approx(1.0, abs=0.05)

    # Confirm no warning was raised on a dataset that is clean within donor.
    assert audit.warnings == []


def test_the_paired_test_uses_the_donor_as_the_unit_of_analysis() -> None:
    """
    Verify a donor-level paired test is reported alongside the cell-level one.

    Cells within a donor are not independent observations, so a cell-level
    p-value overstates the evidence however it is stratified. When each donor
    contributes both arms, the honest unit is the donor: one removal rate per
    donor per arm, paired. That is the test a reviewer will ask for.
    """

    # Build eight donors, each losing three points more in the disease arm.
    plan: dict[tuple[str, str], tuple[int, float]] = {}
    for index in range(8):
        base = 0.05 + 0.01 * index
        plan[(f"D{index}", "Normal")] = (400, base)
        plan[(f"D{index}", "Disease")] = (400, base + 0.03)
    keep, condition, donor = make_cohort(plan=plan)

    # Audit with the donor as the blocking factor.
    audit = audit_differential_attrition(keep=keep, factors={"condition": condition}, block=donor)

    # Confirm the donor-level record exists and used the paired test.
    record = find_test(audit, factor="condition", unit="donor")
    assert record.test == "wilcoxon_signed_rank"
    assert record.n_strata == 8
    assert record.p_value < ATTRITION_ALPHA

    # Confirm it reports the mean per-donor rate for each arm, not a cell count.
    rates = dict(zip(record.levels, record.removal_rate, strict=True))
    assert rates["Disease"] - rates["Normal"] == pytest.approx(0.03, abs=5e-3)

    # Confirm the warning names the donor-level evidence, which is the claim that
    # survives a reviewer, rather than only the cell-level p-value.
    assert any("donor" in message for message in audit.warnings)


def test_a_three_level_factor_falls_back_to_a_chi_square() -> None:
    """
    Verify factors with more than two levels are still tested.

    Batch, timepoint and site are routinely more than two levels, and attrition
    tracking any of them is the same defect as attrition tracking condition.
    """

    # Build three arms with very different removal rates.
    keep, factor, _donor = make_cohort(
        plan={
            ("D1", "A"): (1000, 0.05),
            ("D1", "B"): (1000, 0.10),
            ("D1", "C"): (1000, 0.30),
        }
    )
    factor = factor.rename("batch")

    # Audit the three-level factor.
    audit = audit_differential_attrition(keep=keep, factors={"batch": factor})

    # Confirm the chi-square was used across all three levels.
    record = find_test(audit, factor="batch", unit="cell")
    assert record.test == "chi_square"
    assert record.levels == ("A", "B", "C")
    assert record.p_value < 1e-20

    # Confirm the gap is measured between the extreme levels.
    assert record.rate_difference == pytest.approx(0.25, abs=1e-9)


@pytest.mark.parametrize(
    ("removal", "expected"),
    [(0.0, "no cell was removed"), (1.0, "every cell was removed")],
)
def test_a_degenerate_decision_is_skipped_with_its_reason(removal: float, expected: str) -> None:
    """
    Verify an all-keep or all-remove decision is skipped rather than tested.

    There is no association to measure when the outcome does not vary, and a
    skipped record with a reason is more use to the reader than an absent one or a
    p-value of 1.0.
    """

    # Build a cohort whose decision is constant.
    keep, condition, _donor = make_cohort(
        plan={("D1", "Normal"): (500, removal), ("D1", "Disease"): (500, removal)}
    )

    # Audit the condition factor.
    audit = audit_differential_attrition(keep=keep, factors={"condition": condition})

    # Confirm the record explains itself and carries no p-value.
    record = find_test(audit, factor="condition", unit="cell")
    assert record.p_value is None
    assert expected in record.skipped
    assert audit.warnings == []


def test_a_single_level_factor_is_skipped() -> None:
    """
    Verify a constant factor is skipped, not treated as an error.

    A per-lineage run has one cell type and often one condition; that is a normal
    configuration, not a broken one.
    """

    # Build a cohort with one condition.
    keep, condition, _donor = make_cohort(plan={("D1", "Normal"): (500, 0.1)})

    # Audit the constant factor.
    audit = audit_differential_attrition(keep=keep, factors={"condition": condition})

    # Confirm the skip reason names the single level.
    record = find_test(audit, factor="condition", unit="cell")
    assert record.p_value is None
    assert "one level" in record.skipped


def test_cells_with_no_label_are_excluded_and_counted() -> None:
    """
    Verify unlabelled cells are dropped from the test and reported.

    Testing them as a level named "nan" would invent a study arm; silently
    dropping them would hide that part of the cohort was never audited.
    """

    # Build a cohort and blank the label on a tenth of the disease arm.
    keep, condition, _donor = make_cohort(
        plan={("D1", "Normal"): (1000, 0.05), ("D1", "Disease"): (1000, 0.20)}
    )
    condition = condition.copy()
    condition.iloc[1000:1100] = np.nan

    # Audit the condition factor.
    audit = audit_differential_attrition(keep=keep, factors={"condition": condition})

    # Confirm only labelled cells were counted.
    record = find_test(audit, factor="condition", unit="cell")
    assert sum(record.n_cells) == 1900
    assert record.levels == ("Disease", "Normal")

    # Confirm the exclusion was reported.
    assert any("100" in message and "no 'condition' label" in message for message in audit.warnings)


def test_factors_resolve_from_the_cohort_block_without_being_named() -> None:
    """
    Verify the audited factors come from the cohort schema, not from a list.

    This is the difference between a check that works on the next dataset and a
    check that works on this one. A config that declares its condition and batch
    keys once has already said everything the audit needs to know.
    """

    # Build a cohort whose disease arm loses cells and whose batches do not.
    keep, condition, donor = make_cohort(
        plan={
            (f"D{index}", arm): (200, 0.05 if arm == "Normal" else 0.20)
            for index in range(8)
            for arm in ("Normal", "Disease")
        }
    )

    # Assemble the obs table with cohort-style column names.
    obs = pd.DataFrame(
        {
            "disease_state": condition,
            "chip": ["chip_a" if position % 2 else "chip_b" for position in range(len(keep))],
            "patient": donor,
        },
        index=keep.index,
    )

    # Declare the keys once, the way a real config does.
    audit = audit_qc_stage_attrition(
        obs=obs,
        keep=keep,
        config=QCConfig(),
        cohort=SimpleNamespace(
            condition_key="disease_state", batch_key="chip", donor_key="patient"
        ),
    )

    # Confirm both declared factors were audited under their own names.
    assert {record.factor for record in audit.tests} == {"disease_state", "chip"}

    # Confirm the donor key became the blocking unit rather than a third factor.
    blocked = find_test(audit, factor="disease_state", unit="patient")
    assert blocked.test == "wilcoxon_signed_rank"

    # Confirm the imbalanced factor was flagged and the balanced one was not.
    assert len(audit.warnings) == 1
    assert "disease_state" in audit.warnings[0]


def test_a_run_with_no_declared_keys_audits_nothing_and_says_nothing() -> None:
    """
    Verify an undeclared cohort produces an empty audit, not an error.

    Plenty of legitimate objects have no design at all -- a single-sample test
    run, a per-lineage subset of one arm. The audit has to be safe to leave on by
    default, which means silent when there is nothing to say.
    """

    # Build a cohort and describe none of it.
    keep, _condition, _donor = make_cohort(plan={("D1", "Normal"): (200, 0.1)})

    # Audit with an empty cohort block.
    audit = audit_qc_stage_attrition(
        obs=pd.DataFrame(index=keep.index),
        keep=keep,
        config=QCConfig(),
        cohort=SimpleNamespace(condition_key=None, batch_key=None, donor_key=None),
    )

    # Confirm nothing was tested and nothing was said.
    assert audit.tests == []
    assert audit.warnings == []
    assert audit.to_dataframe().empty


def test_a_factor_that_is_also_the_block_is_not_stratified_on_itself() -> None:
    """
    Verify a cohort that names one column as both condition and donor still tests.

    Stratifying a factor on itself leaves every stratum with a single level and
    no information, so the audit would silently produce nothing. Single-cell
    configs really do alias these keys -- one sample per condition is a common
    pilot design.
    """

    # Build a cohort where the condition column is the only grouping there is.
    keep, condition, _donor = make_cohort(
        plan={("D1", "Normal"): (500, 0.05), ("D1", "Disease"): (500, 0.25)}
    )
    obs = pd.DataFrame({"group": condition}, index=keep.index)

    # Declare the same column as both the condition and the donor.
    audit = audit_qc_stage_attrition(
        obs=obs,
        keep=keep,
        config=QCConfig(),
        cohort=SimpleNamespace(condition_key="group", batch_key=None, donor_key="group"),
    )

    # Confirm the pooled test ran rather than a degenerate stratified one.
    record = find_test(audit, factor="group", unit="cell")
    assert record.test == "fisher_exact"
    assert record.p_value < 1e-10

    # Confirm no paired record was invented for a factor that cannot pair.
    assert [record.unit for record in audit.tests] == ["cell"]


def test_the_audit_reaches_the_stage_result_and_the_run_directory(tmp_path: Path) -> None:
    """
    Verify a real QC stage run warns about differential attrition and writes it.

    The unit tests prove the statistics; this proves the plumbing, which is where
    a check like this normally dies. The measurement has to happen on the
    UNFILTERED object: under ``mode="filter"`` the stage's output has already lost
    the removed cells, and every arm's attrition measured on it is zero.
    """

    # Build a cohort whose disease arm carries far more mitochondrial reads, so a
    # single fixed mito ceiling removes it preferentially. This is the realistic
    # shape of the problem -- nobody writes a rule that names the condition.
    generator = np.random.default_rng(0)
    n_per_arm = 200
    conditions = np.array(["Normal"] * n_per_arm + ["Disease"] * n_per_arm)
    donors = np.array([f"D{position % 8}" for position in range(2 * n_per_arm)])

    # Give the disease arm a mitochondrial fraction above the ceiling set below,
    # and the normal arm one well beneath it.
    mito_counts = np.where(
        conditions == "Disease",
        generator.integers(200, 400, size=2 * n_per_arm),
        generator.integers(1, 10, size=2 * n_per_arm),
    ).astype(float)

    # Build a two-gene matrix: one mitochondrial, one not, plus filler so the
    # gene-level rules have something to keep.
    other = generator.integers(80, 120, size=(2 * n_per_arm, 3)).astype(float)
    matrix = np.column_stack([mito_counts, other])
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"condition": conditions, "donor_id": donors},
            index=[f"cell_{position}" for position in range(2 * n_per_arm)],
        ),
        var=pd.DataFrame(index=["MT-ND1", "ACTB", "GAPDH", "B2M"]),
    )

    # Configure a fixed mito ceiling and declare the cohort keys.
    config = CellQuorumConfig(
        cohort={"condition_key": "condition", "donor_key": "donor_id"},
        qc=QCConfig(
            mode="filter",
            threshold_strategy="fixed",
            basic={"min_genes_per_cell": 1, "min_cells_per_gene": 1, "max_mito_percent": 20.0},
            mad={"enabled": False},
            outputs={"write_h5ad": False, "write_figures": False},
        ),
    )

    # Run the stage.
    paths = PipelinePaths.from_output_dir(tmp_path / "run")
    paths.ensure_directories()
    result = QCStage().run(
        PipelineContext(
            config=config,
            paths=paths,
            adata=adata,
            run_id="attrition-stage-test",
            random_seed=0,
        )
    )

    # Confirm the stage warned, in the warnings a report actually renders.
    assert any("Differential attrition" in message for message in result.warnings)

    # Confirm the audit survives into provenance, skipped tests included.
    summary = result.metrics["attrition_audit"]
    assert summary["flagged_factors"] == ["condition"]

    # Confirm the table landed in the run directory with its rows intact.
    table = pd.read_csv(tmp_path / "run" / "results" / "qc" / "qc_attrition.csv")
    assert set(table["factor"]) == {"condition"}
    assert set(table["unit"]) == {"cell", "donor_id"}

    # Confirm the rates were measured before filtering. Read off the FILTERED
    # object every rate is zero, which is the bug this test exists to catch.
    cell_row = table[table["unit"] == "cell"].iloc[0]
    assert cell_row["rate_difference"] > 0.5


def test_the_audit_can_be_turned_off() -> None:
    """
    Verify the config switch is honoured.

    On by default is right, but a user re-running a published analysis needs the
    old behaviour available without editing the engine.
    """

    # Build an obviously biased cohort.
    keep, condition, _donor = make_cohort(
        plan={("D1", "Normal"): (500, 0.05), ("D1", "Disease"): (500, 0.30)}
    )

    # Audit with the switch off.
    audit = audit_qc_stage_attrition(
        obs=pd.DataFrame({"condition": condition}, index=keep.index),
        keep=keep,
        config=QCConfig(attrition_audit={"enabled": False}),
        cohort=SimpleNamespace(condition_key="condition", batch_key=None, donor_key=None),
    )

    # Confirm nothing ran.
    assert audit.tests == []
    assert audit.warnings == []


def test_the_table_keeps_its_schema_when_nothing_could_be_tested() -> None:
    """
    Verify the audit table has a fixed schema, so a run can be compared to a run.

    A stage artifact whose columns depend on whether anything was found is not
    something a downstream reader can rely on.
    """

    # Audit with no factors at all.
    keep, _condition, _donor = make_cohort(plan={("D1", "Normal"): (10, 0.1)})
    audit = audit_differential_attrition(keep=keep, factors={})

    # Confirm the table is empty but fully typed, against the declared schema
    # rather than a copy of it: a hand-listed prefix here would have to be edited
    # every time a column is added, which is exactly when the invariant matters.
    table = audit.to_dataframe()
    assert table.empty
    assert list(table.columns) == list(ATTRITION_COLUMNS)
    # The identifying columns lead, because a reader scanning the CSV needs to
    # know which population and which unit a row describes before anything else.
    assert list(ATTRITION_COLUMNS[:4]) == ["factor", "subset", "unit", "test"]


def make_leak_cohort() -> SimpleNamespace:
    """
    Build a cohort block naming the three keys the design-leak guard reads.

    Returns:
        Object exposing condition_key, sample_key and donor_key.
    """

    # Return the declared keys.
    return SimpleNamespace(condition_key="condition", sample_key="sample_id", donor_key="donor_id")


def test_grouping_the_mixture_on_the_condition_is_reported_as_a_leak() -> None:
    """
    Verify fitting the mixture per study arm is refused by name.

    This is the configuration that cannot be repaired downstream: each arm gets
    its own boundary, the arms are made more alike than the data are, and the
    absorbed difference is gone from every later test.
    """

    # Group the mixture on the condition.
    warnings = audit_qc_design_leaks(
        config=QCConfig(
            mad={"mito_metric": None},
            mito_mixture={"enabled": True, "groupby": ["cell_type", "condition"]},
        ),
        cohort=make_leak_cohort(),
    )

    # Confirm exactly one warning, naming the config key and the column.
    assert len(warnings) == 1
    assert "mito_mixture.groupby" in warnings[0]
    assert "'condition'" in warnings[0]
    assert "Design leak in QC" in warnings[0]


def test_grouping_the_mixture_on_a_replicate_is_reported_as_a_leak() -> None:
    """
    Verify sample and donor grouping is refused, including in a fallback level.

    A fallback level is a grouping that gets used, so the defect simply arrives
    later. Both keys are named in one warning per grouping rather than one per
    column, because the fix is the same edit.
    """

    # Name a replicate key at the primary level and another in a fallback.
    warnings = audit_qc_design_leaks(
        config=QCConfig(
            mad={"mito_metric": None},
            mito_mixture={
                "enabled": True,
                "groupby": ["cell_type", "sample_id"],
                "fallback_groupby": [["donor_id"], []],
            },
        ),
        cohort=make_leak_cohort(),
    )

    # Confirm the primary level and the offending fallback were both reported,
    # and the empty pooled fallback was not.
    paths = [
        "mito_mixture.groupby" if "mito_mixture.groupby" in message else "fallback"
        for message in warnings
    ]
    assert paths == ["mito_mixture.groupby", "fallback"]
    assert "'sample_id'" in warnings[0]
    assert "fallback_groupby[0]" in warnings[1]
    assert "'donor_id'" in warnings[1]
    assert "CLEANEST" in warnings[0]


def test_per_sample_mad_is_a_leak_only_while_it_thresholds_mitochondrial_content() -> None:
    """
    Verify the MAD check is scoped to the metric the pathology was measured on.

    Grouping depth-like metrics per sample is standard and defensible: sequencing
    depth genuinely varies by sample. The inversion documented in this engine is
    specific to mitochondrial content, so warning about every per-sample MAD
    grouping would teach a user to ignore the warning.
    """

    # Group MAD per sample with mitochondrial MAD ON.
    with_mito = audit_qc_design_leaks(
        config=QCConfig(mad={"groupby": ["sample_id"]}),
        cohort=make_leak_cohort(),
    )

    # Confirm the mitochondrial case is reported.
    assert len(with_mito) == 1
    assert "mad.groupby" in with_mito[0]

    # Group MAD per sample with mitochondrial MAD off.
    without_mito = audit_qc_design_leaks(
        config=QCConfig(mad={"groupby": ["sample_id"], "mito_metric": None}),
        cohort=make_leak_cohort(),
    )

    # Confirm the depth-only case is left alone.
    assert without_mito == []


def test_the_condition_rule_still_applies_to_mad_without_mitochondrial_grouping() -> None:
    """
    Verify narrowing the MAD check by metric does not exempt the condition.

    The mitochondrial scoping above is about which metric inverts with quality.
    Grouping on the study arm is wrong for any metric, so it must survive the
    narrowing.
    """

    # Group MAD on the condition with mitochondrial MAD off.
    warnings = audit_qc_design_leaks(
        config=QCConfig(mad={"groupby": ["condition"], "mito_metric": None}),
        cohort=make_leak_cohort(),
    )

    # Confirm the arm grouping is still reported.
    assert len(warnings) == 1
    assert "'condition'" in warnings[0]


def test_an_identity_grouping_and_a_disabled_rule_are_silent() -> None:
    """
    Verify the guard says nothing about the configuration the engine recommends.

    A check that fires on the documented best practice is worse than no check.
    """

    # Use the recommended grouping: cell identity and nothing else.
    recommended = audit_qc_design_leaks(
        config=QCConfig(
            mad={"mito_metric": None},
            mito_mixture={"enabled": True, "groupby": ["cell_type"]},
        ),
        cohort=make_leak_cohort(),
    )
    assert recommended == []

    # Confirm a disabled mixture is not read at all, even when misconfigured.
    disabled = audit_qc_design_leaks(
        config=QCConfig(mito_mixture={"enabled": False, "groupby": ["condition"]}),
        cohort=make_leak_cohort(),
    )
    assert disabled == []


def test_the_guard_is_silent_when_no_design_keys_are_declared() -> None:
    """
    Verify an undeclared design cannot produce a leak warning.

    The guard compares grouping columns to declared keys, so with nothing declared
    there is no claim to make -- and inventing one from a column name would fire on
    any dataset whose cell-type column happens to be called ``sample``.
    """

    # Declare no keys at all.
    warnings = audit_qc_design_leaks(
        config=QCConfig(
            mad={"mito_metric": None},
            mito_mixture={"enabled": True, "groupby": ["condition", "sample_id"]},
        ),
    )

    # Confirm nothing was reported.
    assert warnings == []


def test_unused_categorical_levels_cannot_invent_a_level_or_a_stratum() -> None:
    """
    Verify a categorical carrying levels no cell has is audited on what is there.

    Every per-lineage object in this project is sliced out of an atlas, and slicing
    keeps the atlas's full category list on every obs column. Grouped on that
    dtype, pandas emits a row per absent level -- a level with no cells in it --
    and the level and stratum counts are what decide which test runs at all.
    """

    # Build a clean cohort of 8 donors, then re-type its labels as categoricals
    # carrying levels and donors that no cell in this object has.
    keep, condition, donor = make_cohort(
        plan={
            (f"D{index}", arm): (100, 0.10) for index in range(8) for arm in ("Normal", "Disease")
        }
    )
    condition = condition.astype(
        pd.CategoricalDtype(categories=["Normal", "Disease", "Treated", "Recovered"])
    )
    donor = donor.astype(pd.CategoricalDtype(categories=[f"D{index}" for index in range(20)]))

    # Audit with both columns carrying their phantom levels.
    audit = audit_differential_attrition(keep=keep, factors={"condition": condition}, block=donor)

    # Confirm only the two present arms were tested.
    cell_record = find_test(audit, factor="condition", unit="cell")
    assert cell_record.levels == ("Disease", "Normal")

    # Confirm only the eight present donors became strata, in the stratified test
    # and in the paired record alike.
    assert cell_record.n_strata == 8
    assert find_test(audit, factor="condition", unit="donor").n_strata == 8

    # Confirm the balanced cohort still reads as balanced.
    assert audit.warnings == []


def test_an_imbalance_in_one_lineage_is_found_where_the_cohort_test_misses_it() -> None:
    """
    Verify the subset pass catches what pooling averages away.

    This is the whole reason the pass exists. A ten-point gap inside a lineage
    that is a tenth of the object shows up as one point overall, which is under
    the gate the cohort warning is deliberately held to -- so without the subset
    pass the run reports clean and the lineage-level contrast downstream is
    partly a statement about which cells survived.
    """

    keep, condition, donor, cell_type = make_one_bad_lineage()

    audit = audit_differential_attrition(
        keep=keep, factors={"condition": condition}, block=donor, subset=cell_type
    )

    # Confirm the cohort gap really is too small to warn about, so the test below
    # is about the subset pass and not about a cohort warning in disguise.
    cohort = find_test(audit, factor="condition", unit="cell")
    assert cohort.rate_difference < ATTRITION_MIN_RATE_DIFFERENCE

    # Confirm the affected lineage is found, on the reviewer-facing unit.
    mast = find_test(audit, factor="condition", unit="donor", subset="Mast")
    assert mast.rate_difference == pytest.approx(0.10, abs=0.005)
    assert mast.is_significant()

    # Confirm the balanced lineage is not flagged despite its own small p-value:
    # it moved by a tenth of a point, which no methods section can act on.
    fibroblast = find_test(audit, factor="condition", unit="donor", subset="Fibroblast")
    assert fibroblast.rate_difference < ATTRITION_MIN_RATE_DIFFERENCE

    # Confirm exactly one warning fired, and that it names the lineage rather than
    # leaving the reader to find it in the table.
    assert len(audit.warnings) == 1
    assert "within 'Mast'" in audit.warnings[0]
    assert "Fibroblast" not in audit.warnings[0]


def test_cohort_records_come_first_and_carry_no_subset() -> None:
    """
    Verify the pre-specified rows lead the table and are identifiable as such.

    Readers and downstream code both take the first row of a unit as "the" test.
    Interleaving subsets, or leaving the subset column blank on both kinds of row,
    would silently turn a lineage result into the headline number.
    """

    keep, condition, donor, cell_type = make_one_bad_lineage()
    audit = audit_differential_attrition(
        keep=keep, factors={"condition": condition}, block=donor, subset=cell_type
    )

    table = audit.to_dataframe()

    # The first two rows are the cohort pair, in cell-then-donor order.
    assert table["subset"].isna().to_numpy()[:2].all()
    assert list(table["unit"][:2]) == ["cell", "donor"]

    # Everything after them is a subset row, and no subset row is unlabelled.
    assert table["subset"][2:].notna().all()
    assert set(table["subset"][2:]) == {"Fibroblast", "Mast"}


def test_subset_pvalues_are_corrected_and_cohort_pvalues_are_not() -> None:
    """
    Verify the multiplicity correction lands on exactly the rows that need it.

    The cohort test is one pre-specified question and its p-value stands as
    computed. The subset pass asks the same question of every lineage at once, so
    reporting its raw p-values as if each were pre-specified would manufacture an
    alarming lineage per run out of nothing.
    """

    from statsmodels.stats.multitest import multipletests

    keep, condition, donor, cell_type = make_one_bad_lineage()
    audit = audit_differential_attrition(
        keep=keep, factors={"condition": condition}, block=donor, subset=cell_type
    )

    # The cohort rows are uncorrected.
    for unit in ("cell", "donor"):
        assert find_test(audit, factor="condition", unit=unit).p_value_adjusted is None

    # Every tested subset row is corrected, and correction can only raise a
    # p-value.
    subset_records = [record for record in audit.tests if record.subset is not None]
    tested = [record for record in subset_records if record.p_value is not None]
    assert tested, "fixture must produce tested subset rows"
    for record in tested:
        assert record.p_value_adjusted is not None
        assert record.p_value_adjusted >= record.p_value

    # At least one row must actually have moved, or a correction that silently
    # stopped being applied would pass every assertion above.
    assert any(record.p_value_adjusted > record.p_value for record in tested)

    # The correction is applied within each unit of analysis, not across both.
    # Pooling them would let a long tail of cell-level rows raise the threshold
    # the reviewer-facing donor-level rows are judged against.
    for unit in ("cell", "donor"):
        family = [record for record in tested if record.unit == unit]
        expected = multipletests([record.p_value for record in family], method="fdr_bh")[1]
        observed = [record.p_value_adjusted for record in family]
        assert observed == pytest.approx(list(expected))


def test_a_record_is_judged_on_its_adjusted_pvalue() -> None:
    """
    Verify the decision rule reads the corrected value where one exists.

    Carrying an adjusted p-value that nothing acts on would be worse than not
    computing it: the table would say corrected and the warnings would not be.
    """

    from cellquorum.stages.qc.attrition import AttritionTest

    record = AttritionTest(
        factor="condition",
        unit="donor",
        test="wilcoxon_signed_rank",
        levels=("Disease", "Normal"),
        n_cells=(100.0, 100.0),
        n_removed=(12.0, 2.0),
        removal_rate=(0.12, 0.02),
        rate_difference=0.10,
        p_value=0.01,
        p_value_adjusted=0.42,
        subset="Mast",
    )

    assert record.decisive_p_value() == 0.42
    assert not record.is_significant()


def test_a_subset_too_small_to_pair_is_skipped_rather_than_reported() -> None:
    """
    Verify an underpowered lineage says so instead of reporting a null.

    An exact signed-rank test on four pairs cannot reach p<0.05 however consistent
    the effect, so a non-significant row for such a lineage is not evidence of
    balance -- it is evidence of arithmetic. Recording the reason is what stops it
    being read as the former.
    """

    keep, condition, donor, cell_type = make_one_bad_lineage(n_donors=8)

    # Restrict the affected lineage to four donors, leaving it under the pairing
    # floor while the object as a whole stays well powered.
    thinned = cell_type.copy()
    few = {"D1", "D2", "D3", "D4"}
    thinned[(cell_type == "Mast") & (~donor.isin(few))] = "Fibroblast"

    audit = audit_differential_attrition(
        keep=keep, factors={"condition": condition}, block=donor, subset=thinned
    )

    record = find_test(audit, factor="condition", unit="donor", subset="Mast")
    assert record.p_value is None
    assert record.skipped is not None and "4 donor(s) contributed both levels" in record.skipped

    # A skipped row is not a test, so it must not enter the correction family and
    # dilute the rows that are.
    assert record.p_value_adjusted is None


def test_cells_with_no_subset_label_stay_in_the_cohort_test() -> None:
    """
    Verify unlabelled cells are excluded from the subset pass and nowhere else.

    QC runs before annotation as often as after it, and a partly annotated object
    is the normal middle case. Dropping its unlabelled cells from the cohort test
    would silently change the headline number depending on how far annotation got.
    """

    keep, condition, donor, cell_type = make_one_bad_lineage()

    # Strip the label from a third of the object.
    partial = cell_type.copy().astype(object)
    partial.iloc[::3] = np.nan

    audit = audit_differential_attrition(
        keep=keep, factors={"condition": condition}, block=donor, subset=partial
    )

    # The cohort row still counts every cell.
    cohort = find_test(audit, factor="condition", unit="cell")
    assert sum(cohort.n_cells) == len(keep)

    # The subset rows count only labelled cells, and there is no subset invented
    # for the gap.
    subsets = {record.subset for record in audit.tests if record.subset is not None}
    assert subsets == {"Fibroblast", "Mast"}
    subset_cells = sum(
        sum(record.n_cells)
        for record in audit.tests
        if record.subset is not None and record.unit == "cell"
    )
    assert subset_cells == int(partial.notna().sum())


def test_the_stage_resolves_the_cell_type_column_without_being_told() -> None:
    """
    Verify the subset column is found by convention, not by configuration.

    The audit has to work on the next dataset without anyone editing it, which
    means resolving the annotation column the same way the figures and the
    publication tables do. A subset pass that only runs when someone remembers to
    name a column is a subset pass that does not run.
    """

    keep, condition, donor, cell_type = make_one_bad_lineage()

    audit = audit_qc_stage_attrition(
        obs=pd.DataFrame(
            {"condition": condition, "donor_id": donor, "cell_type": cell_type},
            index=keep.index,
        ),
        keep=keep,
        config=QCConfig(),
        cohort=SimpleNamespace(condition_key="condition", batch_key=None, donor_key="donor_id"),
    )

    assert {record.subset for record in audit.tests if record.subset is not None} == {
        "Fibroblast",
        "Mast",
    }
    assert audit.to_summary_dict()["flagged_subsets"] == ["condition:Mast"]
    # The cohort itself is not flagged, so the two lists cannot be conflated.
    assert audit.to_summary_dict()["flagged_factors"] == []


def test_the_subset_pass_can_be_turned_off() -> None:
    """
    Verify the subset pass is a switch, so an old analysis can be reproduced.
    """

    keep, condition, donor, cell_type = make_one_bad_lineage()

    audit = audit_qc_stage_attrition(
        obs=pd.DataFrame(
            {"condition": condition, "donor_id": donor, "cell_type": cell_type},
            index=keep.index,
        ),
        keep=keep,
        config=QCConfig(attrition_audit={"audit_subsets": False}),
        cohort=SimpleNamespace(condition_key="condition", batch_key=None, donor_key="donor_id"),
    )

    assert all(record.subset is None for record in audit.tests)
    assert audit.warnings == []


def test_an_explicitly_named_subset_column_wins() -> None:
    """
    Verify a named column overrides the convention.

    The convention covers the columns this engine writes. A collaborator's object
    labelled in a column of their own has to be auditable without renaming it.
    """

    keep, condition, donor, cell_type = make_one_bad_lineage()

    audit = audit_qc_stage_attrition(
        obs=pd.DataFrame(
            {
                "condition": condition,
                "donor_id": donor,
                # Both present: the convention would pick cell_type, and the
                # config has to beat it.
                "cell_type": "Everything",
                "their_labels": cell_type,
            },
            index=keep.index,
        ),
        keep=keep,
        config=QCConfig(attrition_audit={"subset": "their_labels"}),
        cohort=SimpleNamespace(condition_key="condition", batch_key=None, donor_key="donor_id"),
    )

    assert {record.subset for record in audit.tests if record.subset is not None} == {
        "Fibroblast",
        "Mast",
    }


def test_a_named_subset_column_the_object_lacks_is_not_an_error() -> None:
    """
    Verify a missing subset column degrades to no subset pass.

    A reusable config is run against objects at different stages of annotation.
    Raising here would make the audit the thing that stops a pre-annotation run,
    which is not its job.
    """

    keep, condition, donor, _cell_type = make_one_bad_lineage()

    audit = audit_qc_stage_attrition(
        obs=pd.DataFrame({"condition": condition, "donor_id": donor}, index=keep.index),
        keep=keep,
        config=QCConfig(attrition_audit={"subset": "not_a_column"}),
        cohort=SimpleNamespace(condition_key="condition", batch_key=None, donor_key="donor_id"),
    )

    assert all(record.subset is None for record in audit.tests)


def test_the_subset_column_is_never_also_a_factor_or_the_block() -> None:
    """
    Verify a column cannot be subset and tested, or subset and blocked, at once.

    Subsetting on the factor leaves one level per subset; subsetting on the donor
    leaves one donor per subset. Either way every subset row is a skipped record,
    and a table of skips reads as though the audit ran when it did not.
    """

    keep, condition, donor, _cell_type = make_one_bad_lineage()
    obs = pd.DataFrame({"condition": condition, "donor_id": donor}, index=keep.index)
    cohort = SimpleNamespace(condition_key="condition", batch_key=None, donor_key="donor_id")

    for column in ("condition", "donor_id"):
        audit = audit_qc_stage_attrition(
            obs=obs,
            keep=keep,
            config=QCConfig(attrition_audit={"subset": column}),
            cohort=cohort,
        )
        assert all(record.subset is None for record in audit.tests), column


def make_batched_cohort(*, crossed: bool) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Build a cohort with a technical factor either crossed with or nested in the condition.

    Both variants carry the SAME 8-point attrition gap between the two technical
    levels, because the point under test is not the size of the gap but what the
    gap can do to the comparison the study is making.

    Args:
        crossed: True to give each technical level cells from both conditions, the
            way a real capture run holds cases and controls together. False to make
            each level hold one condition only.

    Returns:
        The keep decision, condition labels, donor labels and technical labels.
    """

    if crossed:
        # Dirty donors and clean donors, each contributing to both arms: the gap
        # lives between capture runs and cancels between conditions.
        plan = {
            (donor, condition): (500, rate)
            for donor, rate in (("P1", 0.10), ("P2", 0.10), ("P3", 0.02), ("P4", 0.02))
            for condition in ("Normal", "Lymphedema")
        }
    else:
        # The same gap, but now it runs along the condition: every dirty cell is a
        # case and every clean cell is a control.
        plan = {
            (donor, condition): (500, 0.10 if condition == "Lymphedema" else 0.02)
            for donor in ("P1", "P2", "P3", "P4")
            for condition in ("Normal", "Lymphedema")
        }

    keep, condition, donor = make_cohort(plan=plan)
    if crossed:
        batch = donor.map({"P1": "run_A", "P2": "run_A", "P3": "run_B", "P4": "run_B"})
    else:
        batch = condition.map({"Normal": "run_A", "Lymphedema": "run_B"})
    return keep, condition, donor, batch.rename("batch")


def find_warning(audit, *, factor: str) -> str:
    """Return the single attrition warning naming this factor."""

    matches = [
        warning
        for warning in audit.warnings
        if warning.startswith(f"Differential attrition by '{factor}'")
    ]
    assert len(matches) == 1, f"expected one '{factor}' warning, got {matches}"
    return matches[0]


def test_a_factor_crossed_with_the_contrast_is_reported_as_capture_quality() -> None:
    """
    Verify a crossed technical factor is not called a confounder of the contrast.

    This is the difference between a warning that means something and one that
    fires on every dataset. Any cohort of more than a few captures has a best and
    a worst capture, and with tens of thousands of cells the gap between them is
    always significant -- so wording that calls it a covariate of the study's
    comparison cries wolf every single run. When each capture holds cases AND
    controls, uneven attrition between captures removes cells from both arms, and
    the question of whether the filter tracked the disease is answered by the
    condition rows, which are computed and reported separately.
    """

    keep, condition, donor, batch = make_batched_cohort(crossed=True)

    audit = audit_differential_attrition(
        keep=keep,
        factors={"condition": condition, "batch": batch},
        block=donor,
        contrast="condition",
    )

    warning = find_warning(audit, factor="batch")
    assert "uneven capture quality" in warning
    assert "All 2 levels of 'batch'" in warning
    assert "covariate" not in warning

    # And the reason it is safe to de-escalate: the contrast itself is clean here.
    assert not any(
        warning.startswith("Differential attrition by 'condition'") for warning in audit.warnings
    )


def test_a_factor_nested_in_the_contrast_is_escalated_and_counts_the_confounding() -> None:
    """
    Verify a nested technical factor keeps the covariate wording and quantifies it.

    A capture run that holds only cases is not a technical detail, it IS the
    contrast wearing a different column name, and its attrition gap cannot be
    separated from the disease effect by any downstream model. Naming how many of
    its levels are pure is what lets a reader tell total confounding from the
    partial kind, which is the more common and more insidious case.
    """

    keep, condition, donor, batch = make_batched_cohort(crossed=False)

    audit = audit_differential_attrition(
        keep=keep,
        factors={"condition": condition, "batch": batch},
        block=donor,
        contrast="condition",
    )

    warning = find_warning(audit, factor="batch")
    assert "2 of 2 levels of 'batch'" in warning
    assert "lie entirely within a single 'condition' level" in warning
    assert "covariate, not a filter" in warning
    assert "capture quality" not in warning


def test_the_contrast_itself_always_keeps_the_covariate_wording() -> None:
    """
    Verify the factor the study contrasts is never de-escalated.

    There is nothing to cross the contrast against: an attrition gap along it is a
    confounder of the comparison by definition, and no crossing test can soften
    that.
    """

    keep, condition, donor, batch = make_batched_cohort(crossed=False)

    audit = audit_differential_attrition(
        keep=keep,
        factors={"condition": condition, "batch": batch},
        block=donor,
        contrast="condition",
    )

    warning = find_warning(audit, factor="condition")
    assert "covariate, not a filter" in warning
    assert "capture quality" not in warning


def test_without_a_declared_contrast_no_factor_is_de_escalated() -> None:
    """
    Verify the unconditional wording survives when nothing is named as the contrast.

    A dataset that declares no condition key gets an audit of whatever factors it
    does declare, and with no comparison named, any of them might be the one that
    matters. Guessing would be worse than the extra caution.
    """

    keep, condition, donor, batch = make_batched_cohort(crossed=True)

    audit = audit_differential_attrition(
        keep=keep,
        factors={"condition": condition, "batch": batch},
        block=donor,
    )

    warning = find_warning(audit, factor="batch")
    assert "covariate, not a filter" in warning
    assert "capture quality" not in warning
