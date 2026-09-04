"""Tests for SccodaMethod differential abundance."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import ast
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.backends.registry import build_default_backend_registry
from cellquorum.backends.sccoda_backend import SCCODA_HELPER, build_sccoda_backend
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.comparative.differential_abundance.aggregation import (
    aggregate_celltype_counts,
)
from cellquorum.stages.comparative.differential_abundance.sccoda_method import SccodaMethod

# Check sccoda_env availability once.
_SCCODA_AVAILABLE = build_sccoda_backend().status().available


@pytest.fixture
def synthetic_adata():
    """
    Build a synthetic cohort with 3 cell types across 6 donors (3 Normal, 3 Disease).

    Cell type 2 is enriched in Disease (more cells in Disease samples).
    """

    np.random.seed(42)

    # Generate counts for each donor and cell type (Poisson-like distribution)
    # Donors: N1, N2, N3 (Normal) and D1, D2, D3 (Disease)
    # Cell types: Type0, Type1, Type2
    # Type2 is enriched in Disease samples

    donor_ids = [
        "N1",
        "N1",
        "N1",
        "N2",
        "N2",
        "N2",
        "N3",
        "N3",
        "N3",
        "D1",
        "D1",
        "D1",
        "D2",
        "D2",
        "D2",
        "D3",
        "D3",
        "D3",
    ]
    conditions = ["Normal"] * 9 + ["Disease"] * 9
    cell_types = ["Type0", "Type1", "Type2"] * 6

    # Normal: roughly [100, 50, 30] cells per type per donor
    # Disease: roughly [60, 90, 30] cells per type per donor (Type1 enriched)
    n_cells_list = []
    for _donor, cond in zip(donor_ids, conditions, strict=False):
        ct = cell_types[len(n_cells_list)]
        if cond == "Normal":
            if ct == "Type0":
                n = np.random.poisson(100)
            elif ct == "Type1":
                n = np.random.poisson(50)
            else:  # Type2
                n = np.random.poisson(30)
        else:  # Disease
            if ct == "Type0":
                n = np.random.poisson(60)
            elif ct == "Type1":
                n = np.random.poisson(90)
            else:  # Type2
                n = np.random.poisson(30)
        n_cells_list.append(n)

    # Build per-cell obs
    obs_rows = []
    for donor, cond, ct, n_cells in zip(
        donor_ids, conditions, cell_types, n_cells_list, strict=False
    ):
        for _ in range(n_cells):
            obs_rows.append({"donor": donor, "condition": cond, "cell_type": ct})

    obs = pd.DataFrame(obs_rows)
    n_obs = len(obs)

    # Dummy X matrix (doesn't matter for DA)
    X = np.zeros((n_obs, 10))

    return ad.AnnData(X=X, obs=obs)


@pytest.fixture
def mock_context(tmp_path):
    """Build a mock stage context with paths and backend registry."""

    class Paths:
        scratch = tmp_path / "scratch"
        results = tmp_path / "results"
        figures = tmp_path / "figures"

    class Context:
        paths = Paths()
        backend_registry = build_default_backend_registry()

    return Context()


@pytest.fixture
def paired_adata():
    """Build a donor-paired cohort: 6 donors, each contributing both conditions.

    Distinct from ``synthetic_adata``, where donors are nested within condition.
    Pairing is a property of the cohort rather than a setting, so both shapes need
    covering: this one must be fitted within donor, and that one must not be.
    Type1 rises in Disease in every donor, so the donor-level audit has a known
    answer.
    """

    rng = np.random.default_rng(7)
    rows = []
    for index in range(6):
        donor = f"P{index}"
        for condition, base in (("Normal", (100, 50, 30)), ("Disease", (95, 80, 30))):
            for cell_type, mean in zip(("Type0", "Type1", "Type2"), base, strict=True):
                for _ in range(int(rng.poisson(mean))):
                    rows.append({"donor": donor, "condition": condition, "cell_type": cell_type})

    obs = pd.DataFrame(rows)
    return ad.AnnData(X=np.zeros((len(obs), 10)), obs=obs)


@pytest.mark.skipif(not _SCCODA_AVAILABLE, reason="sccoda_env not available")
def test_sccoda_happy_path_auto_only(synthetic_adata, mock_context):
    """With engine reference selection disabled, only scCODA's own auto fit runs.

    This is the escape hatch that reproduces a pre-existing table: the engine
    normally picks the reference itself, so auto-only has to be asked for.
    """

    method = SccodaMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "Disease",
        "control": "Normal",
        "seed": 0,
        "num_iterations": 2000,  # Fast for testing
        "select_reference": False,
    }

    result = method.run(synthetic_adata, config, mock_context)

    # Should not be a skip
    assert not isinstance(result, MethodSkip)

    # The DA table is present and remains the primary artifact.
    assert len(result.artifacts) >= 1
    artifact = result.artifacts[0]
    assert artifact.name == "da_results"
    assert artifact.path.name == "da_sccoda.csv"
    assert artifact.kind == "csv"

    # Read the CSV and verify structure
    df = pd.read_csv(artifact.path)
    assert set(df.columns) == {
        "cell_type",
        "log2_fold_change",
        "inclusion_probability",
        "credible_effect",
        "reference",
    }

    # Should contain "auto" reference
    assert "auto" in df["reference"].values

    # Should have 3 rows (3 cell types, auto reference only)
    assert len(df) == 3

    # Metrics should be populated
    assert result.metrics["case"] == "Disease"
    assert result.metrics["control"] == "Normal"
    assert result.metrics["n_samples"] == 6
    assert result.metrics["n_celltypes"] == 3
    assert result.metrics["reference_source"] == "sccoda_automatic"

    # The sampler diagnostics are what separate a null result from a dead chain, so
    # they must survive the trip back from the helper.
    assert 0.0 < result.metrics["sccoda_acceptance_rate_min"] <= 1.0
    assert result.metrics["sccoda_formula"] == 'Q("condition")'


@pytest.mark.skipif(not _SCCODA_AVAILABLE, reason="sccoda_env not available")
def test_sccoda_engine_picks_the_reference_by_default(synthetic_adata, mock_context):
    """The engine chooses the reference and records the criterion behind it.

    Type2 is held at 30 cells per sample in both arms while Type0 and Type1 trade
    places, so it is the only stable denominator in this cohort and the selector has
    a knowable right answer.
    """

    result = SccodaMethod().run(
        synthetic_adata,
        {
            "cell_type_col": "cell_type",
            "condition_col": "condition",
            "donor_col": "donor",
            "case": "Disease",
            "control": "Normal",
            "seed": 0,
            "num_iterations": 2000,
        },
        mock_context,
    )

    assert not isinstance(result, MethodSkip)
    assert result.metrics["reference_source"] == "engine"
    assert result.metrics["reference_celltype"] == "Type2"
    assert result.metrics["reference_relaxed"] is False

    df = pd.read_csv(result.artifacts[0].path)
    assert set(df["reference"]) == {"auto", "Type2"}

    # The criterion table is emitted so a reader can see what was rejected.
    criterion = {a.name: a for a in result.artifacts}["da_reference_criterion"]
    table = pd.read_csv(criterion.path).set_index("cell_type")
    assert bool(table.loc["Type2", "selected"])
    assert table.loc["Type2", "clr_variance"] == table["clr_variance"].min()

    assert any("steadiest share" in note for note in result.notes)


@pytest.mark.skipif(not _SCCODA_AVAILABLE, reason="sccoda_env not available")
def test_sccoda_dual_reference(synthetic_adata, mock_context):
    """Run scCODA with explicit reference and verify both auto and explicit appear."""

    method = SccodaMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "Disease",
        "control": "Normal",
        "reference_celltype": "Type0",
        "seed": 0,
        "num_iterations": 2000,
    }

    result = method.run(synthetic_adata, config, mock_context)

    assert not isinstance(result, MethodSkip)

    df = pd.read_csv(result.artifacts[0].path)

    # Should contain BOTH "auto" and "Type0" in reference column
    references = set(df["reference"].values)
    assert "auto" in references
    assert "Type0" in references

    # Should have two sets of results (one per reference)
    assert len(df) == 6  # 3 cell types × 2 references


@pytest.mark.skipif(not _SCCODA_AVAILABLE, reason="sccoda_env not available")
def test_sccoda_declines_pairing_when_donors_are_nested(synthetic_adata, mock_context):
    """A donor term is refused when no donor spans both arms.

    In this cohort each donor appears in one condition only, so donor is collinear
    with condition: adding it would remove the contrast of interest rather than the
    donor effect. The refusal has to be recorded, not silent.
    """

    result = SccodaMethod().run(
        synthetic_adata,
        {
            "cell_type_col": "cell_type",
            "condition_col": "condition",
            "donor_col": "donor",
            "case": "Disease",
            "control": "Normal",
            "seed": 0,
            "num_iterations": 2000,
            # Even when pairing is demanded, the design cannot support it.
            "pair_by_donor": "always",
        },
        mock_context,
    )

    assert not isinstance(result, MethodSkip)
    assert result.metrics["paired_by_donor"] is False
    assert result.metrics["n_paired_donors"] == 0
    assert any("collinear with condition" in note for note in result.notes)
    assert result.metrics["sccoda_formula"] == 'Q("condition")'

    # With no donor pairs there is nothing for the concordance audit to say.
    assert result.metrics["n_donor_consistent"] is None


@pytest.mark.skipif(not _SCCODA_AVAILABLE, reason="sccoda_env not available")
def test_sccoda_pairs_by_donor_and_audits_donor_concordance(paired_adata, mock_context):
    """A matched cohort is fitted within donor and its calls are audited per donor."""

    result = SccodaMethod().run(
        paired_adata,
        {
            "cell_type_col": "cell_type",
            "condition_col": "condition",
            "donor_col": "donor",
            "case": "Disease",
            "control": "Normal",
            "seed": 0,
            "num_iterations": 2000,
        },
        mock_context,
    )

    assert not isinstance(result, MethodSkip)
    assert result.metrics["paired_by_donor"] is True
    assert result.metrics["n_paired_donors"] == 6

    # Donor must enter as a category, not as a number. Donor ids are frequently
    # integers, and patsy would otherwise fit a linear trend across donor index.
    assert result.metrics["sccoda_formula"] == 'Q("condition") + C(Q("donor"))'
    assert any("donor-paired" in note for note in result.notes)

    # The concordance audit runs and its columns are merged into the DA table.
    concordance = {a.name: a for a in result.artifacts}["da_donor_concordance"]
    audit = pd.read_csv(concordance.path).set_index("cell_type")
    assert set(audit.index) == {"Type0", "Type1", "Type2"}
    assert audit.loc["Type1", "n_pairs"] == 6
    assert audit.loc["Type1", "direction"] == 1

    df = pd.read_csv(result.artifacts[0].path)
    assert {"pattern", "n_agree", "sign_test_p"} <= set(df.columns)
    assert result.metrics["n_donor_consistent"] >= 1


def test_a_declared_unpaired_design_is_not_overridden_by_the_data():
    """``design.paired: false`` is an instruction, and under ``auto`` it wins.

    Inferring pairing from the data is the right default when nothing is declared, and
    the wrong answer when something is: a cohort whose donors happen to span both arms
    would otherwise be donor-modelled against instruction, and scCODA's fit would then
    disagree with every other method in the same run about what design was tested.
    """
    resolve = SccodaMethod._resolve_pairing

    # Nothing declared: the data-driven rule stands.
    assert resolve("auto", 9, declared=None)[0] is True
    # Declared unpaired: pairing is off, and the reason names both inputs.
    off, reason = resolve("auto", 9, declared=False)
    assert off is False
    assert "paired=false" in reason and "auto" in reason
    # An explicit switch still beats the declaration in both directions.
    assert resolve("always", 9, declared=False)[0] is True
    assert resolve("never", 9, declared=True)[0] is False

    # The override is one-directional: declaring a cohort matched does not create the
    # degrees of freedom to model it below the house floor.
    assert resolve("auto", 2, declared=True)[0] is False
    assert resolve("always", 2, declared=True)[0] is True


@pytest.mark.skipif(not _SCCODA_AVAILABLE, reason="sccoda_env not available")
def test_sccoda_determinism(synthetic_adata, mock_context, tmp_path):
    """Run scCODA twice with the same seed and verify the posterior is identical.

    Asserting only ``credible_effect`` is too weak to protect this. When the fit
    was genuinely non-reproducible, two runs on a byte-identical input returned
    inclusion probabilities of 0.729 vs 0.875, and the boolean still matched --
    so the boolean-only assertion passed on roughly three runs in five and this
    test read as a flake instead of the defect it was. The posterior is the thing
    that has to be reproducible; the classification is downstream of it.
    """

    method = SccodaMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "Disease",
        "control": "Normal",
        "seed": 0,
        "num_iterations": 2000,
    }

    # Run 1
    context1 = mock_context
    result1 = method.run(synthetic_adata, config, context1)
    assert not isinstance(result1, MethodSkip)
    df1 = pd.read_csv(result1.artifacts[0].path)

    # Run 2 (fresh context with different paths)
    class Paths:
        scratch = tmp_path / "run2_scratch"
        results = tmp_path / "run2_results"
        figures = tmp_path / "run2_figures"

    class Context:
        paths = Paths()
        backend_registry = build_default_backend_registry()

    context2 = Context()
    result2 = method.run(synthetic_adata, config, context2)
    assert not isinstance(result2, MethodSkip)
    df2 = pd.read_csv(result2.artifacts[0].path)

    # Same seed, same input, same rows -- in the same order.
    assert df1["cell_type"].tolist() == df2["cell_type"].tolist()

    # The posterior itself, not just its thresholded summary. Exact equality is the
    # right bar: the fit is single-threaded with op determinism on, so the reduction
    # order is fixed and there is no legitimate source of last-bit drift left.
    for column in ("inclusion_probability", "log2_fold_change"):
        pd.testing.assert_series_equal(df1[column], df2[column], check_exact=True)

    assert df1["credible_effect"].tolist() == df2["credible_effect"].tolist()


def test_sccoda_helper_pins_single_threaded_execution():
    """The fit helper must pin both TF thread pools and the BLAS/OpenMP thread counts.

    This is asserted against the *source* rather than by running the helper, for two
    reasons. The helper only imports tensorflow inside ``sccoda_env``, so this test
    process cannot execute the code path; and the pinning has to happen before any op
    is constructed, so there is no later moment at which it could be observed from
    outside anyway.

    It is worth guarding because of how the defect presented. Dropping these lines
    does not break anything visibly -- it reintroduces a fit that returns a different
    posterior on maybe two runs in five, which reads as a flaky test rather than as
    irreproducible science. The AST walk (rather than a text search) is deliberate:
    commenting the calls out is exactly the regression to catch, and a text search
    would still find them in the comment.

    The environment variables must be *assigned*, not ``setdefault``-ed. A machine
    that already exports OMP_NUM_THREADS would silently keep its own value and lose
    the pin, so yielding to the ambient environment is itself the bug.
    """

    tree = ast.parse(SCCODA_HELPER.read_text())

    pinned_pools = set()
    assigned_env = set()
    deferred_env = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = ast.unparse(node.func)
            if target.endswith(
                ("set_inter_op_parallelism_threads", "set_intra_op_parallelism_threads")
            ):
                assert (
                    node.args and ast.literal_eval(node.args[0]) == 1
                ), f"{target} must be pinned to 1 thread, got {ast.unparse(node)}"
                pinned_pools.add(target.rsplit(".", 1)[-1])
            elif target == "os.environ.setdefault" and len(node.args) == 2:
                deferred_env.add(ast.literal_eval(node.args[0]))
        elif isinstance(node, ast.Assign):
            for goal in node.targets:
                if (
                    isinstance(goal, ast.Subscript)
                    and ast.unparse(goal.value) == "os.environ"
                    and isinstance(goal.slice, ast.Constant)
                ):
                    assigned_env.add(goal.slice.value)

    assert pinned_pools == {
        "set_inter_op_parallelism_threads",
        "set_intra_op_parallelism_threads",
    }, f"both TF thread pools must be pinned in {SCCODA_HELPER.name}, found {pinned_pools}"

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_DETERMINISTIC_OPS"):
        assert variable not in deferred_env, (
            f"{SCCODA_HELPER.name} sets {variable} with setdefault, which yields to the "
            f"calling environment and loses the pin; assign it instead"
        )
        assert (
            variable in assigned_env
        ), f"{SCCODA_HELPER.name} must assign {variable} before the TF import"


def test_sccoda_skip_missing_cell_type_col(synthetic_adata, mock_context):
    """SccodaMethod should skip when cell_type_col is missing."""

    method = SccodaMethod()
    config = {
        "cell_type_col": "nonexistent_column",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "Disease",
        "control": "Normal",
    }

    result = method.run(synthetic_adata, config, mock_context)

    assert isinstance(result, MethodSkip)
    assert "cell_type_col" in result.reason or "nonexistent_column" in result.reason


def test_sccoda_skip_missing_case_control(synthetic_adata, mock_context):
    """SccodaMethod should skip when case or control is unset."""

    method = SccodaMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        # No case/control
    }

    result = method.run(synthetic_adata, config, mock_context)

    assert isinstance(result, MethodSkip)
    assert "case" in result.reason or "control" in result.reason


def _sccoda_effects_auto() -> pd.DataFrame:
    """A single-reference (auto) scCODA effects table over the synthetic cell types."""
    return pd.DataFrame(
        {
            "cell_type": ["Type0", "Type1", "Type2"],
            "log2_fold_change": [-0.8, 1.1, 0.05],
            "inclusion_probability": [0.82, 0.9, 0.2],
            "credible_effect": [True, True, False],
            "reference": ["auto", "auto", "auto"],
        }
    )


def test_sccoda_composition_helper_emits_figure(synthetic_adata, mock_context):
    """The composition helper renders the two-panel figure to disk (no sccoda_env)."""

    cc = aggregate_celltype_counts(
        synthetic_adata,
        donor_col="donor",
        condition_col="condition",
        cell_type_col="cell_type",
    )
    artifacts = SccodaMethod()._composition_artifacts(
        _sccoda_effects_auto(),
        cc,
        condition_col="condition",
        donor_col="donor",
        case="Disease",
        control="Normal",
        config={},
        context=mock_context,
    )

    assert artifacts, "expected an scCODA composition figure artifact"
    assert all(a.kind == "figure" and a.name == "da_sccoda_composition" for a in artifacts)
    suffixes = set()
    for a in artifacts:
        assert Path(a.path).exists()
        suffixes.add(Path(a.path).suffix)
    # save_figure writes dual PDF + PNG.
    assert suffixes == {".pdf", ".png"}


def test_sccoda_composition_helper_respects_disable_flag(synthetic_adata, mock_context):
    """Setting write_da_figure=False suppresses the composition figure."""

    cc = aggregate_celltype_counts(
        synthetic_adata,
        donor_col="donor",
        condition_col="condition",
        cell_type_col="cell_type",
    )
    artifacts = SccodaMethod()._composition_artifacts(
        _sccoda_effects_auto(),
        cc,
        condition_col="condition",
        donor_col="donor",
        case="Disease",
        control="Normal",
        config={"write_da_figure": False},
        context=mock_context,
    )
    assert artifacts == []


def _two_fits(primary_credible: list[str], alternate_credible: list[str]) -> tuple:
    """Build the two fit blocks with stated credible sets."""

    def block(reference: str, credible: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cell_type": ["A", "B", "C"],
                "credible_effect": [ct in credible for ct in ["A", "B", "C"]],
                "reference": reference,
            }
        )

    return block("B", primary_credible), block("auto", alternate_credible)


def test_reference_sensitivity_reports_an_unchanged_credible_set():
    """The second fit is already paid for; whether it agrees is the robustness claim."""

    primary, alternate = _two_fits(["A", "C"], ["C", "A"])

    metrics = SccodaMethod._reference_sensitivity(primary, alternate)

    assert metrics["n_credible_alternate_reference"] == 2
    assert metrics["credible_set_reference_stable"] is True
    assert metrics["credible_set_reference_disagreement"] is None


def test_reference_sensitivity_names_the_cell_types_that_disagree():
    """A credible set that depends on the denominator is a weaker claim, and says so."""

    primary, alternate = _two_fits(["A", "C"], ["A"])

    metrics = SccodaMethod._reference_sensitivity(primary, alternate)

    assert metrics["credible_set_reference_stable"] is False
    assert metrics["credible_set_reference_disagreement"] == "C"


def test_reference_sensitivity_keeps_its_schema_when_only_one_fit_ran():
    """Metric keys must not appear and disappear between runs."""

    primary, _ = _two_fits(["A"], [])

    metrics = SccodaMethod._reference_sensitivity(primary, primary.iloc[0:0])

    assert set(metrics) == {
        "n_credible_alternate_reference",
        "credible_set_reference_stable",
        "credible_set_reference_disagreement",
    }
    assert all(value is None for value in metrics.values())


def test_restack_marks_the_reported_fit_only_when_there_are_two():
    """The marker exists to disambiguate; with one fit there is nothing to disambiguate."""

    primary, alternate = _two_fits(["A"], ["A"])

    stacked = SccodaMethod._restack(primary, alternate)
    assert list(stacked["is_primary"]) == [True] * 3 + [False] * 3

    alone = SccodaMethod._restack(primary, primary.iloc[0:0])
    assert "is_primary" not in alone.columns
