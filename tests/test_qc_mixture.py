"""Tests for CellQuorum mitochondrial mixture-model QC filtering."""

from __future__ import annotations

# Import Path for run-directory assertions.
from pathlib import Path

# Import AnnData to drive a real stage run.
import anndata as ad

# Import NumPy for synthetic data generation and numeric assertions.
import numpy as np

# Import pandas for metric table construction.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import a stable log-sum-exp, to recompute a posterior from a recorded model.
from scipy.special import logsumexp

# Import the top-level config and pipeline context, to run the stage end to end.
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.context import PipelineContext, PipelinePaths

# Import QC configuration models used by the mixture policy.
from cellquorum.stages.qc.config import QCConfig, QCMitoMixtureConfig

# Import the decision layer, to confirm the policy reaches a keep flag.
# Import the mixture model under test.
from cellquorum.stages.qc.mixture import (
    MIQC_POSTERIOR_COLUMN,
    PROJECTION_WARN_DISAGREEMENT,
    QCMixtureError,
    fit_mito_mixture,
)

# Import the QC stage, to confirm the fitted model reaches the run directory.
from cellquorum.stages.qc.stage import QCStage

# Import the threshold layer, to confirm the model is expressed as thresholds.


def make_mixture_metrics(
    *,
    n_intact: int = 800,
    n_compromised: int = 200,
    intact_intercept: float = 2.0,
    intact_slope: float = 0.0,
    compromised_intercept: float = 18.0,
    compromised_slope: float = -0.002,
    noise: float = 0.4,
    seed: int = 0,
    cell_type: str = "Fibroblasts",
) -> pd.DataFrame:
    """
    Build a metric table containing a known two-component mitochondrial mixture.

    Intact cells sit on a flat low line and compromised cells on a high line that
    falls with library complexity, which is the relationship miQC models: as a
    membrane-compromised cell loses cytoplasmic RNA, its detected-gene count drops
    and its mitochondrial share rises.

    Args:
        n_intact: Number of intact cells.
        n_compromised: Number of compromised cells.
        intact_intercept: Mitochondrial percentage of an intact cell at zero genes.
        intact_slope: Complexity slope of the intact population.
        compromised_intercept: Mitochondrial percentage of a compromised cell at
            zero genes.
        compromised_slope: Complexity slope of the compromised population.
        noise: Standard deviation of the mitochondrial residual.
        seed: Seed for the synthetic draw.
        cell_type: Cell-type label written to every row.

    Returns:
        Cell-level QC metric table with a truth column named ``is_compromised``.
    """

    # Draw reproducibly.
    generator = np.random.default_rng(seed)

    # Draw library complexity for both populations, giving compromised cells the
    # shallower range they have in real data.
    intact_complexity = generator.uniform(1500, 4000, n_intact)
    compromised_complexity = generator.uniform(400, 2500, n_compromised)

    # Place each population on its own regression line.
    intact_mito = (
        intact_intercept + intact_slope * intact_complexity + generator.normal(0.0, noise, n_intact)
    )
    compromised_mito = (
        compromised_intercept
        + compromised_slope * compromised_complexity
        + generator.normal(0.0, noise, n_compromised)
    )

    # Assemble the table with the columns the model and the QC layers require.
    complexity = np.concatenate([intact_complexity, compromised_complexity])
    return pd.DataFrame(
        {
            "pct_counts_mito": np.clip(np.concatenate([intact_mito, compromised_mito]), 0.0, 100.0),
            "n_genes_by_counts": complexity,
            "total_counts": complexity * 3.0,
            "log1p_total_counts": np.log1p(complexity * 3.0),
            "log1p_n_genes_by_counts": np.log1p(complexity),
            "sample_id": "S1",
            "cell_type": cell_type,
            "is_compromised": np.concatenate(
                [np.zeros(n_intact, dtype=bool), np.ones(n_compromised, dtype=bool)]
            ),
        },
        index=[f"cell_{position}" for position in range(n_intact + n_compromised)],
    )


def make_gene_metrics() -> pd.DataFrame:
    """
    Build a minimal gene-level metric table for threshold construction.

    Returns:
        Gene-level QC metric table.
    """

    # Return deterministic gene-level metric values.
    return pd.DataFrame(
        {"n_cells_by_counts": [1.0, 5.0, 20.0], "total_counts": [1.0, 50.0, 500.0]},
        index=["gene_1", "gene_2", "gene_3"],
    )


def test_fit_recovers_the_two_known_components() -> None:
    """
    Verify the fit recovers the component structure it was given.

    The compromised component must be the high-intercept one and must fall with
    complexity, because that ordering is what identifies it as compromised.
    """

    # Fit the model on data with a known structure.
    result = fit_mito_mixture(make_mixture_metrics(), QCMitoMixtureConfig(enabled=True))

    # Confirm exactly one pooled model was fit and it converged.
    assert len(result.models) == 1
    model = result.models[0]
    assert model.converged is True
    assert model.fallback is None

    # Confirm the compromised component recovered its high intercept.
    assert model.compromised_intercept == pytest.approx(18.0, abs=1.0)

    # Confirm the intact component recovered its low intercept.
    assert model.intact_intercept == pytest.approx(2.0, abs=1.0)

    # Confirm the compromised component falls with complexity, which is the
    # relationship that identifies mitochondrial leakage rather than lineage.
    assert model.compromised_slope < 0

    # Confirm the recovered mixing weight is close to the injected 20%.
    assert model.compromised_weight == pytest.approx(0.2, abs=0.05)


def test_the_recorded_model_reproduces_its_own_posterior() -> None:
    """
    Verify the model record is complete enough to recompute what it decided.

    A record that names the two component lines but not their residual variances
    cannot reproduce its own posterior, because the variances set how far a cell
    has to sit from a line before it stops belonging to it. Reproducing the
    posterior from the recorded numbers alone is the test that the run reported
    its reasoning rather than only its verdict.
    """

    # Fit with both post-processing rules off, so the published posterior is the
    # model's own belief and nothing else.
    metrics = make_mixture_metrics()
    result = fit_mito_mixture(
        metrics,
        QCMitoMixtureConfig(enabled=True, keep_all_below_boundary=False, enforce_left_cutoff=False),
    )
    model = result.models[0]

    # Confirm both residual variances were recorded as positive numbers.
    assert model.compromised_variance > 0.0
    assert model.intact_variance > 0.0

    # Recompute the mixture log-density from the record alone.
    mito = metrics["pct_counts_mito"].to_numpy()
    complexity = metrics["n_genes_by_counts"].to_numpy()
    log_density = np.column_stack(
        [
            np.log(1.0 - model.compromised_weight)
            - 0.5 * np.log(2.0 * np.pi * model.intact_variance)
            - (mito - model.intact_intercept - model.intact_slope * complexity) ** 2
            / (2.0 * model.intact_variance),
            np.log(model.compromised_weight)
            - 0.5 * np.log(2.0 * np.pi * model.compromised_variance)
            - (mito - model.compromised_intercept - model.compromised_slope * complexity) ** 2
            / (2.0 * model.compromised_variance),
        ]
    )

    # Confirm the reproduced posterior matches the published one.
    reproduced = np.exp(log_density[:, 1] - logsumexp(log_density, axis=1))
    np.testing.assert_allclose(result.posterior.to_numpy(), reproduced, atol=1e-6)


def test_compromised_cells_are_flagged_and_intact_cells_are_not() -> None:
    """
    Verify the policy separates the injected populations.

    The two lines are far apart, so a correct model should recover nearly all of
    the compromised cells without sacrificing intact ones.
    """

    # Fit the model and read the calls.
    metrics = make_mixture_metrics()
    result = fit_mito_mixture(metrics, QCMitoMixtureConfig(enabled=True))
    flagged = result.probabilities > 0.75

    # Confirm nearly every injected compromised cell was recovered.
    truth = metrics["is_compromised"]
    assert flagged[truth].mean() > 0.9

    # Confirm intact cells were almost entirely spared.
    assert flagged[~truth].mean() < 0.02


def test_restart_zero_makes_the_usual_case_seed_independent() -> None:
    """
    Verify a single-restart fit does not depend on the seed.

    Restart 0 is initialised deterministically from an ordinary least-squares
    residual split, so a run that uses only that restart must be reproducible
    regardless of ``random_state``. A QC boundary that moved when an unrelated
    seed changed would be indefensible.
    """

    # Fit twice with different seeds and a single restart.
    metrics = make_mixture_metrics()
    first = fit_mito_mixture(
        metrics, QCMitoMixtureConfig(enabled=True, n_restarts=1, random_state=0)
    )
    second = fit_mito_mixture(
        metrics, QCMitoMixtureConfig(enabled=True, n_restarts=1, random_state=99)
    )

    # Confirm the per-cell probabilities are identical.
    pd.testing.assert_series_equal(first.probabilities, second.probabilities)


def test_group_below_min_cells_is_kept_and_reported() -> None:
    """
    Verify a group too small to fit is kept rather than filtered.

    Deleting cells because a model could not be estimated would make a numerical
    limitation look like a biological finding, so the fail-safe direction is to
    keep and to say so.
    """

    # Fit with a min_cells above the group size.
    metrics = make_mixture_metrics(n_intact=40, n_compromised=10)
    result = fit_mito_mixture(metrics, QCMitoMixtureConfig(enabled=True, min_cells=100))

    # Confirm every cell was kept.
    assert (result.probabilities == 0.0).all()

    # Confirm the deferral was recorded against the group.
    assert len(result.models) == 1
    assert result.models[0].fallback is not None
    assert "min_cells" in result.models[0].fallback

    # Confirm the unfiltered cells were reported rather than passed over.
    assert any("received no mitochondrial mixture model" in w for w in result.warnings)


def test_fallback_grouping_scores_a_group_too_small_to_fit_alone() -> None:
    """
    Verify a rare cell type borrows a coarser model instead of going unfiltered.

    The fallback model is estimated on the wider group's cells but applied only to
    the cells still waiting, so strength is borrowed without the finer groups
    being re-decided. This is ``level_policy='per_group'``, which has to be asked
    for: it leaves cells judged at two different levels, and the default refuses
    that (see ``test_one_unfittable_group_moves_the_whole_dataset_to_the_next_level``).
    """

    # Build one large cell type and one too small to fit on its own.
    large = make_mixture_metrics(seed=0, cell_type="Fibroblasts")
    small = make_mixture_metrics(n_intact=40, n_compromised=10, seed=1, cell_type="Neutrophils")
    small.index = [f"rare_{position}" for position in range(len(small))]
    metrics = pd.concat([large, small])

    # Fit with a pooled fallback behind the per-cell-type grouping.
    result = fit_mito_mixture(
        metrics,
        QCMitoMixtureConfig(
            enabled=True,
            groupby=["cell_type"],
            fallback_groupby=[[]],
            min_cells=100,
            level_policy="per_group",
        ),
    )

    # Confirm no cell was left without a model.
    assert not any("received no mitochondrial mixture model" in w for w in result.warnings)

    # Confirm a level-1 fallback model scored exactly the rare cells.
    fallback_models = [model for model in result.models if model.level == 1]
    assert len(fallback_models) == 1
    assert fallback_models[0].n_assigned == len(small)

    # Confirm the fallback model was estimated on more cells than it scored.
    assert fallback_models[0].n_cells == len(metrics)

    # Confirm the rare cell type's compromised cells were still recovered.
    flagged = result.probabilities.loc[small.index] > 0.75
    assert flagged[small["is_compromised"].to_numpy()].mean() > 0.8


def make_two_groups_one_too_small(*, flatten_second: bool = False) -> pd.DataFrame:
    """
    Build two fitting groups where the second one cannot be fit on its own.

    Args:
        flatten_second: When True the second group is large enough but carries a
            constant mitochondrial fraction, so it fails in the FIT rather than
            the size screen. Both routes must resolve the same way.

    Returns:
        Cell-level metric table carrying a ``sample_id`` with two values.
    """

    # Build one group large and clean enough to fit on its own.
    large = make_mixture_metrics(seed=0)
    large["sample_id"] = "S_large"

    # Build the second group either too small to fit, or unfittable in substance.
    if flatten_second:
        second = make_mixture_metrics(seed=1)
        second["pct_counts_mito"] = 3.0
    else:
        second = make_mixture_metrics(n_intact=40, n_compromised=10, seed=1)
    second["sample_id"] = "S_small"
    second.index = [f"second_{position}" for position in range(len(second))]

    # Return the concatenated table.
    return pd.concat([large, second])


@pytest.mark.parametrize("flatten_second", [False, True])
def test_one_unfittable_group_moves_the_whole_dataset_to_the_next_level(
    flatten_second: bool,
) -> None:
    """
    Verify the grouping level is resolved once for the dataset, not per group.

    This is the defect the uniform policy exists to remove. Resolving the level
    per group means some cells are judged by a per-sample model and the rest by a
    pooled one -- and because group SIZE correlates with study arm in almost every
    real cohort (rare condition, fewer cells), which cells got which model then
    correlates with the design factor. A threshold that varies with the factor
    under test is a covariate, not a filter. So a level is used only if EVERY
    group can be fit at it; otherwise the whole dataset drops to the next level.

    Both failure routes must behave identically: a group below ``min_cells`` and a
    group whose fit fails outright are the same fact about that level.
    """

    # Fit two samples, one of which cannot be fit alone, with a pooled fallback.
    metrics = make_two_groups_one_too_small(flatten_second=flatten_second)
    result = fit_mito_mixture(
        metrics,
        QCMitoMixtureConfig(
            enabled=True,
            groupby=["sample_id"],
            fallback_groupby=[[]],
            min_cells=100,
        ),
    )

    # Confirm exactly one model actually scored cells, and it scored all of them.
    fitted = [model for model in result.models if model.fallback is None]
    assert len(fitted) == 1
    assert fitted[0].n_assigned == len(metrics)

    # Confirm it is the pooled fallback, so the fittable sample did NOT keep its
    # own finer model while its neighbour borrowed a coarser one.
    assert fitted[0].level == 1
    assert fitted[0].group == "all"

    # Confirm the abandoned level was reported rather than silently skipped.
    assert any("SAME grouping level" in warning for warning in result.warnings)


def test_uniform_level_projects_ceilings_at_the_level_it_fit() -> None:
    """
    Verify the reported ceiling is grouped the way the model that produced it was.

    Projecting a pooled model onto per-sample ceilings would reintroduce exactly
    what the uniform level removed: a different reported number per group, and so
    a threshold that varies with the design factor, from a model that never had
    per-group parameters to justify it.
    """

    # Fit with a per-sample grouping that has to fall back to pooled.
    result = fit_mito_mixture(
        make_two_groups_one_too_small(),
        QCMitoMixtureConfig(
            enabled=True,
            groupby=["sample_id"],
            fallback_groupby=[[]],
            min_cells=100,
        ),
    )

    # Confirm one pooled ceiling, carrying no grouping columns.
    assert len(result.ceilings) == 1
    assert result.ceilings[0].groupby_columns == ()
    assert result.ceilings[0].ceiling is not None


def test_per_group_policy_still_borrows_strength_and_says_what_it_cost() -> None:
    """
    Verify the old per-group behaviour remains available, with the caveat attached.

    Borrowing strength for one rare group is a real capability: on an atlas with
    thirty cell types, dragging all of them down to one pooled model because one
    type has sixty cells is worse science, since mitochondrial baseline genuinely
    is lineage-specific. So the behaviour stays reachable -- but a run that ends up
    with cells judged at different levels must say so, because that is the
    condition under which the threshold can act as a design covariate.
    """

    # Fit the same data with the per-group policy selected explicitly.
    metrics = make_two_groups_one_too_small()
    result = fit_mito_mixture(
        metrics,
        QCMitoMixtureConfig(
            enabled=True,
            groupby=["sample_id"],
            fallback_groupby=[[]],
            min_cells=100,
            level_policy="per_group",
        ),
    )

    # Confirm the fittable sample kept its own model and the small one borrowed.
    fitted = [model for model in result.models if model.fallback is None]
    assert {model.level for model in fitted} == {0, 1}

    # Confirm every cell was scored.
    assert sum(model.n_assigned for model in fitted) == len(metrics)

    # Confirm the mixed levels were reported as a design-covariate risk.
    assert any("different grouping levels" in warning for warning in result.warnings)


def test_missing_metric_column_is_an_error() -> None:
    """
    Verify a missing modelled metric raises rather than silently disabling.

    Neither modelled metric has a substitute, so proceeding without one would
    quietly deliver a different policy than the config asked for.
    """

    # Drop the mitochondrial metric.
    metrics = make_mixture_metrics().drop(columns=["pct_counts_mito"])

    # Confirm the omission is reported as an error naming the column.
    with pytest.raises(QCMixtureError, match="pct_counts_mito"):
        fit_mito_mixture(metrics, QCMitoMixtureConfig(enabled=True))


def test_cells_with_non_finite_metrics_are_kept() -> None:
    """
    Verify a cell with a missing metric value is kept, not dropped.

    A non-finite metric is treated as a threshold failure downstream, so the model
    must return a keeping probability for these cells rather than a missing one.
    """

    # Blank the metrics for two cells.
    metrics = make_mixture_metrics()
    metrics.loc["cell_0", "pct_counts_mito"] = np.nan
    metrics.loc["cell_1", "n_genes_by_counts"] = np.nan

    # Fit the model.
    result = fit_mito_mixture(metrics, QCMitoMixtureConfig(enabled=True))

    # Confirm both cells received a finite keeping probability.
    assert result.probabilities.loc["cell_0"] == 0.0
    assert result.probabilities.loc["cell_1"] == 0.0
    assert np.isfinite(result.probabilities).all()


def test_keep_all_below_boundary_spares_cells_under_the_intact_line() -> None:
    """
    Verify the boundary rule keeps cells below the intact component's own line.

    Without it, an intact cell can be assigned to the compromised component purely
    because it sits at the sparse edge of the complexity range.
    """

    # Fit with and without the boundary rule.
    metrics = make_mixture_metrics()
    with_rule = fit_mito_mixture(
        metrics, QCMitoMixtureConfig(enabled=True, keep_all_below_boundary=True)
    )
    without_rule = fit_mito_mixture(
        metrics, QCMitoMixtureConfig(enabled=True, keep_all_below_boundary=False)
    )

    # Confirm the rule never flags more cells than its absence does.
    assert (with_rule.probabilities > 0.75).sum() <= (without_rule.probabilities > 0.75).sum()

    # Confirm no cell below the fitted intact line is flagged.
    model = with_rule.models[0]
    boundary = model.intact_intercept + model.intact_slope * metrics["n_genes_by_counts"]
    below = metrics["pct_counts_mito"] < boundary
    assert (with_rule.probabilities[below] == 0.0).all()


def test_hardening_counts_say_how_many_cells_each_rule_moved() -> None:
    """
    Verify the two post-processing rules report their own effect on the call.

    The rules run after the model and can override it in both directions, so the
    number of discarded cells is a joint product of a fit and two policies. Left
    unreported, there is no way to tell a model-driven cut from a policy-driven
    one without re-instrumenting the code, which is exactly what these counts
    exist to avoid.
    """

    # Fit with both rules on and projection off, so the published call is the
    # hardened one the counts describe.
    metrics = make_mixture_metrics()
    config = QCMitoMixtureConfig(
        enabled=True,
        keep_all_below_boundary=True,
        enforce_left_cutoff=True,
        monotone_mito_projection=False,
    )
    result = fit_mito_mixture(metrics, config)
    model = result.models[0]

    # Confirm the raw count is the model's own verdict at the configured cutoff.
    assert model.n_raw_compromised == int((result.posterior > config.posterior_cutoff).sum())

    # Confirm the counts account for the whole distance between the model's
    # verdict and the call that was actually applied.
    applied = int((result.probabilities > config.posterior_cutoff).sum())
    assert applied == (
        model.n_raw_compromised - model.n_rescued_below_boundary + model.n_swept_left_cutoff
    )

    # Confirm disabling a rule zeroes its own count and leaves the raw verdict
    # unchanged, since the rules do not feed back into the fit.
    without_rules = fit_mito_mixture(
        metrics,
        config.model_copy(update={"keep_all_below_boundary": False, "enforce_left_cutoff": False}),
    )
    assert without_rules.models[0].n_rescued_below_boundary == 0
    assert without_rules.models[0].n_swept_left_cutoff == 0
    assert without_rules.models[0].n_raw_compromised == model.n_raw_compromised


def test_projection_makes_the_rule_monotone_in_mitochondrial_content() -> None:
    """
    Verify no discarded cell is cleaner than a retained one in the same group.

    This is the property the projection exists to guarantee. The unprojected
    posterior depends on complexity as well as mitochondrial content, so it can
    discard a cell at a lower mitochondrial fraction than one it keeps -- which
    makes the rule a depth filter wearing a mitochondrial label.
    """

    # Fit with projection enabled.
    metrics = make_mixture_metrics()
    result = fit_mito_mixture(
        metrics, QCMitoMixtureConfig(enabled=True, monotone_mito_projection=True)
    )
    flagged = result.probabilities > 0.75

    # Confirm the populations are actually separated, so the check is meaningful.
    assert flagged.any()
    assert (~flagged).any()

    # Confirm the cleanest discarded cell is dirtier than the dirtiest retained
    # one.
    mito = metrics["pct_counts_mito"]
    assert mito[flagged].min() > mito[~flagged].max()


def test_projection_records_a_faithful_ceiling_per_group() -> None:
    """
    Verify the projected ceiling is recorded and reproduces the model.

    On well-separated data the model's decision is already a mitochondrial cut, so
    the projection should lose nothing and the ceiling should sit between the two
    populations.
    """

    # Fit two cell types with different mitochondrial baselines, as lineages have.
    fibroblasts = make_mixture_metrics(seed=0, cell_type="Fibroblasts")
    keratinocytes = make_mixture_metrics(
        intact_intercept=8.0, compromised_intercept=28.0, seed=1, cell_type="Keratinocytes"
    )
    keratinocytes.index = [f"kc_{position}" for position in range(len(keratinocytes))]
    metrics = pd.concat([fibroblasts, keratinocytes])

    # Fit per cell type.
    result = fit_mito_mixture(metrics, QCMitoMixtureConfig(enabled=True, groupby=["cell_type"]))

    # Confirm one ceiling per group, recorded with its grouping metadata.
    ceilings = {ceiling.group: ceiling for ceiling in result.ceilings}
    assert set(ceilings) == {"Fibroblasts", "Keratinocytes"}
    assert ceilings["Fibroblasts"].groupby_columns == ("cell_type",)
    assert ceilings["Fibroblasts"].group_values == ("Fibroblasts",)

    # Confirm each ceiling reproduces its own model exactly.
    assert ceilings["Fibroblasts"].disagreement == 0
    assert ceilings["Keratinocytes"].disagreement == 0

    # Confirm each ceiling separates that lineage's own two populations, and that
    # the lineage with the higher baseline received the higher ceiling. A single
    # shared ceiling could not do both.
    assert 3.0 < ceilings["Fibroblasts"].ceiling < 16.0
    assert 9.0 < ceilings["Keratinocytes"].ceiling < 26.0
    assert ceilings["Keratinocytes"].ceiling > ceilings["Fibroblasts"].ceiling

    # Confirm the table form carries the same values.
    table = result.ceilings_to_dataframe()
    assert list(table.columns[:5]) == [
        "group",
        "groupby_columns",
        "group_values",
        "n_cells",
        "ceiling",
    ]
    assert len(table) == 2


def test_projection_repairs_and_reports_a_non_monotone_posterior() -> None:
    """
    Verify projection fixes a genuinely non-monotone rule and says that it did.

    The compromised line here crosses the intact one inside the observed
    complexity range, so the raw posterior really does depend on complexity. With
    the two miQC post-rules disabled, that dependence is visible in the decision:
    the raw rule discards cells at a *lower* mitochondrial fraction than others it
    keeps. Projection must repair that, and must report the disagreement rather
    than absorbing it, since a large disagreement means the fit was not measuring
    viability in the first place.
    """

    # Build overlapping populations by letting the compromised line fall through
    # the intact one.
    overlapping = {
        "intact_intercept": 6.0,
        "compromised_intercept": 26.0,
        "compromised_slope": -0.009,
        "noise": 0.3,
        "seed": 3,
    }
    metrics = make_mixture_metrics(**overlapping)
    mito = metrics["pct_counts_mito"]

    # Take the raw posterior, with both post-rules off so nothing has already
    # smoothed the complexity dependence away.
    raw = fit_mito_mixture(
        metrics,
        QCMitoMixtureConfig(
            enabled=True,
            keep_all_below_boundary=False,
            enforce_left_cutoff=False,
            monotone_mito_projection=False,
        ),
    )
    raw_flagged = raw.probabilities > 0.75

    # Confirm the raw rule is non-monotone: it discards a cleaner cell than one it
    # keeps, which is the defect projection exists to remove.
    assert mito[raw_flagged].min() < mito[~raw_flagged].max()

    # Project the same model.
    projected = fit_mito_mixture(
        metrics,
        QCMitoMixtureConfig(
            enabled=True,
            keep_all_below_boundary=False,
            enforce_left_cutoff=False,
            monotone_mito_projection=True,
        ),
    )
    projected_flagged = projected.probabilities > 0.75

    # Confirm the projected rule is monotone.
    assert mito[projected_flagged].min() > mito[~projected_flagged].max()

    # Confirm the repair was reported, naming complexity as the cause.
    assert projected.ceilings[0].disagreement > 0
    assert projected.ceilings[0].disagreement_fraction > PROJECTION_WARN_DISAGREEMENT
    assert any("library complexity" in warning for warning in projected.warnings)


def test_unfittable_group_yields_no_ceiling_and_keeps_every_cell() -> None:
    """
    Verify a group with no fittable structure filters nothing.

    A constant mitochondrial fraction carries no mixture to find. A two-component
    model would ordinarily split it anyway; refusing the fit and recording a null
    ceiling is how "no cut" becomes an expressible outcome.
    """

    # Flatten the mitochondrial metric to a constant.
    metrics = make_mixture_metrics()
    metrics["pct_counts_mito"] = 3.0

    # Fit the model.
    result = fit_mito_mixture(metrics, QCMitoMixtureConfig(enabled=True))

    # Confirm every cell was kept.
    assert (result.probabilities == 0.0).all()

    # Confirm a null ceiling was recorded rather than an arbitrary number.
    assert len(result.ceilings) == 1
    assert result.ceilings[0].ceiling is None
    assert result.ceilings[0].n_removed == 0


def make_mixture_adata(
    *, n_intact: int = 320, n_compromised: int = 80, seed: int = 0
) -> ad.AnnData:
    """
    Build counts whose mitochondrial fraction carries a two-component mixture.

    The mixture is fit on metrics the stage computes from counts, so a stage-level
    test cannot hand it a metric table: the structure has to be built into the
    matrix. Compromised cells get few expressed genes and a high mitochondrial
    share, which is the joint pattern the model keys on.

    Args:
        n_intact: Number of intact cells.
        n_compromised: Number of compromised cells.
        seed: Seed for the synthetic draw.

    Returns:
        AnnData object with one mitochondrial gene and a panel of others.
    """

    # Draw reproducibly.
    generator = np.random.default_rng(seed)
    n_other = 60
    n_cells = n_intact + n_compromised

    # Give each population its own expressed-gene count and mitochondrial share.
    expressed = np.concatenate(
        [
            generator.integers(30, n_other, n_intact),
            generator.integers(5, 25, n_compromised),
        ]
    )
    mito_fraction = np.concatenate(
        [
            generator.uniform(0.02, 0.04, n_intact),
            generator.uniform(0.16, 0.30, n_compromised),
        ]
    )

    # Fill the non-mitochondrial panel to the drawn complexity, then set the
    # mitochondrial count to hit the drawn fraction.
    other = np.zeros((n_cells, n_other), dtype=float)
    for position, count in enumerate(expressed):
        other[position, generator.choice(n_other, size=count, replace=False)] = 5.0
    other_total = other.sum(axis=1)
    mito_counts = np.round(other_total * mito_fraction / (1.0 - mito_fraction))

    # Assemble the object with the mitochondrial gene first.
    return ad.AnnData(
        X=np.column_stack([mito_counts, other]),
        obs=pd.DataFrame(index=[f"cell_{position}" for position in range(n_cells)]),
        var=pd.DataFrame(index=["MT-ND1", *[f"GENE{position}" for position in range(n_other)]]),
    )


def test_the_fitted_model_and_its_posterior_reach_the_run_directory(tmp_path: Path) -> None:
    """
    Verify a real run persists the model and the per-cell posterior.

    Everything above this test operates on in-memory results. What a reader of a
    finished run actually has is the run directory, and until now that directory
    recorded the mitochondrial bound without the fit that produced it or the
    belief it held about any individual cell. Recovering either meant
    instrumenting the engine and re-running.
    """

    # Run the stage with the mixture as the only mitochondrial policy.
    paths = PipelinePaths.from_output_dir(tmp_path / "run")
    paths.ensure_directories()
    result = QCStage().run(
        PipelineContext(
            config=CellQuorumConfig(
                qc=QCConfig(
                    mode="flag_no_drop",
                    threshold_strategy="mad",
                    basic={"min_genes_per_cell": 1, "min_cells_per_gene": 1},
                    mad={"mito_metric": None},
                    mito_mixture={"enabled": True},
                    outputs={"write_h5ad": False, "write_figures": False},
                )
            ),
            paths=paths,
            adata=make_mixture_adata(),
            run_id="mixture-stage-test",
            random_seed=0,
        )
    )
    assert result.status == "success"

    # Confirm the per-cell posterior landed in the metric table, still carrying
    # its intermediate values.
    qc_dir = tmp_path / "run" / "results" / "qc"
    cell_metrics = pd.read_csv(qc_dir / "cell_metrics.csv", index_col=0)
    assert MIQC_POSTERIOR_COLUMN in cell_metrics.columns
    posterior = cell_metrics[MIQC_POSTERIOR_COLUMN]
    assert not posterior.isin([0.0, 1.0]).all()

    # Confirm the fitted model landed beside it, complete enough to be rerun.
    mixture = pd.read_csv(qc_dir / "qc_mito_mixture.csv")
    assert len(mixture) == 1
    for column in (
        "compromised_intercept",
        "compromised_slope",
        "compromised_weight",
        "compromised_variance",
        "intact_intercept",
        "intact_slope",
        "intact_variance",
        "n_raw_compromised",
        "n_rescued_below_boundary",
        "n_swept_left_cutoff",
    ):
        assert column in mixture.columns
    assert mixture["compromised_variance"].iloc[0] > 0.0
    assert mixture["intact_variance"].iloc[0] > 0.0

    # Confirm the recorded model found the injected compromised population rather
    # than an arbitrary split.
    assert 0.1 < mixture["compromised_weight"].iloc[0] < 0.4


def test_config_refuses_two_competing_mitochondrial_policies() -> None:
    """
    Verify the mixture and the mitochondrial MAD rule cannot both be enabled.

    Both are adaptive mitochondrial policies. Running both means the stricter one
    silently wins, and which one that is would vary by dataset, so the run would
    not be describable in a methods section.
    """

    # Confirm enabling both is rejected, with a message naming the way out.
    with pytest.raises(ValueError, match="competing adaptive mitochondrial"):
        QCConfig(
            mad={"mito_metric": "pct_counts_mito"},
            mito_mixture={"enabled": True},
        )

    # Confirm the mixture alone is accepted.
    config = QCConfig(mad={"mito_metric": None}, mito_mixture={"enabled": True})
    assert config.mito_mixture.enabled is True
