"""The mitochondrial mixture figure must be wired from the fit to the file.

The panel itself is tested elsewhere; what these tests cover is the path between
the fitted model and the figure on disk, which is where the interesting failures
are:

* an adaptive mitochondrial cut is a MODEL, not a number, so the figure exists
  only when a model was actually fit — a run on fixed or MAD thresholds must
  produce no mixture figure rather than an empty axes;
* the applied bound is drawn as one horizontal line, so it may only be passed
  when ONE bound applies to the whole object. A grouped fit derives a ceiling per
  group, and drawing any single one of those across a pooled scatter would assert
  a bound most of the plotted cells were never judged against;
* the panel colours cells by the posterior the model assigned them, which lives
  in the threshold step's derived metrics rather than in the raw metric table, so
  the frame builder has to carry that column through;
* the posterior must NOT join the metric menu. Every metric in that menu is
  measured from the cell and gets a donor-paired test; a posterior is the
  filter's own verdict, and testing it per donor tests the filter against itself.
"""

from __future__ import annotations

from pathlib import Path

# Import matplotlib and select the headless backend before pyplot is imported.
import matplotlib

# Import NumPy for the fixture metrics.
import numpy as np

# Import pandas for the fixture tables.
import pandas as pd

matplotlib.use("Agg")

from cellquorum.stages.qc.artifacts import resolve_mixture_panel_inputs  # noqa: E402
from cellquorum.stages.qc.mixture import (  # noqa: E402
    MIQC_POSTERIOR_COLUMN,
    MitoCeiling,
    MitoMixtureModel,
    MitoMixtureResult,
)
from cellquorum.visualization.qc.panels import (  # noqa: E402
    METRIC_LABELS,
    assemble_qc_frame,
    write_qc_panels,
)

# Cells per condition arm in the fixture cohort.
CELLS_PER_ARM = 60


def make_model(group: str = "all") -> MitoMixtureModel:
    """
    Build a fitted model record with both variances, so the boundary is drawable.

    Args:
        group: Group label the model was fit on.

    Returns:
        A model record whose compromised component is the steeper, higher one.
    """

    # Give the compromised component the higher intercept and the steeper decline
    # with complexity, which is the direction damaged cells actually take.
    return MitoMixtureModel(
        group=group,
        n_cells=2 * CELLS_PER_ARM,
        converged=True,
        n_iterations=12,
        log_likelihood=-1234.5,
        compromised_weight=0.18,
        compromised_intercept=14.0,
        compromised_slope=-0.004,
        intact_intercept=3.0,
        intact_slope=-0.0004,
        n_compromised=20,
        compromised_variance=6.0,
        intact_variance=1.2,
    )


def make_ceiling(group: str = "all", ceiling: float | None = 7.4) -> MitoCeiling:
    """
    Build a projected-ceiling record.

    Args:
        group: Group label the projection applies to.
        ceiling: Mitochondrial percentage above which cells are discarded, or
            None for the case where the model flagged nothing.

    Returns:
        A ceiling record.
    """

    return MitoCeiling(
        groupby_columns=() if group == "all" else ("cell_type",),
        group_values=() if group == "all" else (group,),
        group=group,
        n_cells=2 * CELLS_PER_ARM,
        ceiling=ceiling,
        n_removed=0 if ceiling is None else 18,
        disagreement=2,
        disagreement_fraction=2 / (2 * CELLS_PER_ARM),
    )


def make_threshold_result(mixture: MitoMixtureResult | None) -> MitoMixtureResult | None:
    """Identity, kept so the call sites below read unchanged.

    ``resolve_mixture_panel_inputs`` used to take a threshold result and reach inside it for the
    fitted mixture. It now takes the mixture, because the mixture never needed the threshold path
    to reach a figure — that indirection was the tie that kept two QC systems alive.
    """
    return mixture


def make_cohort() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build obs, metrics and decisions for a small paired cohort.

    Returns:
        ``(obs, metrics, decisions)``, filtered on a mitochondrial ceiling so the
        panel has both a retained and a removed population to draw.
    """

    rng = np.random.default_rng(11)
    records = []
    for donor in ("P1", "P2", "P3"):
        for condition in ("Normal", "Lymphedema"):
            for index in range(CELLS_PER_ARM // 3):
                records.append(
                    {
                        "cell": f"{donor}_{condition}_{index}",
                        "sample_id": f"{donor}_{condition}",
                        "donor_id": donor,
                        "condition": condition,
                    }
                )
    obs = pd.DataFrame(records).set_index("cell")

    n = len(obs)
    metrics = pd.DataFrame(
        {
            "total_counts": rng.lognormal(mean=8.5, sigma=0.4, size=n),
            "n_genes_by_counts": rng.lognormal(mean=7.4, sigma=0.35, size=n),
            "pct_counts_mito": rng.gamma(shape=2.0, scale=2.0, size=n),
        },
        index=obs.index,
    )
    high_mito = metrics["pct_counts_mito"] > 7.4
    decisions = pd.DataFrame({"mito_mixture": high_mito}, index=obs.index)
    decisions["keep"] = ~high_mito
    return obs, metrics, decisions


def make_frame(with_posterior: bool = True) -> pd.DataFrame:
    """
    Build the tidy per-cell frame the panels read.

    Args:
        with_posterior: Attach the per-cell compromised posterior, as a run using
            the mixture policy would.

    Returns:
        The assembled frame.
    """

    obs, metrics, decisions = make_cohort()
    if with_posterior:
        # Monotone in mitochondrial percentage, which is what the projected filter
        # acts on, and bounded so the colour scale spans its full range.
        scaled = metrics["pct_counts_mito"] / metrics["pct_counts_mito"].max()
        metrics = metrics.assign(**{MIQC_POSTERIOR_COLUMN: scaled.clip(0.0, 1.0)})
    return assemble_qc_frame(
        obs=obs,
        cell_metrics=metrics,
        cell_decisions=decisions,
        sample_key="sample_id",
        donor_key="donor_id",
        condition_key="condition",
    )


def test_resolve_returns_nothing_without_a_mixture():
    """A run on fixed or MAD thresholds has no model, so there is no figure."""
    assert resolve_mixture_panel_inputs(make_threshold_result(None)) == (None, None)


def test_resolve_returns_nothing_when_no_group_was_fit():
    """The policy can run and fit nothing; an empty table is still no figure."""
    result = MitoMixtureResult(probabilities=pd.Series(dtype=float))
    assert resolve_mixture_panel_inputs(make_threshold_result(result)) == (None, None)


def test_resolve_returns_the_single_unambiguous_ceiling():
    """One ceiling applies to the whole object, so it can be drawn as a line."""
    result = MitoMixtureResult(
        probabilities=pd.Series(dtype=float),
        models=[make_model()],
        ceilings=[make_ceiling()],
    )
    models, ceiling = resolve_mixture_panel_inputs(make_threshold_result(result))
    assert models is not None and len(models) == 1
    assert ceiling == 7.4


def test_resolve_withholds_the_ceiling_from_a_grouped_fit():
    """Per-group ceilings must not be drawn as one line across pooled cells."""
    result = MitoMixtureResult(
        probabilities=pd.Series(dtype=float),
        models=[make_model("LEC"), make_model("Mast")],
        ceilings=[make_ceiling("LEC", 5.7), make_ceiling("Mast", 11.2)],
    )
    models, ceiling = resolve_mixture_panel_inputs(make_threshold_result(result))
    assert len(models) == 2
    assert ceiling is None


def test_resolve_withholds_a_ceiling_that_filtered_nothing():
    """A group the model flagged nothing in has no bound to report."""
    result = MitoMixtureResult(
        probabilities=pd.Series(dtype=float),
        models=[make_model()],
        ceilings=[make_ceiling(ceiling=None)],
    )
    models, ceiling = resolve_mixture_panel_inputs(make_threshold_result(result))
    assert models is not None
    assert ceiling is None


def test_resolve_returns_models_when_the_projection_was_off():
    """A fit with no projection at all still gets its figure, without the line."""
    result = MitoMixtureResult(probabilities=pd.Series(dtype=float), models=[make_model()])
    models, ceiling = resolve_mixture_panel_inputs(make_threshold_result(result))
    assert models is not None and len(models) == 1
    assert ceiling is None


def test_frame_carries_the_posterior_from_the_metric_table():
    """The panel needs the model's per-cell verdict, so the frame must keep it."""
    frame = make_frame(with_posterior=True)
    assert MIQC_POSTERIOR_COLUMN in frame.columns
    values = frame[MIQC_POSTERIOR_COLUMN].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_frame_omits_the_posterior_when_the_run_had_no_model():
    """A fixed-threshold run has no posterior, and the column is simply absent."""
    assert MIQC_POSTERIOR_COLUMN not in make_frame(with_posterior=False).columns


def test_posterior_is_not_offered_as_a_metric():
    """The posterior is the filter's verdict, not a measurement of the cell.

    The metric menu drives the sample matrix and the donor-paired dumbbells. A
    donor-paired test of mean P(compromised) asks whether the filter thought one
    arm was worse, which it did by construction — so the column has to stay out.
    """
    assert MIQC_POSTERIOR_COLUMN not in METRIC_LABELS


def test_panels_write_the_mixture_figure_when_a_model_was_fit(tmp_path: Path):
    """The mixture figure lands on disk alongside the rest of the panel set."""
    paths = write_qc_panels(
        make_frame(),
        tmp_path,
        mixture_models=MitoMixtureResult(
            probabilities=pd.Series(dtype=float), models=[make_model()]
        ).to_dataframe(),
        mixture_ceiling=7.4,
        formats=("png",),
        dpi=80,
    )
    names = {Path(path).stem for path in paths}
    assert "qc_mito_mixture" in names


def test_panels_omit_the_mixture_figure_without_a_model(tmp_path: Path):
    """No model means no figure — not a stub, and not an empty axes."""
    paths = write_qc_panels(make_frame(), tmp_path, formats=("png",), dpi=80)
    names = {Path(path).stem for path in paths}
    assert "qc_mito_mixture" not in names
    # Sanity: the rest of the sheet still rendered, so the assertion above is
    # about the mixture panel and not about a writer that produced nothing.
    assert names


def test_panels_write_the_mixture_figure_without_a_posterior(tmp_path: Path):
    """A run that fit a model but recorded no posterior still gets the figure.

    Older runs published the adjusted probability only. The panel falls back to
    the keep/remove split there, and a figure in two colours is worth more than
    no figure at all.
    """
    paths = write_qc_panels(
        make_frame(with_posterior=False),
        tmp_path,
        mixture_models=MitoMixtureResult(
            probabilities=pd.Series(dtype=float), models=[make_model()]
        ).to_dataframe(),
        formats=("png",),
        dpi=80,
    )
    assert "qc_mito_mixture" in {Path(path).stem for path in paths}
