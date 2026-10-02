"""The V2 QC figures must obey their spec, not merely render.

Every assertion here corresponds to a named defect in the figures that came before, because
"the figure was produced" is not the property that matters — a figure can render perfectly and
still order the arms backwards, hide the cells a bound acts on, or plot zero for a metric the
assay never measured.

The defects, from the figure spec's own audit of the previous set:

    patient order was string-sorted           P1, P10, P12, P2, P4
    arms disagreed between figures            composition faceted Disease first, the
                                              split-violin put Lymphedema on the left
    an absent metric became a value           0 is a measurement; "not measured" is not
    thresholds were hard-coded into figures   V2 emits no such bound, so none may be drawn
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellquorum.visualization.figstyle import natural_key, paired_condition_order
from cellquorum.visualization.qc.graded import (
    RAW_METRICS,
    _control_arm,
    plot_metric_rainclouds,
)


def _cohort(
    n_per_arm: int = 120, donors: tuple[str, ...] = ("P1", "P2", "P10", "P12")
) -> pd.DataFrame:
    """A paired cohort with the columns the calibration figures read."""
    generator = np.random.default_rng(0)
    rows = []
    for donor in donors:
        for arm in ("Normal", "Lymphedema"):
            shift = 1.25 if arm == "Lymphedema" else 1.0
            rows.append(
                pd.DataFrame(
                    {
                        "donor_id": donor,
                        "condition": arm,
                        "sample_id": f"{donor}_{arm}",
                        "total_counts": generator.lognormal(8.0, 0.6, n_per_arm),
                        "n_genes_by_counts": generator.lognormal(7.2, 0.5, n_per_arm),
                        "pct_counts_mito": generator.gamma(2.0, 2.0 * shift, n_per_arm),
                        "qc_state_initial": "core",
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    frame.index = pd.Index([f"cell_{position}" for position in range(len(frame))])
    return frame


# ═══ Ordering: the rule two previous figures broke in opposite directions ═══════════


def test_donors_sort_naturally_not_lexically() -> None:
    """``P2`` must precede ``P10``.

    The string order `P1, P10, P12, P2, P4` appeared on every donor axis in the previous figure
    set. It is not merely untidy: a reader comparing two panels counts positions, and a cohort
    that is ordered differently from how it is tabulated invites a misread.
    """
    assert sorted(["P10", "P2", "P1", "P12", "P4"], key=natural_key) == [
        "P1",
        "P2",
        "P4",
        "P10",
        "P12",
    ]


def test_the_control_arm_comes_first_within_each_donor() -> None:
    """Normal left, Lymphedema right, donor-major — the house rule, in one place."""
    order = paired_condition_order(
        _cohort(), donor_col="donor_id", condition_col="condition", control="Normal"
    )
    assert order == [
        ("P1", "Normal"),
        ("P1", "Lymphedema"),
        ("P2", "Normal"),
        ("P2", "Lymphedema"),
        ("P10", "Normal"),
        ("P10", "Lymphedema"),
        ("P12", "Normal"),
        ("P12", "Lymphedema"),
    ]


def test_a_donor_missing_an_arm_contributes_the_arm_it_has() -> None:
    """An unpaired donor must not create a gap or drop out silently."""
    frame = _cohort(donors=("P1", "P2"))
    frame = frame[~((frame["donor_id"] == "P2") & (frame["condition"] == "Lymphedema"))]
    order = paired_condition_order(
        frame, donor_col="donor_id", condition_col="condition", control="Normal"
    )
    assert order == [("P1", "Normal"), ("P1", "Lymphedema"), ("P2", "Normal")]


def test_the_control_arm_is_recognised_without_hard_coding_this_study() -> None:
    """CellQuorum is an engine, so "control first" has to work on a cohort named differently."""
    for label in ("Normal", "control", "Healthy", "untreated"):
        frame = pd.DataFrame({"condition": [label, "Treated"]})
        assert _control_arm(frame, "condition") == label

    # And when nothing conventional matches, the answer is None rather than a guess — the figure
    # then falls back to natural order and says so in its title.
    frame = pd.DataFrame({"condition": ["ArmA", "ArmB"]})
    assert _control_arm(frame, "condition") is None


# ═══ Never invent a measurement that was not made ══════════════════════════════════


def test_an_absent_metric_produces_no_figure_rather_than_a_figure_of_zeros(tmp_path: Path) -> None:
    """Substituting 0 for an unmeasured axis is the failure the spec forbids by name.

    A hemoglobin panel of zeros on a tissue with no blood reads as a measurement, and a reader
    has no way to tell it from a real one.
    """
    written = plot_metric_rainclouds(
        _cohort(),
        tmp_path / "qc_metric_missing.png",
        metric="pct_counts_hemoglobin",
        label="Hemoglobin %",
        log_scale=False,
        donor_column="donor_id",
        condition_column="condition",
        control_label="Normal",
    )
    assert written is None
    assert not list(tmp_path.glob("*.png"))


def test_a_metric_of_all_nan_produces_no_figure(tmp_path: Path) -> None:
    """Present-but-empty is the same claim as absent, and must not render either."""
    frame = _cohort()
    frame["pct_counts_mito"] = np.nan
    assert (
        plot_metric_rainclouds(
            frame,
            tmp_path / "qc_metric_nan.png",
            metric="pct_counts_mito",
            label="Mitochondrial %",
            log_scale=False,
            donor_column="donor_id",
            condition_column="condition",
            control_label="Normal",
        )
        is None
    )


def test_a_missing_donor_column_skips_rather_than_raises(tmp_path: Path) -> None:
    """A cohort that declares no donor cannot be paired, and that is not an error."""
    frame = _cohort().drop(columns=["donor_id"])
    assert (
        plot_metric_rainclouds(
            frame,
            tmp_path / "qc_metric_nodonor.png",
            metric="pct_counts_mito",
            label="Mitochondrial %",
            log_scale=False,
            donor_column="donor_id",
            condition_column="condition",
            control_label="Normal",
        )
        is None
    )


# ═══ Output naming and formats ═════════════════════════════════════════════════════


@pytest.mark.parametrize("metric", ["total_counts", "n_genes_by_counts", "pct_counts_mito"])
def test_each_metric_writes_png_pdf_and_svg(tmp_path: Path, metric: str) -> None:
    """The spec requires all three: PNG to look at, PDF to place, SVG to edit."""
    written = plot_metric_rainclouds(
        _cohort(),
        tmp_path / f"qc_metric_{metric}.png",
        metric=metric,
        label=metric,
        log_scale=metric != "pct_counts_mito",
        donor_column="donor_id",
        condition_column="condition",
        control_label="Normal",
    )
    assert written is not None
    for suffix in (".png", ".pdf", ".svg"):
        companion = written.with_suffix(suffix)
        assert companion.exists(), f"{suffix} was not written"
        assert companion.stat().st_size > 0


def test_the_raw_metric_list_covers_the_axes_the_spec_names() -> None:
    """The figure set is defined by the spec, so drift in this list is a spec violation.

    Raw metrics, not severities: a severity has a null subtracted and a saturating transform
    applied, so it cannot be read to decide where the null belongs.
    """
    named = {metric for metric, _label, _log in RAW_METRICS}
    assert {
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mito",
        "pct_counts_ribo",
        "pct_counts_in_top_20_genes",
        "doublet_score",
    } <= named
    # MALAT1 and the dissociation-stress score are read from the persisted RAW axis value, not
    # from the severity that was derived from it.
    assert "qc_ev_malat1_fraction_value" in named
    assert "qc_ev_dissociation_stress_value" in named
    assert not any(metric.endswith("_severity") for metric in named)


# ═══ Determinism ═══════════════════════════════════════════════════════════════════


def test_the_same_seed_draws_the_same_cells() -> None:
    """Point subsampling must be seeded, or a rerun silently redraws a different figure.

    Asserted on the drawn coordinates rather than on the saved file. Both PNG and SVG embed a
    creation timestamp and matplotlib salts clip-path ids per figure, so comparing bytes always
    fails and tells you nothing about the figure; the property that matters is which cells were
    picked and where they were put.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cellquorum.visualization.figstyle import raincloud

    sample = np.random.default_rng(7).lognormal(3.0, 1.0, 4000)

    def offsets(seed: int) -> np.ndarray:
        _figure, axis = plt.subplots()
        raincloud(axis, sample, 0.0, color="#1B4F8A", points=250, seed=seed)
        drawn = np.vstack([collection.get_offsets() for collection in axis.collections])
        plt.close(_figure)
        return drawn

    assert np.array_equal(offsets(0), offsets(0)), "same seed must draw the same cells"
    # And the control: a different seed must actually change the sample, or the seed is ignored
    # and the first assertion passes for the wrong reason.
    assert not np.array_equal(offsets(0), offsets(1))


def test_a_metric_with_no_spread_reports_its_flatness_instead_of_drawing_it(
    tmp_path: Path,
) -> None:
    """A flat metric gets a sentence from the caller, not a panel.

    Hemoglobin on this tissue is exactly 0.000 for 98.5% of cells — skin, no blood. Drawing a
    raincloud of that produced geometry rather than information: the box collapsed to zero
    height, the density estimate degenerated, and the panel rendered as empty rectangles on an
    axis running to -0.15. The flatness is a real QC finding and `write_graded_qc_figures`
    reports it in words; the figure would only have hidden it.

    Distinct from an ABSENT metric, which also returns None: that one was never measured. The
    caller separates them, and the spec forbids conflating them.
    """
    frame = _cohort()
    frame["pct_counts_mito"] = 0.0
    assert (
        plot_metric_rainclouds(
            frame,
            tmp_path / "qc_metric_flat.png",
            metric="pct_counts_mito",
            label="Mitochondrial %",
            log_scale=False,
            donor_column="donor_id",
            condition_column="condition",
            control_label="Normal",
        )
        is None
    )
    assert not list(tmp_path.glob("*.png"))


def test_a_metric_with_narrow_but_real_spread_still_draws(tmp_path: Path) -> None:
    """The control. The flatness guard must not swallow a genuinely narrow distribution.

    A metric can be tightly concentrated and still worth plotting; only a zero inter-quartile
    range with almost every cell on one value is a non-distribution.
    """
    frame = _cohort()
    generator = np.random.default_rng(3)
    frame["pct_counts_mito"] = generator.normal(2.0, 0.05, len(frame))
    written = plot_metric_rainclouds(
        frame,
        tmp_path / "qc_metric_narrow.png",
        metric="pct_counts_mito",
        label="Mitochondrial %",
        log_scale=False,
        donor_column="donor_id",
        condition_column="condition",
        control_label="Normal",
    )
    assert written is not None and written.exists()
