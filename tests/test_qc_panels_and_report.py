"""The QC figure panels and the HTML QC report must describe the FILTER.

Both artifacts exist because the per-metric QC plots could not answer "what did
QC remove". So the invariants under test are all about the pre-filter population:

* every summary counts input cells, not survivors — a table built from the
  surviving object reports zero removed however many cells were dropped;
* per-rule failure counts never sum to the total removed, and the honest
  marginal number is "cells this rule alone caught";
* a panel with nothing to show is omitted, not stubbed with an empty axes;
* the HTML report is one self-contained file, because a report that needs a CDN
  is unreadable from an offline run directory in two years.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.qc import panels as panels_module  # noqa: E402
from cellquorum.visualization.qc.html_report import (  # noqa: E402
    build_rule_attribution_table,
    build_sample_qc_table,
    render_qc_html_report,
    write_qc_html_report,
)
from cellquorum.visualization.qc.panels import (  # noqa: E402
    MATRIX_METRICS,
    METRIC_LABELS,
    SMALL_CELL_TYPE,
    UNLABELLED,
    QCPanelError,
    assemble_qc_frame,
    cell_type_display_rows,
    order_cell_types,
    plot_paired_dumbbell,
    plot_sample_matrix,
    resolve_cell_type_keys,
    summarize_by_cell_type,
    summarize_by_sample,
    summarize_rules,
    write_qc_cell_type_figure,
    write_qc_panels,
)
from cellquorum.visualization.qc.publication_table import (  # noqa: E402
    Column,
    QCTableError,
    TypesetTable,
    build_cell_type_table,
    build_cohort_table,
    build_criteria_table,
    render_table_latex,
    write_qc_publication_tables,
)

# Donors deliberately named so that a plain string sort scrambles them: P2 must
# come before P10 in every panel that lists donors.
DONORS = ("P1", "P2", "P10")
CONDITIONS = ("Normal", "Lymphedema")
CELLS_PER_SAMPLE = 40


def _cohort() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build obs, metrics and decisions for a paired 3-donor cohort.

    Two rules with deliberate overlap: a mito rule and a complexity rule, where
    some cells trip both. That overlap is the thing the attribution panel exists
    to report, so a fixture without it would not exercise it.
    """
    rng = np.random.default_rng(7)
    records = []
    for donor in DONORS:
        for condition in CONDITIONS:
            for index in range(CELLS_PER_SAMPLE):
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
            "pct_counts_mito": rng.gamma(shape=2.0, scale=1.5, size=n),
            "pct_counts_ribo": rng.gamma(shape=3.0, scale=2.0, size=n),
            # Constant on purpose: a metric with no variance must be dropped from
            # the matrix rather than drawn as a uniform column.
            "pct_counts_hemoglobin": np.zeros(n),
        },
        index=obs.index,
    )

    high_mito = metrics["pct_counts_mito"] > 6.0
    low_complexity = metrics["n_genes_by_counts"] < 1200.0
    decisions = pd.DataFrame(
        {
            "fixed_max_mito_percent": high_mito,
            "fixed_min_genes_per_cell": low_complexity,
        },
        index=obs.index,
    )
    decisions["keep"] = ~(high_mito | low_complexity)
    return obs, metrics, decisions


def _thresholds() -> pd.DataFrame:
    """The applied bounds, in the shape ``QCThresholdResult.to_dataframe`` emits."""
    return pd.DataFrame(
        [
            {
                "rule_name": "fixed_max_mito_percent",
                "metric": "pct_counts_mito",
                "lower": None,
                "upper": 6.0,
                "strategy": "fixed",
            },
            {
                "rule_name": "fixed_min_genes_per_cell",
                "metric": "n_genes_by_counts",
                "lower": 1200.0,
                "upper": None,
                "strategy": "fixed",
            },
        ]
    )


def _frame() -> pd.DataFrame:
    """The tidy per-cell frame the panels read, built the replot way."""
    obs, metrics, decisions = _cohort()
    return assemble_qc_frame(
        obs=obs,
        cell_metrics=metrics,
        cell_decisions=decisions,
        sample_key="sample_id",
        donor_key="donor_id",
        condition_key="condition",
    )


def _annotated_cohort() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The same cohort with cell-type labels, uneven attrition, and gaps.

    Attrition has to differ between cell types or the outputs it feeds prove
    nothing, so nearly every removal is concentrated in one granular type. The
    fixture also carries the two cases that are easy to get wrong and common in
    practice: a type of five cells, where a percentage is not a rate, and a block
    of cells annotation never labelled at all.
    """
    obs, metrics, decisions = _cohort()
    failed = list(decisions.index[~decisions["keep"]])
    passed = list(decisions.index[decisions["keep"]])
    assert len(failed) > 20 and len(passed) > 60, "fixture must have both arms"

    labels: dict[str, str] = {}
    # A depleted type: every removal but ten, plus twenty survivors.
    for cell in failed[:-10] + passed[:20]:
        labels[cell] = "LEC PROX1 hi"
    # The bulk of the dataset, carrying the ten remaining removals.
    for cell in passed[20:-45] + failed[-10:]:
        labels[cell] = "LEC ACKR4+"
    # A type too small for a percentage to mean anything.
    for cell in passed[-45:-40]:
        labels[cell] = "Sweat gland"
    # A coarse group with exactly one granular member.
    for cell in passed[-40:-10]:
        labels[cell] = "Fibroblast CCL19+"
    # passed[-10:] stay out of the map: annotation never saw them, so they are
    # NaN — not a cell type called "nan".

    granular = pd.Series(labels).reindex(obs.index)
    annotated = obs.copy()
    annotated["cell_type"] = granular.map(
        {
            "LEC PROX1 hi": "LEC",
            "LEC ACKR4+": "LEC",
            "Sweat gland": "LEC",
            "Fibroblast CCL19+": "Fibroblast",
        }
    )
    annotated["cell_type_granular"] = granular
    return annotated, metrics, decisions


def _annotated_frame() -> pd.DataFrame:
    """The per-cell frame with cell-type labels resolved."""
    obs, metrics, decisions = _annotated_cohort()
    return assemble_qc_frame(
        obs=obs,
        cell_metrics=metrics,
        cell_decisions=decisions,
        sample_key="sample_id",
        donor_key="donor_id",
        condition_key="condition",
    )


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------


def test_assemble_frame_covers_every_input_cell() -> None:
    """The frame is indexed by input cells, so it can show what left."""
    obs, metrics, decisions = _cohort()
    frame = _frame()

    # Precondition: the fixture must actually remove cells.
    assert int((~decisions["keep"]).sum()) > 0

    assert len(frame) == len(obs)
    assert int(frame["keep"].sum()) == int(decisions["keep"].sum())
    assert int((~frame["keep"]).sum()) == int((~decisions["keep"]).sum())

    # Metrics travel for the removed cells too, or a pre-filter histogram would
    # still only show survivors.
    assert frame.loc[~frame["keep"], "total_counts"].notna().all()


def test_assemble_frame_from_annotated_obs_matches_the_csv_path() -> None:
    """Both entry points agree: in-pipeline obs, and CSVs read off a finished run.

    The stage writes ``cellquorum_qc_*`` columns onto the figure object; the
    replot path has only the tables. A disagreement here would mean figures drawn
    during a run and figures redrawn afterwards tell different stories.
    """
    obs, metrics, decisions = _cohort()
    annotated = obs.join(metrics)
    annotated["cellquorum_qc_keep"] = decisions["keep"]
    for rule in ("fixed_max_mito_percent", "fixed_min_genes_per_cell"):
        annotated[f"cellquorum_qc_{rule}"] = decisions[rule]

    from_obs = assemble_qc_frame(
        obs=annotated, sample_key="sample_id", donor_key="donor_id", condition_key="condition"
    )
    from_csv = _frame()

    assert sorted(from_obs.columns) == sorted(from_csv.columns)
    pd.testing.assert_frame_equal(
        from_obs[sorted(from_obs.columns)], from_csv[sorted(from_csv.columns)]
    )


def test_assemble_frame_requires_a_keep_decision() -> None:
    """No keep/fail column and no decision table is a hard error, not a guess."""
    obs, metrics, _ = _cohort()
    with pytest.raises(QCPanelError, match="keep/fail decision"):
        assemble_qc_frame(obs=obs.join(metrics), sample_key="sample_id")


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def test_sample_summary_counts_the_pre_filter_population() -> None:
    """cells_in is every input cell for that sample, and the totals reconcile."""
    frame = _frame()
    table = summarize_by_sample(frame)

    assert len(table) == len(DONORS) * len(CONDITIONS)
    assert (table["cells_in"] == CELLS_PER_SAMPLE).all()
    assert table["cells_removed"].sum() == int((~frame["keep"]).sum())
    assert (table["cells_removed"] > 0).any(), "fixture must drop cells somewhere"
    assert np.allclose(table["pct_removed"], 100.0 * table["cells_removed"] / table["cells_in"])


def test_rule_summary_reports_unique_contribution_not_just_gross_failures() -> None:
    """Rules overlap, so gross counts oversell each rule's independent effect.

    The dark inner bar on the attribution panel is ``n_unique``: cells no other
    rule would have caught. It must never exceed the gross count, and the gross
    counts must over-sum the total removed whenever any cell trips two rules.
    """
    frame = _frame()
    table = summarize_rules(frame, _thresholds())

    assert len(table) == 2
    assert (table["n_unique"] <= table["n_failed"]).all()

    n_removed = int((~frame["keep"]).sum())
    overlap = int(
        (frame["rule:fixed_max_mito_percent"] & frame["rule:fixed_min_genes_per_cell"]).sum()
    )
    assert overlap > 0, "fixture must have cells failing both rules"
    assert table["n_failed"].sum() == n_removed + overlap
    assert table["n_unique"].sum() == n_removed - overlap


def test_rule_labels_carry_the_applied_bound() -> None:
    """A rule name alone is not a QC record; the number it enforced is."""
    labels = " | ".join(summarize_rules(_frame(), _thresholds())["label"])

    assert "Mitochondrial %" in labels
    assert "6" in labels
    assert "Genes per cell" in labels
    # Thousands separated, never in exponent form: "1.2e+03" is not a threshold a
    # reader can check against a metric table.
    assert "1,200" in labels


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def test_sample_matrix_drops_constant_metrics_and_orders_donors_naturally() -> None:
    """No blank column, and P2 before P10.

    The hemoglobin row on the old sheet was a uniform block of colour because the
    metric is zero everywhere in this tissue; a lexical donor sort produced
    P1, P10, P2, which reads as a scrambled cohort.
    """
    import matplotlib.pyplot as plt

    table = summarize_by_sample(_frame())
    figure, axes = plt.subplots(figsize=(5, 4))
    plot_sample_matrix(axes, table, case_label="Lymphedema")

    columns = [label.get_text() for label in axes.get_xticklabels()]
    assert "%hb" not in columns
    assert "%mito" in columns

    rows = [label.get_text() for label in axes.get_yticklabels()]
    assert rows == [
        "P1_Normal",
        "P1_Lymphedema",
        "P2_Normal",
        "P2_Lymphedema",
        "P10_Normal",
        "P10_Lymphedema",
    ]
    plt.close(figure)


def test_paired_dumbbell_refuses_an_unpaired_table() -> None:
    """Without donor/condition there is no within-donor contrast to draw."""
    import matplotlib.pyplot as plt

    frame = _frame().drop(columns=["donor"])
    table = summarize_by_sample(frame)
    figure, axes = plt.subplots()
    with pytest.raises(QCPanelError, match="donor"):
        plot_paired_dumbbell(axes, table, "pct_counts_mito", case_label="Lymphedema")
    plt.close(figure)


def test_write_qc_panels_emits_the_full_set_without_layout_warnings(tmp_path: Path) -> None:
    """Every panel renders, and matplotlib has no complaint about the layout.

    The layout warnings are not cosmetic: "axes sizes collapsed to zero" means
    constrained layout gave up and the sheet was saved with overlapping panels.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = write_qc_panels(
            _frame(), tmp_path, thresholds=_thresholds(), case_label="Lymphedema"
        )

    layout_warnings = [str(w.message) for w in caught if "constrained_layout" in str(w.message)]
    assert layout_warnings == []

    stems = {path.stem for path in paths}
    assert stems == {
        "qc_overview",
        "qc_attrition",
        "qc_joint_density",
        "qc_sample_attrition",
        "qc_sample_matrix",
        "qc_donor_paired",
    }
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def _axes_titles(axes) -> list:
    """The axes' three title artists, whichever carry text.

    ``ax.title`` is only the CENTRED title. Every panel here titles with
    ``loc="left"``, which matplotlib keeps on a separate artist, so reading
    ``ax.title`` finds an empty string and measures nothing — which is how the
    first version of this test passed against a figure that visibly read
    "APopulation size".
    """
    candidates = [
        getattr(axes, "_left_title", None),
        axes.title,
        getattr(axes, "_right_title", None),
    ]
    return [title for title in candidates if title is not None and title.get_text()]


def _panel_letter_collisions(figure) -> tuple[list[str], int]:
    """Overlaps between each panel letter and other text on its axes, and a count.

    Measured on the rendered canvas, because that is the only place the collision
    exists: the letter is offset in POINTS from the axes corner and the title is
    left-aligned to that same corner, so whether they touch depends on the glyph
    metrics of the font and on nothing a reader of the source can see.

    Returns:
        The collision messages, and how many (letter, text) pairs were compared —
        the caller asserts that is non-zero, so a lookup that finds no artists
        fails loudly instead of passing vacuously.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    collisions: list[str] = []
    comparisons = 0
    for axes in figure.axes:
        letters = [
            child
            for child in axes.texts
            if len(child.get_text()) == 1 and child.get_text().isupper()
        ]
        # Titles, subtitles and any other label on the axes. The letter sits above
        # the axes, so in-axes data labels cannot collide with it by construction;
        # comparing against everything costs nothing and catches the subtitle too.
        others = _axes_titles(axes) + [
            child for child in axes.texts if child not in letters and child.get_text()
        ]
        for letter in letters:
            box = letter.get_window_extent(renderer=renderer)
            for other in others:
                comparisons += 1
                other_box = other.get_window_extent(renderer=renderer)
                if box.overlaps(other_box):
                    collisions.append(
                        f"panel letter {letter.get_text()!r} overlaps "
                        f"{other.get_text()[:40]!r} "
                        f"(letter {box.bounds}, other {other_box.bounds})"
                    )
    return collisions, comparisons


def test_no_panel_letter_overlaps_its_own_title(tmp_path: Path, monkeypatch) -> None:
    """Panel letters clear their titles on every shipped QC figure.

    This has now shipped broken twice — "D" written through "Doublet-score
    separation", and the by-cell-type sheet reading "APopulation size" — because
    the offset is a hand-tuned literal per call site and the only way to check it
    was to look at the PNG. Capture the figures instead of writing them, and
    measure.
    """
    captured: list = []

    def capture(figure, *args, **kwargs):
        captured.append(figure)
        return []

    monkeypatch.setattr(panels_module, "save_figure", capture)
    write_qc_panels(_annotated_frame(), tmp_path, thresholds=_thresholds(), case_label="Lymphedema")

    assert captured, "no figures were rendered"
    collisions: list[str] = []
    comparisons = 0
    for figure in captured:
        found, count = _panel_letter_collisions(figure)
        collisions += found
        comparisons += count
    assert comparisons > 0, "measured nothing: no panel letters or no titles were found"
    assert collisions == [], "\n".join(collisions)


def test_write_qc_panels_omits_per_sample_panels_when_there_are_no_samples(
    tmp_path: Path,
) -> None:
    """A single unlabelled library gets a shorter sheet, not empty axes."""
    frame = _frame().drop(columns=["sample", "donor", "condition"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = write_qc_panels(frame, tmp_path, thresholds=_thresholds())

    assert [str(w.message) for w in caught if "constrained_layout" in str(w.message)] == []
    stems = {path.stem for path in paths}
    assert stems == {"qc_overview", "qc_attrition", "qc_joint_density"}


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def test_html_sample_table_reports_removal_and_reconciles_to_total() -> None:
    """The TOTAL row is the cohort, and its medians are pooled over cells.

    Averaging per-sample medians would weight a 20-cell sample the same as a
    500-cell one, which on this cohort is a factor of five.
    """
    obs, metrics, decisions = _cohort()
    table = build_sample_qc_table(
        cell_metrics=metrics,
        cell_decisions=decisions,
        obs=obs,
        sample_key="sample_id",
        donor_key="donor_id",
        condition_key="condition",
    )

    per_sample = table.loc[table["sample"] != "TOTAL"]
    total = table.loc[table["sample"] == "TOTAL"].iloc[0]

    assert len(per_sample) == len(DONORS) * len(CONDITIONS)
    assert int(total["cells_in"]) == len(obs)
    assert int(total["cells_removed"]) == int((~decisions["keep"]).sum())
    assert int(total["cells_removed"]) > 0
    assert int(per_sample["cells_removed"].sum()) == int(total["cells_removed"])

    # Pooled over cells, which for a skewed metric differs from the mean of the
    # per-sample medians.
    assert total["median_pct_mito"] == pytest.approx(float(metrics["pct_counts_mito"].median()))


def test_html_sample_table_falls_back_to_one_row_without_a_sample_column() -> None:
    """No sample labels still yields a usable cohort report, not an exception."""
    obs, metrics, decisions = _cohort()
    table = build_sample_qc_table(
        cell_metrics=metrics,
        cell_decisions=decisions,
        obs=obs.drop(columns=["sample_id"]),
        sample_key="sample_id",
    )

    assert list(table["sample"]) == ["all cells", "TOTAL"]
    assert int(table["cells_removed"].iloc[-1]) == int((~decisions["keep"]).sum())


def test_html_report_is_a_single_self_contained_file(tmp_path: Path) -> None:
    """No CDN, no sidecar assets: a run directory must render years later offline."""
    obs, metrics, decisions = _cohort()
    path = write_qc_html_report(
        tmp_path / "qc_report.html",
        cell_metrics=metrics,
        cell_decisions=decisions,
        obs=obs,
        sample_key="sample_id",
        donor_key="donor_id",
        condition_key="condition",
        thresholds=_thresholds(),
        gene_summary={"n_genes": 2000, "n_genes_kept": 1240},
        project="test_cohort",
        floors={"min_genes_per_cell": 200, "min_cells_per_gene": 3},
        case_label="Lymphedema",
    )
    html = path.read_text(encoding="utf-8")

    assert list(tmp_path.iterdir()) == [path]
    assert "http://" not in html and "https://" not in html
    assert "<style" in html and "<script" in html

    # The numbers a reader checks first.
    n_removed = int((~decisions["keep"]).sum())
    assert f"{len(obs):,}" in html
    assert f"{n_removed:,}" in html
    # Gene filtering is reported, since it is invisible in every cell-level panel.
    assert "1,240" in html


def test_typeset_cohort_table_describes_the_retained_dataset() -> None:
    """Table 1 counts attrition pre-filter but reports medians on survivors.

    The two halves answer different questions and are easy to conflate: a median
    computed over every input cell describes the cells QC threw away as much as
    the ones the paper analyses.
    """
    obs, metrics, decisions = _cohort()
    table = build_cohort_table(
        cell_metrics=metrics,
        cell_decisions=decisions,
        obs=obs,
        sample_key="sample_id",
        donor_key="donor_id",
        condition_key="condition",
        thresholds=_thresholds(),
        case_label="Lymphedema",
    )

    assert table.total is not None
    assert int(table.total["cells_in"]) == len(obs)
    assert int(table.total["cells_removed"]) == int((~decisions["keep"]).sum())

    retained = decisions["keep"].astype(bool)
    assert table.total["median_pct_mito"] == pytest.approx(
        float(metrics.loc[retained, "pct_counts_mito"].median())
    )
    assert table.total["median_pct_mito"] != pytest.approx(
        float(metrics["pct_counts_mito"].median())
    )

    # Donors become row groups, so a donor column would be an empty stub repeating
    # each heading.
    assert table.row_group == "donor"
    assert "donor" not in {column.key for column in table.columns}


def test_typeset_cohort_table_reads_control_first_and_donors_numerically() -> None:
    """Row order matches the figures: P2 before P10, Normal before Lymphedema."""
    obs, metrics, decisions = _cohort()
    table = build_cohort_table(
        cell_metrics=metrics,
        cell_decisions=decisions,
        obs=obs,
        sample_key="sample_id",
        donor_key="donor_id",
        condition_key="condition",
        case_label="Lymphedema",
    )

    assert list(table.body["sample"]) == [
        "P1_Normal",
        "P1_Lymphedema",
        "P2_Normal",
        "P2_Lymphedema",
        "P10_Normal",
        "P10_Lymphedema",
    ]


def test_typeset_cohort_note_excludes_gene_level_criteria() -> None:
    """A note under a table of cells must not list the gene expression filter.

    "n cells by counts < 3" printed beneath a per-cell table reads as a cell
    criterion when it is the gene filter, reported separately in the same note.
    """
    obs, metrics, decisions = _cohort()
    thresholds = _thresholds()
    thresholds["axis"] = "cell"
    gene_rule = pd.DataFrame(
        {
            "rule_name": ["fixed_min_cells_per_gene"],
            "metric": ["n_cells_by_counts"],
            "lower": [3.0],
            "upper": pd.Series([np.nan], dtype="float64"),
            "strategy": ["fixed"],
            "axis": ["gene"],
        }
    )
    thresholds = pd.concat([thresholds, gene_rule], ignore_index=True)
    table = build_cohort_table(
        cell_metrics=metrics,
        cell_decisions=decisions,
        obs=obs,
        sample_key="sample_id",
        donor_key="donor_id",
        thresholds=thresholds,
        gene_summary={"n_genes": 2000, "n_genes_kept": 1240},
    )

    assert table.source_note is not None
    assert "Mitochondrial %" in table.source_note
    assert "cells per gene" not in table.source_note.lower()
    assert "1,240 of 2,000 genes retained" in table.source_note


def test_typeset_criteria_table_totals_the_union_not_the_sum() -> None:
    """The total row is "any criterion", which is less than the column sum."""
    _, _, decisions = _cohort()
    table = build_criteria_table(cell_decisions=decisions, thresholds=_thresholds())

    n_removed = int((~decisions["keep"]).sum())
    assert table.total is not None
    assert int(table.total["cells_failed"]) == n_removed
    assert int(table.body["cells_failed"].sum()) > n_removed


def test_latex_table_is_valid_booktabs_with_escaped_specials() -> None:
    """Every row has the declared column count, and no raw LaTeX special survives.

    An unescaped ``_`` in a sample name is a compile error, and a text-mode ``<``
    silently prints as inverted punctuation instead of a less-than sign.
    """
    obs, metrics, decisions = _cohort()
    table = build_cohort_table(
        cell_metrics=metrics,
        cell_decisions=decisions,
        obs=obs,
        sample_key="sample_id",
        donor_key="donor_id",
        condition_key="condition",
        thresholds=_thresholds(),
        case_label="Lymphedema",
    )
    tex = render_table_latex(table)

    assert tex.count("\\toprule") == 1
    assert tex.count("\\bottomrule") == 1
    assert "\\cmidrule(lr){" in tex
    spec = "".join("l" if column.align == "left" else "r" for column in table.columns)
    assert f"\\begin{{tabular}}{{{spec}}}" in tex

    body_rows = [
        line for line in tex.splitlines() if line.endswith("\\\\") and "multicolumn" not in line
    ]
    assert body_rows
    for line in body_rows:
        assert line.count("&") == len(table.columns) - 1, line

    assert "P1\\_Normal" in tex
    assert "P1_Normal" not in tex
    # The criteria note carries both a percent sign and a comparison operator.
    assert "\\%" in tex
    assert "\\textgreater{}" in tex or "\\textless{}" in tex


def test_html_tables_page_is_self_contained_and_marks_the_total_row(
    tmp_path: Path,
) -> None:
    """One offline page, with the total row and spanners marked up for styling."""
    obs, metrics, decisions = _cohort()
    paths = write_qc_publication_tables(
        tmp_path,
        cell_metrics=metrics,
        cell_decisions=decisions,
        obs=obs,
        sample_key="sample_id",
        donor_key="donor_id",
        condition_key="condition",
        thresholds=_thresholds(),
        gene_summary={"n_genes": 2000, "n_genes_kept": 1240},
        case_label="Lymphedema",
        project="test_cohort",
    )

    names = {path.name for path in paths}
    assert names == {
        "qc_tables.html",
        "qc_table1_cohort.tex",
        "qc_table1_cohort.png",
        "qc_table2_criteria.tex",
        "qc_table2_criteria.png",
    }
    for path in paths:
        assert path.stat().st_size > 0

    html = (tmp_path / "qc_tables.html").read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html
    assert 'class="total"' in html
    assert 'colspan="4">Cells<' in html
    assert "Table 1." in html and "Table 2." in html


def test_typeset_tables_render_without_matplotlib_warnings(tmp_path: Path) -> None:
    """A table figure is laid out by hand, so any matplotlib warning is a defect."""
    obs, metrics, decisions = _cohort()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_qc_publication_tables(
            tmp_path,
            cell_metrics=metrics,
            cell_decisions=decisions,
            obs=obs,
            sample_key="sample_id",
            donor_key="donor_id",
            condition_key="condition",
            thresholds=_thresholds(),
            formats=("png",),
        )

    assert [str(w.message) for w in caught if "matplotlib" in str(w.category).lower()] == []


def test_typeset_table_rejects_a_column_absent_from_the_body() -> None:
    """A typo in a column key fails at construction, not silently in the render."""
    with pytest.raises(QCTableError, match="absent from the body"):
        TypesetTable(
            title="Table X",
            columns=(Column("missing", "Missing", fmt="int"),),
            body=pd.DataFrame({"present": [1, 2]}),
        )


# ---------------------------------------------------------------------------
# By cell type
# ---------------------------------------------------------------------------


def test_cell_type_keys_prefer_the_named_column_over_the_convention() -> None:
    """An explicit key wins; a missing one is None, not a silent substitution."""
    obs, _, _ = _annotated_cohort()
    obs["ref_state"] = "LEC"

    assert resolve_cell_type_keys(obs) == ("cell_type", "cell_type_granular")
    assert resolve_cell_type_keys(obs, cell_type_key="ref_state") == (
        "ref_state",
        "cell_type_granular",
    )
    # Named but absent resolves to None rather than falling back, so a typo
    # surfaces as a skipped table instead of a table grouped by something else.
    assert resolve_cell_type_keys(obs, cell_type_key="lineage")[0] is None
    # An unannotated object is a normal case: QC runs before annotation too.
    assert resolve_cell_type_keys(obs[["sample_id"]]) == (None, None)


def test_cell_type_summary_counts_the_pre_filter_population() -> None:
    """Every cell that entered QC is counted, and subtotals equal their members.

    The failure this locks out is the one that made the first by-cell-type table
    useless: labels read off the FILTERED object, so every cell type reported
    zero removed and the removals piled into an unlabelled row.
    """
    frame = _annotated_frame()
    rows, group_totals = summarize_by_cell_type(frame)

    assert int(rows["cells_in"].sum()) == len(frame)
    assert int(rows["cells_removed"].sum()) == int((~frame["keep"]).sum())
    assert group_totals is not None
    for group, subtotal in group_totals.iterrows():
        members = rows.loc[rows["group"] == group]
        assert int(subtotal["cells_in"]) == int(members["cells_in"].sum())
        assert int(subtotal["cells_removed"]) == int(members["cells_removed"].sum())

    # Attrition is genuinely uneven, which is the whole point of the breakdown.
    depleted = rows.set_index("cell_type").loc["LEC PROX1 hi"]
    spared = rows.set_index("cell_type").loc["Fibroblast CCL19+"]
    assert float(depleted["pct_removed"]) > 50.0
    assert float(spared["pct_removed"]) == 0.0


def test_cell_type_medians_describe_the_retained_cells() -> None:
    """The median columns describe the analysed dataset, not the input to QC."""
    frame = _annotated_frame()
    retained, _ = summarize_by_cell_type(frame, median_population="retained")
    everything, _ = summarize_by_cell_type(frame, median_population="all")

    depleted = "LEC PROX1 hi"
    kept_median = float(retained.set_index("cell_type").loc[depleted, "n_genes_by_counts"])
    all_median = float(everything.set_index("cell_type").loc[depleted, "n_genes_by_counts"])
    # The type is mostly removed low-complexity cells, so dropping them must move
    # the median up; equality would mean the population argument did nothing.
    assert kept_median > all_median

    # Counts are pre-filter either way — only the medians follow the argument.
    pd.testing.assert_series_equal(retained["cells_removed"], everything["cells_removed"])


def test_cell_types_read_largest_first_with_unlabelled_last() -> None:
    """Size order, not the alphabet, and the bookkeeping row at the bottom."""
    frame = _annotated_frame()
    rows, group_totals = summarize_by_cell_type(frame)

    assert list(group_totals.index) == ["LEC", "Fibroblast", UNLABELLED]
    lec = rows.loc[rows["group"] == "LEC", "cell_type"].tolist()
    assert lec == ["LEC ACKR4+", "LEC PROX1 hi", "Sweat gland"]
    assert rows["cell_type"].iloc[-1] == UNLABELLED

    # And the unlabelled row sorts last on size alone too: it is not a population.
    inflated = rows.copy()
    inflated.loc[inflated["cell_type"] == UNLABELLED, "cells_in"] = 10_000
    reordered = order_cell_types(inflated, name="cell_type", group="group")
    assert reordered["cell_type"].iloc[-1] == UNLABELLED


def test_unlabelled_cells_are_named_and_never_dropped() -> None:
    """Cells annotation never saw stay countable instead of vanishing."""
    frame = _annotated_frame()
    rows, _ = summarize_by_cell_type(frame)

    unlabelled = rows.set_index("cell_type").loc[UNLABELLED]
    assert int(unlabelled["cells_in"]) == 10
    # Not spelled "nan": that string sorts in as though it were a cell type.
    assert "nan" not in set(rows["cell_type"])
    assert int(rows["cells_in"].sum()) == len(frame)


def test_display_rows_interleave_subtotals_with_their_members() -> None:
    """Draw order is subtotal-then-members, exactly as the table prints it."""
    frame = _annotated_frame()
    rows, group_totals = summarize_by_cell_type(frame)
    display = cell_type_display_rows(rows, group_totals)

    assert display["label"].tolist() == [
        "LEC",
        "LEC ACKR4+",
        "LEC PROX1 hi",
        "Sweat gland",
        "Fibroblast",
        "Fibroblast CCL19+",
        UNLABELLED,
    ]
    assert display["is_group"].tolist() == [True, False, False, False, True, False, True]
    # Subtotal rows carry their group's own aggregate, not a repeat of a member.
    lec = display.loc[display["label"] == "LEC"].iloc[0]
    assert int(lec["cells_in"]) == int(group_totals.loc["LEC", "cells_in"])

    # A flat list is a valid shape too: no coarse label means no subtotal rows.
    flat_rows, flat_totals = summarize_by_cell_type(frame.drop(columns=["cell_type"]))
    assert flat_totals is None
    assert not cell_type_display_rows(flat_rows, flat_totals)["is_group"].any()


def test_cell_type_figure_and_table_agree_row_for_row() -> None:
    """The figure and Table 3 are two renderings of one summary, or they are a bug.

    A figure whose rows are ordered differently from the table beside it gets read
    as disagreeing about the data, so the ordering is shared code and this is the
    test that keeps it shared.
    """
    obs, metrics, decisions = _annotated_cohort()
    frame = _annotated_frame()

    table = build_cell_type_table(cell_metrics=metrics, cell_decisions=decisions, obs=obs)
    rows, group_totals = summarize_by_cell_type(frame)
    display = cell_type_display_rows(rows, group_totals)

    assert table is not None
    assert (
        table.body["cell_type"].tolist() == display.loc[~display["is_group"], "cell_type"].tolist()
    )
    for column in ("cells_in", "cells_removed", "cells_kept"):
        assert table.body[column].tolist() == (display.loc[~display["is_group"], column].tolist())
    # And the subtotal bars match the bands.
    assert list(table.group_totals.index) == display.loc[display["is_group"], "label"].tolist()


def test_cell_type_figure_is_written_without_warnings(tmp_path: Path) -> None:
    """One figure, three panels, and no matplotlib complaints about the layout."""
    frame = _annotated_frame()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = write_qc_cell_type_figure(frame, tmp_path, formats=("png",))

    assert [p.name for p in paths] == ["qc_cell_type.png"]
    assert paths[0].stat().st_size > 0
    assert [str(w.message) for w in caught if "matplotlib" in str(w.category).lower()] == []


def test_cell_type_figure_is_skipped_rather_than_stubbed(tmp_path: Path) -> None:
    """No labels, or one row, means no figure — not an axes with one bar in it."""
    assert write_qc_cell_type_figure(_frame(), tmp_path) == []

    frame = _annotated_frame()
    single = frame.copy()
    single["cell_type"] = "LEC"
    single["cell_type_granular"] = "LEC"
    assert write_qc_cell_type_figure(single, tmp_path) == []
    assert not list(tmp_path.glob("qc_cell_type*"))


def test_cell_type_panel_set_includes_the_by_cell_type_figure(tmp_path: Path) -> None:
    """An annotated cohort gets the extra figure; an unannotated one does not."""
    annotated = write_qc_panels(_annotated_frame(), tmp_path / "annotated")
    plain = write_qc_panels(_frame(), tmp_path / "plain")

    assert "qc_cell_type.png" in {p.name for p in annotated}
    assert "qc_cell_type.png" not in {p.name for p in plain}
    # The rest of the sheet is unaffected by the addition.
    assert {p.name for p in plain} < {p.name for p in annotated}


def test_small_cell_types_are_drawn_but_marked() -> None:
    """A five-cell population is plotted, and flagged as too small to read as a rate."""
    frame = _annotated_frame()
    rows, group_totals = summarize_by_cell_type(frame)
    display = cell_type_display_rows(rows, group_totals)

    tiny = display.loc[display["label"] == "Sweat gland"].iloc[0]
    assert int(tiny["cells_in"]) < SMALL_CELL_TYPE

    figure, axes = matplotlib.pyplot.subplots()
    from cellquorum.visualization.qc.panels import RULE_FILL, plot_cell_type_attrition

    plot_cell_type_attrition(axes, display)
    # Every row gets a bar, the tiny ones included: dropping them would hide that
    # the population exists, which is the opposite of what a QC panel is for.
    bars = [
        patch
        for patch in axes.patches
        if isinstance(patch, matplotlib.patches.Rectangle) and patch.get_height() == 0.74
    ]
    assert len(bars) == len(display)
    # Drawn, but faded: five cells cannot carry a percentage, and a full-strength
    # bar beside the real populations invites reading one as a rate.
    tiny_at = int(display.index[display["label"] == "Sweat gland"][0])
    real_at = int(display.index[display["label"] == "LEC PROX1 hi"][0])
    assert bars[tiny_at].get_facecolor() == matplotlib.colors.to_rgba(RULE_FILL)
    assert bars[real_at].get_facecolor() != bars[tiny_at].get_facecolor()
    matplotlib.pyplot.close(figure)


def test_html_report_omits_the_per_sample_section_for_a_single_sample() -> None:
    """One sample makes the per-sample table the funnel restated. Drop it."""
    obs, metrics, decisions = _cohort()
    table = build_sample_qc_table(
        cell_metrics=metrics,
        cell_decisions=decisions,
        obs=obs.drop(columns=["sample_id"]),
        sample_key="sample_id",
    )
    rules = build_rule_attribution_table(cell_decisions=decisions, thresholds=_thresholds())

    html = render_qc_html_report(sample_table=table, rule_table=rules, thresholds=_thresholds())

    assert "Per-sample attrition" not in html
    # The rule attribution is still worth reporting for one library.
    assert "Which rule removed what" in html


def test_the_donor_row_carries_an_attrition_dumbbell() -> None:
    """The reviewer's question belongs beside the metrics that answer it.

    "Did QC take more from the diseased arm, donor by donor?" is the paired test a
    reviewer asks for, and the dumbbell is already the right shape for it: one
    point per donor per arm, sorted by the within-donor difference, with the
    signed-rank p-value in the empty band. Drawing it anywhere but next to the
    metric panels would separate the effect from its cause.
    """
    import matplotlib.pyplot as plt

    table = summarize_by_sample(_frame())
    figure, axes = plt.subplots()
    plot_paired_dumbbell(
        axes,
        table,
        "pct_removed",
        case_label="Lymphedema",
        xlabel="Cells removed by QC (%)",
        title="QC attrition by donor",
    )

    # The wording is the caller's, not the metric menu's fallback of "pct removed".
    assert axes.get_xlabel() == "Cells removed by QC (%)"
    assert any("attrition" in title.get_text().lower() for title in _axes_titles(axes))

    # One dumbbell per donor that contributed both arms.
    assert len(axes.get_yticklabels()) == len(DONORS)
    plt.close(figure)


def test_attrition_is_not_offered_as_a_cell_metric() -> None:
    """Removal is what the filter DID, not a property measured from the cell.

    The metric menu drives the sample matrix and the per-metric histograms, both
    of which describe cells. A column of per-sample removal percentages in either
    would be a category error, and on the matrix it would be scaled against
    mitochondrial percentages as though the two were comparable.
    """
    assert "pct_removed" not in METRIC_LABELS
    assert "pct_removed" not in MATRIX_METRICS
