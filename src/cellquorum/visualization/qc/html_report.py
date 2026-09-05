"""Self-contained HTML QC report.

The QC stage already writes every number a reader could want, spread across
``cell_metrics.csv``, ``cell_decisions.csv``, ``gene_decisions.csv`` and
``thresholds.csv``. Reading the filter off those means joining four tables by
hand, which is why the 503-cell drop on the 2026-09-01 VEC run went unnoticed
for a day. This module answers "what did QC do to my cohort" on one page:
the cohort funnel, per-sample attrition, which rule removed what, and the
thresholds that were actually applied.

One file, no assets, no CDN — it opens from a filesystem path and survives being
emailed to a collaborator. Sorting is a few lines of inline JavaScript rather
than a table library for the same reason.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError

# Condition chip colours. Two hues, far apart in both hue and lightness, so the
# chips stay distinguishable under deuteranopia and in greyscale print.
_NORMAL_CHIP = "#24608F"
_CASE_CHIP = "#C41E3A"
_NEUTRAL = "#6B7280"
_KEEP_BAR = "#4C9A6A"
_FAIL_BAR = "#C25B5B"


class QCHTMLReportError(CellQuorumDataError):
    """Report failures building the HTML QC report."""


_CSS = """
:root {
  --ink: #1F2933; --muted: #6B7280; --rule: #E3E7EB; --bg: #FFFFFF;
  --band: #F7F9FA; --accent: #24608F;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 2rem 4rem; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 0.25rem; }
h2 {
  font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.09em; color: var(--muted); margin: 2.75rem 0 0.75rem;
  padding-bottom: 0.4rem; border-bottom: 1px solid var(--rule);
}
.sub { color: var(--muted); font-size: 0.85rem; margin: 0 0 2rem; }
.sub code { background: var(--band); padding: 0.1rem 0.35rem; border-radius: 3px; }

/* Cohort funnel: the headline the CSVs bury. */
.funnel { display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; margin: 0 0 0.5rem; }
.stat {
  flex: 1 1 0; min-width: 128px; padding: 0.9rem 1.1rem;
  border-left: 3px solid var(--rule);
}
.stat:first-child { border-left-color: var(--accent); }
.stat .n { font-size: 1.65rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.stat .k {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--muted); margin-top: 0.15rem;
}
.stat.warn .n { color: #B45309; }

table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
caption { caption-side: bottom; text-align: left; color: var(--muted);
          font-size: 0.78rem; padding-top: 0.6rem; }
th, td { padding: 0.42rem 0.6rem; text-align: right; white-space: nowrap; }
th:first-child, td:first-child, th.txt, td.txt { text-align: left; }
thead th {
  position: sticky; top: 0; background: var(--bg); z-index: 2;
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--muted);
  border-bottom: 1.5px solid var(--ink); cursor: pointer; user-select: none;
}
thead th:hover { color: var(--ink); }
thead th::after { content: " \\2195"; opacity: 0.25; font-weight: 400; }
thead th.nosort { cursor: default; }
thead th.nosort::after { content: ""; }
tbody tr:nth-child(even) { background: var(--band); }
tbody tr.total { font-weight: 600; border-top: 1.5px solid var(--ink); background: var(--bg); }
tbody td { border-bottom: 1px solid var(--rule); }

.chip {
  display: inline-block; padding: 0.06rem 0.45rem; border-radius: 9px;
  font-size: 0.72rem; font-weight: 600; color: #FFF;
}
/* Attrition bar: length is the signal, the number is the precision. */
.bar { display: flex; align-items: center; gap: 0.45rem; justify-content: flex-end; }
.bar .track {
  width: 74px; height: 7px; border-radius: 4px; background: var(--rule); overflow: hidden;
}
.bar .fill { height: 100%; border-radius: 4px; }
.bar .val { min-width: 3.1rem; }
.muted { color: var(--muted); }
.nil { color: var(--muted); }
footer { margin-top: 3.5rem; padding-top: 1rem; border-top: 1px solid var(--rule);
         color: var(--muted); font-size: 0.78rem; }
"""

# Click a header to sort. Numeric columns carry data-v so "1,809" and "13.2%"
# sort as numbers rather than strings; the TOTAL row is pinned to the bottom.
_JS = """
document.querySelectorAll('table').forEach(function (table) {
  table.querySelectorAll('thead th:not(.nosort)').forEach(function (th, col) {
    var asc = true;
    th.addEventListener('click', function () {
      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      var totals = rows.filter(function (r) { return r.classList.contains('total'); });
      var data = rows.filter(function (r) { return !r.classList.contains('total'); });
      var key = function (row) {
        var cell = row.cells[col];
        if (!cell) { return ''; }
        var raw = cell.getAttribute('data-v');
        return raw === null ? cell.textContent.trim() : parseFloat(raw);
      };
      data.sort(function (a, b) {
        var x = key(a), y = key(b);
        if (typeof x === 'number' && typeof y === 'number') {
          if (isNaN(x)) { return 1; }
          if (isNaN(y)) { return -1; }
          return asc ? x - y : y - x;
        }
        return asc ? String(x).localeCompare(String(y)) : String(y).localeCompare(String(x));
      });
      asc = !asc;
      data.concat(totals).forEach(function (r) { body.appendChild(r); });
    });
  });
});
"""


def _esc(value: object) -> str:
    """HTML-escape a value for text content."""
    return html.escape("" if value is None else str(value))


def _num(value: object, digits: int = 0) -> str:
    """Format a number with thousands separators, or an em dash when absent."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '<span class="nil">&mdash;</span>'
    return f"{float(value):,.{digits}f}"


def _cell(value: object, digits: int = 0) -> str:
    """A right-aligned numeric cell that sorts numerically."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '<td data-v="NaN"><span class="nil">&mdash;</span></td>'
    return f'<td data-v="{float(value)}">{float(value):,.{digits}f}</td>'


def _pct_bar(pct: float, *, warn_above: float = 25.0) -> str:
    """A proportion rendered as bar plus number, so scale reads at a glance."""
    if pct is None or not np.isfinite(pct):
        return '<td data-v="NaN"><span class="nil">&mdash;</span></td>'
    width = max(0.0, min(100.0, float(pct)))
    colour = _FAIL_BAR if pct >= warn_above else _NEUTRAL
    return (
        f'<td data-v="{float(pct)}"><span class="bar">'
        f'<span class="track"><span class="fill" style="width:{width:.1f}%;'
        f'background:{colour}"></span></span>'
        f'<span class="val">{pct:.1f}%</span></span></td>'
    )


def _chip(label: str, *, case_label: str | None) -> str:
    """A condition label as a coloured chip."""
    colour = _CASE_CHIP if case_label and str(label) == str(case_label) else _NORMAL_CHIP
    return f'<span class="chip" style="background:{colour}">{_esc(label)}</span>'


# Metrics summarised as medians: QC distributions are skewed and a mean is
# dragged by the same outliers the filter exists to remove. Shared by every
# grouping — per sample, per cell type — so the numbers agree across tables.
_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("total_counts", "median_umi"),
    ("n_genes_by_counts", "median_genes"),
    ("pct_counts_mito", "median_pct_mito"),
    ("pct_counts_ribo", "median_pct_ribo"),
)


def build_qc_cell_frame(
    *,
    cell_metrics: pd.DataFrame,
    cell_decisions: pd.DataFrame,
    obs: pd.DataFrame,
    labels: dict[str, str | None],
) -> pd.DataFrame:
    """Join decisions, metrics and chosen obs labels on the PRE-filter index.

    Every QC summary starts here. ``cell_decisions`` is indexed by every input
    cell, so joining to it — rather than to the surviving object — is what makes
    ``cells_removed`` non-zero.

    Args:
        cell_metrics: Per-cell QC metrics, indexed by input cell.
        cell_decisions: Per-cell QC decisions, indexed by input cell.
        obs: Cell metadata carrying the label columns.
        labels: ``{output name: obs column}``. Entries whose column is absent
            from ``obs`` are dropped, so callers can ask for optional labels.

    Returns:
        A frame with ``keep``, the resolved label columns and any available
        median metric columns.

    Raises:
        QCHTMLReportError: If the keep/fail decision is absent.
    """

    if "keep" not in cell_decisions.columns:
        raise QCHTMLReportError(
            "cell_decisions must carry a 'keep' column. "
            f"Present: {sorted(cell_decisions.columns)}"
        )

    from cellquorum.visualization.qc.panels import label_series

    frame = pd.DataFrame(index=cell_decisions.index)
    frame["keep"] = cell_decisions["keep"].astype(bool)
    for name, column in labels.items():
        if column and column in obs.columns:
            # Named, not stringified: an obs off a filtered object has a hole for
            # every removed cell, and astype(str) would spell that hole "nan" and
            # then sort it in as though it were a sample or a cell type.
            frame[name] = label_series(obs[column], frame.index)
    for source, target in _METRIC_COLUMNS:
        if source in cell_metrics.columns:
            frame[target] = pd.to_numeric(
                cell_metrics[source].reindex(frame.index), errors="coerce"
            )
    return frame


def summarize_qc_rows(
    frame: pd.DataFrame,
    *,
    label: str,
    name: str,
    carry: tuple[str, ...] = (),
    median_population: str = "all",
) -> pd.DataFrame:
    """Aggregate a QC cell frame into one attrition row per label value.

    Args:
        frame: Output of :func:`build_qc_cell_frame`.
        label: Column to group by.
        name: Name the grouping column takes in the result.
        carry: Other label columns to carry through, reported as ``"mixed"``
            when a group spans several values.
        median_population: ``"all"`` or ``"retained"`` — which cells the metric
            medians describe. Attrition counts are always pre-filter.

    Returns:
        One row per group value, in first-appearance order.

    Raises:
        QCHTMLReportError: If ``median_population`` is not a known value.
    """

    if median_population not in {"all", "retained"}:
        raise QCHTMLReportError(
            f"median_population must be 'all' or 'retained', got {median_population!r}."
        )

    rows: list[dict[str, object]] = []
    for value, chunk in frame.groupby(label, observed=True, dropna=False, sort=True):
        row: dict[str, object] = {name: str(value)}
        for other in carry:
            if other in chunk.columns:
                values = chunk[other].dropna().unique()
                row[other] = str(values[0]) if len(values) == 1 else "mixed"
        rows.append({**row, **summarize_qc_pool(chunk, median_population=median_population)})
    columns = [name, *(c for c in carry if c in frame.columns)]
    return pd.DataFrame(rows, columns=None if rows else columns)


def summarize_qc_pool(frame: pd.DataFrame, *, median_population: str = "all") -> dict[str, object]:
    """Aggregate a QC cell frame into one pooled row.

    Medians pool the cells rather than averaging per-group medians, which would
    weight a 21-cell group like a 490-cell one.

    Args:
        frame: Output of :func:`build_qc_cell_frame`, or a subset of it.
        median_population: ``"all"`` or ``"retained"``.

    Returns:
        Attrition counts and metric medians, without any label column.
    """

    n_in = int(len(frame))
    n_keep = int(frame["keep"].sum()) if n_in else 0
    row: dict[str, object] = {
        "cells_in": n_in,
        "cells_kept": n_keep,
        "cells_removed": n_in - n_keep,
        "pct_removed": 100.0 * (n_in - n_keep) / n_in if n_in else float("nan"),
    }
    described = frame.loc[frame["keep"]] if median_population == "retained" else frame
    for _, target in _METRIC_COLUMNS:
        if target in frame.columns:
            # NaN when a group lost every cell, which is the honest value: it
            # contributed nothing to the analysed dataset.
            row[target] = float(described[target].median()) if len(described) else float("nan")
    return row


def build_sample_qc_table(
    *,
    cell_metrics: pd.DataFrame,
    cell_decisions: pd.DataFrame,
    obs: pd.DataFrame,
    sample_key: str,
    donor_key: str | None = None,
    condition_key: str | None = None,
    case_label: str | None = None,
    median_population: str = "all",
) -> pd.DataFrame:
    """
    Build the per-sample QC attrition and metric table.

    Aggregation happens on the PRE-filter index. ``cell_decisions`` is indexed by
    every input cell, so joining metrics to it — rather than to the surviving
    object — is what makes ``cells_removed`` non-zero.

    Args:
        cell_metrics: Per-cell QC metrics, indexed by input cell.
        cell_decisions: Per-cell QC decisions, indexed by input cell.
        obs: Cell metadata carrying the sample/donor/condition columns.
        sample_key: Column identifying the library/sample.
        donor_key: Optional column identifying the donor.
        condition_key: Optional column identifying the condition.
        case_label: Optional condition value marking the case arm, used to put the
            control sample first within each donor.
        median_population: Which cells the metric medians describe. ``"all"``
            summarises every input cell, which explains why cells were removed;
            ``"retained"`` summarises survivors only, which describes the dataset
            the analysis actually runs on. Attrition counts are unaffected — those
            are always pre-filter.

    Returns:
        One row per sample plus a trailing ``TOTAL`` row. An object with no sample
        column collapses to the ``TOTAL`` row alone rather than raising: the
        cohort funnel and rule attribution are still worth reporting for a single
        unlabelled library, and a hard failure here would cost the whole report.

    Raises:
        QCHTMLReportError: If the keep/fail decision is absent.
    """

    frame = build_qc_cell_frame(
        cell_metrics=cell_metrics,
        cell_decisions=cell_decisions,
        obs=obs,
        labels={"sample": sample_key, "donor": donor_key, "condition": condition_key},
    )
    if "sample" not in frame.columns:
        frame["sample"] = "all cells"

    table = summarize_qc_rows(
        frame,
        label="sample",
        name="sample",
        carry=("donor", "condition"),
        median_population=median_population,
    )

    # Order by donor then condition so paired samples sit adjacent — the layout
    # that makes a within-donor quality imbalance visible. Shared with the figure
    # panels so a reader comparing table to figure sees the same row order.
    from cellquorum.visualization.qc.panels import order_samples

    table = order_samples(table, case_label=case_label)

    total: dict[str, object] = {"sample": "TOTAL"}
    for label in ("donor", "condition"):
        if label in table.columns:
            total[label] = ""
    total.update(summarize_qc_pool(frame, median_population=median_population))
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    return table


def build_rule_attribution_table(
    *,
    cell_decisions: pd.DataFrame,
    thresholds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the per-rule removal table, with each rule's threshold alongside.

    Rules overlap — a low-complexity cell often fails several at once — so the
    per-rule counts deliberately do not sum to the total removed. The table
    reports each rule's marginal contribution (cells only that rule caught)
    next to its gross count, which is what tells you whether a rule is doing
    work no other rule does.

    Args:
        cell_decisions: Per-cell decisions with one boolean column per rule.
        thresholds: Optional threshold table to attach bounds from.

    Returns:
        One row per rule, ordered by gross cells failed, descending.
    """

    # Graded QC records the driver per cell, so attribution needs no rule columns. Preferred
    # when present: under grading there are no threshold rules to attribute to, and a table of
    # rule names would describe a system that did not run.
    if "qc_primary_driver" in cell_decisions.columns:
        from cellquorum.visualization.qc.graded import graded_attribution_table

        graded = graded_attribution_table(cell_decisions)
        if not graded.empty:
            return graded

    reserved = {"keep", "fail_any_qc", "failed_rules"}
    rule_columns = [
        column
        for column in cell_decisions.columns
        if column not in reserved and cell_decisions[column].dropna().isin([True, False]).all()
    ]
    if not rule_columns:
        return pd.DataFrame(columns=["rule", "cells_failed", "pct_of_input", "only_this_rule"])

    flags = cell_decisions[rule_columns].fillna(False).astype(bool)
    n_input = int(len(cell_decisions))
    n_rules_failed = flags.sum(axis=1)

    bounds: dict[str, tuple[float | None, float | None]] = {}
    if thresholds is not None and "rule_name" in thresholds.columns:
        for _, record in thresholds.iterrows():
            lower = record.get("lower")
            upper = record.get("upper")
            bounds[str(record["rule_name"])] = (
                float(lower) if pd.notna(lower) else None,
                float(upper) if pd.notna(upper) else None,
            )

    rows = []
    for rule in rule_columns:
        failed = flags[rule]
        lower, upper = bounds.get(rule, (None, None))
        rows.append(
            {
                "rule": rule,
                "lower": lower,
                "upper": upper,
                "cells_failed": int(failed.sum()),
                "pct_of_input": 100.0 * int(failed.sum()) / n_input if n_input else float("nan"),
                # Cells this rule alone caught: remove it and these cells survive.
                "only_this_rule": int((failed & n_rules_failed.eq(1)).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("cells_failed", ascending=False).reset_index(drop=True)


def render_qc_html_report(
    *,
    sample_table: pd.DataFrame,
    rule_table: pd.DataFrame,
    thresholds: pd.DataFrame | None = None,
    gene_summary: dict[str, int] | None = None,
    project: str = "CellQuorum",
    floors: dict[str, int | None] | None = None,
    case_label: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    """
    Render the QC report as a single self-contained HTML document.

    Args:
        sample_table: Output of :func:`build_sample_qc_table`.
        rule_table: Output of :func:`build_rule_attribution_table`.
        thresholds: Optional applied-threshold table.
        gene_summary: Optional gene-level counts (``n_genes``, ``n_genes_kept``).
        project: Project or run name for the heading.
        floors: The applied absolute floors, so a reader knows what could remove a barcode.
            Every floor ``None`` means nothing was removed at all.
        case_label: Condition treated as the case arm, for chip colouring.
        generated_at: Timestamp override, for reproducible output in tests.

    Returns:
        A complete HTML document.
    """

    stamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    total_row = sample_table.loc[sample_table["sample"].eq("TOTAL")]
    n_in = int(total_row["cells_in"].iloc[0]) if len(total_row) else 0
    n_kept = int(total_row["cells_kept"].iloc[0]) if len(total_row) else 0
    n_removed = n_in - n_kept
    pct_removed = 100.0 * n_removed / n_in if n_in else 0.0
    n_samples = int(len(sample_table) - len(total_row))

    body: list[str] = ['<div class="wrap">']
    body.append(f"<h1>QC report &mdash; {_esc(project)}</h1>")
    # QC has no mode: the floors are the only thing that can remove a barcode, and graded
    # adjudication assigns permissions without deleting. So the honest header states the floors
    # rather than a mode name, and says plainly when none of them is set.
    active = {name: value for name, value in (floors or {}).items() if value is not None}
    floor_text = (
        " &middot; ".join(
            f"<code>{_esc(name)} &ge; {value:,}</code>" for name, value in active.items()
        )
        if active
        else "<code>no floors set</code> &middot; no barcode was removed"
    )
    body.append(f'<p class="sub">Generated {stamp} &middot; {floor_text}</p>')

    # Cohort funnel.
    worst = sample_table.loc[~sample_table["sample"].eq("TOTAL")]
    worst_pct = float(worst["pct_removed"].max()) if len(worst) else float("nan")
    worst_name = (
        str(worst.loc[worst["pct_removed"].idxmax(), "sample"])
        if len(worst) and np.isfinite(worst_pct)
        else "—"
    )
    body.append('<div class="funnel">')
    body.append(
        f'<div class="stat"><div class="n">{_num(n_in)}</div><div class="k">Cells in</div></div>'
    )
    body.append(
        f'<div class="stat"><div class="n">{_num(n_kept)}</div><div class="k">Retained</div></div>'
    )
    body.append(
        f'<div class="stat warn"><div class="n">{_num(n_removed)}</div>'
        f'<div class="k">Removed ({pct_removed:.1f}%)</div></div>'
    )
    if n_samples > 1:
        body.append(
            f'<div class="stat"><div class="n">{_num(n_samples)}</div>'
            '<div class="k">Samples</div></div>'
        )
    if n_samples > 1 and np.isfinite(worst_pct):
        body.append(
            f'<div class="stat"><div class="n">{worst_pct:.1f}%</div>'
            f'<div class="k">Worst sample &middot; {_esc(worst_name)}</div></div>'
        )
    body.append("</div>")

    # Per-sample attrition. Skipped for a single unlabelled library: the table
    # would be the funnel restated as one row plus its own total.
    if n_samples > 1:
        body.append("<h2>Per-sample attrition</h2>")
        optional = [c for c in ("donor", "condition") if c in sample_table.columns]
        metric_labels = [
            ("median_umi", "Median UMI", 0),
            ("median_genes", "Median genes", 0),
            ("median_pct_mito", "Median %mito", 2),
            ("median_pct_ribo", "Median %ribo", 2),
        ]
        present_metrics = [m for m in metric_labels if m[0] in sample_table.columns]
        header = [
            "Sample",
            *[c.title() for c in optional],
            "Cells in",
            "Kept",
            "Removed",
            "% removed",
        ]
        header += [label for _, label, _ in present_metrics]
        body.append("<table><thead><tr>")
        for index, name in enumerate(header):
            css = ' class="txt"' if index <= len(optional) else ""
            body.append(f"<th{css}>{_esc(name)}</th>")
        body.append("</tr></thead><tbody>")
        for _, record in sample_table.iterrows():
            is_total = str(record["sample"]) == "TOTAL"
            body.append('<tr class="total">' if is_total else "<tr>")
            body.append(f'<td class="txt">{_esc(record["sample"])}</td>')
            for column in optional:
                value = record.get(column, "")
                if column == "condition" and value:
                    body.append(f'<td class="txt">{_chip(str(value), case_label=case_label)}</td>')
                else:
                    body.append(f'<td class="txt">{_esc(value)}</td>')
            body.append(_cell(record.get("cells_in")))
            body.append(_cell(record.get("cells_kept")))
            body.append(_cell(record.get("cells_removed")))
            body.append(_pct_bar(record.get("pct_removed")))
            for key, _, digits in present_metrics:
                body.append(_cell(record.get(key), digits))
            body.append("</tr>")
        body.append(
            "</tbody><caption>Medians are computed on all input cells for that sample, "
            "before filtering. Click any header to sort.</caption></table>"
        )

    # Rule attribution.
    if len(rule_table):
        body.append("<h2>Which rule removed what</h2>")
        body.append(
            "<table><thead><tr>"
            '<th class="txt">Rule</th><th>Lower</th><th>Upper</th>'
            "<th>Cells failed</th><th>% of input</th><th>Only this rule</th>"
            "</tr></thead><tbody>"
        )
        for _, record in rule_table.iterrows():
            body.append("<tr>")
            body.append(f'<td class="txt">{_esc(record["rule"])}</td>')
            body.append(_cell(record.get("lower"), 2))
            body.append(_cell(record.get("upper"), 2))
            body.append(_cell(record.get("cells_failed")))
            body.append(_pct_bar(record.get("pct_of_input")))
            body.append(_cell(record.get("only_this_rule")))
            body.append("</tr>")
        body.append(
            "</tbody><caption>Rules overlap, so <em>cells failed</em> does not sum to the "
            "total removed. <em>Only this rule</em> counts cells no other rule caught &mdash; "
            "a rule with zero here is fully redundant.</caption></table>"
        )

    # Applied thresholds.
    if thresholds is not None and len(thresholds):
        body.append("<h2>Applied thresholds</h2>")
        columns = [
            c
            for c in ("axis", "metric", "rule_name", "lower", "upper", "source", "n_observations")
            if c in thresholds.columns
        ]
        body.append("<table><thead><tr>")
        for column in columns:
            css = ' class="txt"' if column in {"axis", "metric", "rule_name", "source"} else ""
            body.append(f"<th{css}>{_esc(column.replace('_', ' ').title())}</th>")
        body.append("</tr></thead><tbody>")
        for _, record in thresholds.iterrows():
            body.append("<tr>")
            for column in columns:
                value = record.get(column)
                if column in {"axis", "metric", "rule_name", "source"}:
                    body.append(f'<td class="txt">{_esc(value)}</td>')
                elif pd.isna(value):
                    body.append('<td class="nil">&mdash;</td>')
                else:
                    body.append(_cell(value, 3 if column in {"lower", "upper"} else 0))
            body.append("</tr>")
        body.append(
            "</tbody><caption>MAD bounds are data-derived; <code>fixed</code> bounds come "
            "from config. <em>N observations</em> is the population the bound was estimated "
            "on.</caption></table>"
        )

    # Gene filtering.
    if gene_summary:
        n_genes = int(gene_summary.get("n_genes", 0))
        n_genes_kept = int(gene_summary.get("n_genes_kept", 0))
        body.append("<h2>Gene filtering</h2>")
        body.append('<div class="funnel">')
        body.append(
            f'<div class="stat"><div class="n">{_num(n_genes)}</div>'
            '<div class="k">Genes in</div></div>'
        )
        body.append(
            f'<div class="stat"><div class="n">{_num(n_genes_kept)}</div>'
            '<div class="k">Retained</div></div>'
        )
        removed = n_genes - n_genes_kept
        pct = 100.0 * removed / n_genes if n_genes else 0.0
        body.append(
            f'<div class="stat warn"><div class="n">{_num(removed)}</div>'
            f'<div class="k">Removed ({pct:.1f}%)</div></div>'
        )
        body.append("</div>")

    body.append(
        "<footer>Written by CellQuorum QC. Every number here is derived from "
        "<code>cell_metrics.csv</code>, <code>cell_decisions.csv</code> and "
        "<code>thresholds.csv</code> in this directory &mdash; those tables remain "
        "canonical.</footer>"
    )
    body.append("</div>")

    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>QC report &mdash; {_esc(project)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"{''.join(body)}"
        f"<script>{_JS}</script></body></html>"
    )


def write_qc_html_report(
    output_path: str | Path,
    *,
    cell_metrics: pd.DataFrame,
    cell_decisions: pd.DataFrame,
    obs: pd.DataFrame,
    sample_key: str,
    donor_key: str | None = None,
    condition_key: str | None = None,
    thresholds: pd.DataFrame | None = None,
    gene_summary: dict[str, int] | None = None,
    project: str = "CellQuorum",
    floors: dict[str, int | None] | None = None,
    case_label: str | None = None,
) -> Path:
    """
    Build and write the HTML QC report.

    Args:
        output_path: Destination ``.html`` path.
        cell_metrics: Per-cell QC metrics, indexed by input cell.
        cell_decisions: Per-cell QC decisions, indexed by input cell.
        obs: Cell metadata carrying sample/donor/condition columns.
        sample_key: Column identifying the library/sample.
        donor_key: Optional donor column.
        condition_key: Optional condition column.
        thresholds: Optional applied-threshold table.
        gene_summary: Optional gene-level counts.
        project: Project or run name for the heading.
        floors: The applied absolute floors.
        case_label: Condition treated as the case arm.

    Returns:
        The written path.
    """

    sample_table = build_sample_qc_table(
        cell_metrics=cell_metrics,
        cell_decisions=cell_decisions,
        obs=obs,
        sample_key=sample_key,
        donor_key=donor_key,
        condition_key=condition_key,
        case_label=case_label,
    )
    rule_table = build_rule_attribution_table(
        cell_decisions=cell_decisions,
        thresholds=thresholds,
    )
    document = render_qc_html_report(
        sample_table=sample_table,
        rule_table=rule_table,
        thresholds=thresholds,
        gene_summary=gene_summary,
        project=project,
        floors=floors,
        case_label=case_label,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


__all__ = [
    "QCHTMLReportError",
    "build_rule_attribution_table",
    "build_sample_qc_table",
    "render_qc_html_report",
    "write_qc_html_report",
]
