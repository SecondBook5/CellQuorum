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

import base64
import html
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.visualization.qc.summarise import (
    order_samples,
    summarize_qc_pool,
    summarize_qc_rows,
)

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
/* Figures are inlined as data URIs, so they scale to the container rather than to their pixel
   size — a 2,700px-wide raincloud would otherwise force a horizontal scrollbar on the page. */
figure.fig { margin: 1.4rem 0; }
figure.fig img {
  width: 100%; height: auto; display: block;
  border: 1px solid var(--rule); border-radius: 3px; background: #fff;
}
figure.fig figcaption {
  font-size: 0.78rem; color: var(--muted); margin-top: 0.35rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
ul.notes { color: var(--ink); font-size: 0.9rem; }
ul.notes li { margin: 0.3rem 0; }
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


def _as_float(value: object) -> float:
    """Coerce a table cell to a float, mapping absent or unparseable values to NaN.

    Row records come out of pandas typed as a wide union — a cell may be a float, a numpy
    scalar, a string or None — and the formatters below take a number. Coercing once here keeps
    the call sites honest about the fact that a missing cell is NaN rather than zero.
    """
    if value is None:
        return float("nan")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _num(value: object, digits: int = 0) -> str:
    """Format a number with thousands separators, or an em dash when absent."""
    number = _as_float(value)
    if not np.isfinite(number):
        return '<span class="nil">&mdash;</span>'
    return f"{number:,.{digits}f}"


def _cell(value: object, digits: int = 0) -> str:
    """A right-aligned numeric cell that sorts numerically."""
    number = _as_float(value)
    if not np.isfinite(number):
        return '<td data-v="NaN"><span class="nil">&mdash;</span></td>'
    return f'<td data-v="{number}">{number:,.{digits}f}</td>'


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
#: Graded verdict columns, read but never recomputed here: this report displays QC's
#: conclusion, it does not form one.
_STATE_COLUMN = "qc_state_initial"
_REASON_COLUMN = "qc_state_reason"

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


# ─── Graded adjudication: what the evidence concluded ────────────────────────────────

#: Reason codes emitted by the adjudicator, with the sentence a reader needs. Keyed on the
#: enum values in ``stages.qc.evidence.AdjudicationReason``; an unknown code still renders,
#: under its raw name, because a silently dropped row would understate the cohort.
_REASON_TEXT: dict[str, str] = {
    "no_concern": "No evidence family reached the concern bar.",
    "single_family_concern": "One family raised concern; concordance requires two.",
    "supporting_evidence_only": (
        "Only metabolic/stress evidence, which cannot establish damage alone."
    ),
    "concordant_severe_damage": "Severe evidence from independent families agreed.",
    "uninformative_barcode": "Too little signal to model at all.",
    "probable_multiplet": "Probably more than one cell; not a damage call.",
    "withheld_low_coverage": "Too few usable families to reach a verdict.",
}

_STATE_TEXT: dict[str, str] = {
    "core": "may fit the biological reference",
    "borderline": "retained and projected, never fits",
    "quarantine": "informs nothing",
}


def build_graded_state_table(obs: pd.DataFrame) -> pd.DataFrame:
    """Per-state counts, and the reasons that produced them.

    Returns an empty frame when the object carries no graded verdict, so a floors-only run
    renders without the section rather than with an empty one.
    """
    if _STATE_COLUMN not in obs.columns:
        return pd.DataFrame()

    state = obs[_STATE_COLUMN].astype(str)
    reason = (
        obs[_REASON_COLUMN].astype(str)
        if _REASON_COLUMN in obs.columns
        else pd.Series("", index=obs.index)
    )
    rows: list[dict[str, object]] = []
    for value in ("core", "borderline", "quarantine"):
        selected = state == value
        if not bool(selected.any()):
            continue
        within = reason[selected].value_counts()
        rows.append(
            {
                "state": value,
                "meaning": _STATE_TEXT.get(value, ""),
                "cells": int(selected.sum()),
                "pct": 100.0 * float(selected.mean()),
                "reasons": [(str(k), int(v)) for k, v in within.items()],
            }
        )
    return pd.DataFrame(rows)


def build_eligibility_table(obs: pd.DataFrame) -> pd.DataFrame:
    """Per-analysis FIT counts, which is what the verdict actually controls.

    The states are a summary; the masks are the mechanism. A reader who wants to know what QC
    *did* needs to see how many cells may fit each model, because that is the number every
    downstream cohort statistic is computed from.
    """
    columns = sorted(column for column in obs.columns if column.startswith("qc_fit_"))
    if not columns:
        return pd.DataFrame()
    total = int(len(obs))
    return pd.DataFrame(
        [
            {
                "analysis": column.removeprefix("qc_fit_").replace("_", " "),
                "may_fit": int(obs[column].sum()),
                "pct": 100.0 * float(obs[column].mean()) if total else 0.0,
            }
            for column in columns
        ]
    )


def embed_png(path: Path) -> str:
    """Inline one PNG as a data URI, so the report stays a single portable file.

    Linking would be smaller, but a run directory copied or archived without its figures then
    renders a page of broken images, and the report is the artefact people forward.
    """
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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
        metrics=[target for _, target in _METRIC_COLUMNS],
        label="sample",
        name="sample",
        carry=("donor", "condition"),
        median_population=median_population,
    )

    # Order by donor then condition so paired samples sit adjacent — the layout
    # that makes a within-donor quality imbalance visible. Shared with the figure
    # panels so a reader comparing table to figure sees the same row order.
    table = order_samples(table, case_label=case_label)

    total: dict[str, object] = {"sample": "TOTAL"}
    for label in ("donor", "condition"):
        if label in table.columns:
            total[label] = ""
    total.update(
        summarize_qc_pool(
            frame,
            median_population=median_population,
            metrics=[target for _, target in _METRIC_COLUMNS],
        )
    )
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
    graded_states: pd.DataFrame | None = None,
    eligibility: pd.DataFrame | None = None,
    figures: Sequence[Path] = (),
    notes: Sequence[str] = (),
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
        graded_states: Output of :func:`build_graded_state_table`. Omitted sections are skipped,
            so a floors-only run renders without an empty graded heading.
        eligibility: Output of :func:`build_eligibility_table`.
        figures: PNGs to inline, in reading order. Embedded as data URIs rather than linked, so
            the report survives being copied away from its run directory.
        notes: Findings that replace a figure — a metric with no spread, for instance — so the
            information is not lost with the panel that could not show it.
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

    # Graded adjudication. Placed directly under the funnel because it is the verdict: the
    # floors below removed 13 barcodes of 201,871 on the validation cohort, while this decides
    # what the other 201,858 are allowed to do.
    if graded_states is not None and not graded_states.empty:
        body.append("<h2>What the evidence concluded</h2>")
        body.append(
            '<p class="sub">Technical evidence, graded per cell within its own lineage, then '
            "combined. Two routes to quarantine only: a barcode too uninformative to model, or "
            "severe evidence from at least two <em>independent</em> families. A single extreme "
            "axis never condemns a cell on its own.</p>"
        )
        body.append('<div class="funnel">')
        for _, record in graded_states.iterrows():
            css = "stat" if record["state"] == "core" else "stat warn"
            body.append(
                f'<div class="{css}"><div class="n">{_num(record["cells"])}</div>'
                f'<div class="k">{_esc(record["state"])} '
                f'({float(record["pct"]):.1f}%)</div></div>'
            )
        body.append("</div>")

        body.append("<table><thead><tr>")
        for name, css in (
            ("State", ' class="txt"'),
            ("Cells", ""),
            ("%", ""),
            ("Why", ' class="txt"'),
        ):
            body.append(f"<th{css}>{name}</th>")
        body.append("</tr></thead><tbody>")
        for _, record in graded_states.iterrows():
            reasons = record["reasons"]
            for index, (code, count) in enumerate(reasons):
                body.append("<tr>")
                if index == 0:
                    body.append(
                        f'<td class="txt" rowspan="{len(reasons)}"><strong>'
                        f'{_esc(record["state"])}</strong><br>'
                        f'<span class="nil">{_esc(record["meaning"])}</span></td>'
                    )
                body.append(_cell(count))
                share = 100.0 * count / int(record["cells"]) if int(record["cells"]) else 0.0
                body.append(_pct_bar(share, warn_above=101.0))
                body.append(
                    f'<td class="txt"><code>{_esc(code)}</code> &mdash; '
                    f"{_esc(_REASON_TEXT.get(code, 'reason not described'))}</td>"
                )
                body.append("</tr>")
        body.append(
            "</tbody><caption>Percentages are within the state, not of the cohort. "
            "A cell has exactly one reason.</caption></table>"
        )

    if eligibility is not None and not eligibility.empty:
        body.append("<h2>What each analysis may be fitted on</h2>")
        body.append(
            '<p class="sub">The verdict assigns permissions rather than deleting cells, so this '
            "is what QC actually controls: every cohort statistic downstream is estimated from "
            "the cells counted here.</p>"
        )
        body.append(
            '<table><thead><tr><th class="txt">Analysis</th><th>May fit</th>'
            "<th>% of cells</th></tr></thead><tbody>"
        )
        for _, record in eligibility.iterrows():
            body.append("<tr>")
            body.append(f'<td class="txt">{_esc(record["analysis"])}</td>')
            body.append(_cell(record["may_fit"]))
            body.append(_pct_bar(float(record["pct"]), warn_above=101.0))
            body.append("</tr>")
        body.append(
            "</tbody><caption>A cell may be transformed by a model it is not permitted to fit "
            "&mdash; that is the point of the distinction.</caption></table>"
        )

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
            body.append(_pct_bar(_as_float(record.get("pct_removed"))))
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
            body.append(_pct_bar(_as_float(record.get("pct_of_input"))))
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

    # Evidence figures, inlined. These are the calibration outputs: distributions the bars were
    # read off, and the population-level view of what the verdict cost.
    if figures:
        body.append("<h2>Evidence and calibration figures</h2>")
        body.append(
            '<p class="sub">Distributions are on RAW metrics, pre-filter, because a figure drawn '
            "after filtering cannot justify the bound that produced it. Paired panels read "
            "control-arm-left within each donor.</p>"
        )
        for path in figures:
            if not path.exists():
                continue
            body.append('<figure class="fig">')
            body.append(f'<img alt="{_esc(path.stem)}" src="{embed_png(path)}">')
            body.append(f"<figcaption>{_esc(path.stem)}</figcaption>")
            body.append("</figure>")

    if notes:
        body.append('<h2>Notes</h2><ul class="notes">')
        for note in notes:
            body.append(f"<li>{_esc(note)}</li>")
        body.append("</ul>")

    body.append(
        "<footer>Written by CellQuorum QC. Every number here is read from "
        "<code>cell_metrics.csv</code>, <code>cell_floors.csv</code> and the graded "
        "evidence columns on <code>qc.h5ad</code> in this directory &mdash; those remain "
        "canonical, and this report recomputes none of them.</footer>"
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
    figures: Sequence[Path] = (),
    notes: Sequence[str] = (),
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
        figures: PNGs to inline, in reading order.
        notes: Findings that stand in for a figure that could not be drawn.

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
        # Read off obs rather than recomputed: this report displays the verdict QC reached, and
        # a report that re-derived it could disagree with the object it describes.
        graded_states=build_graded_state_table(obs),
        eligibility=build_eligibility_table(obs),
        figures=figures,
        notes=notes,
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
