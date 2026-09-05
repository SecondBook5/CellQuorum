"""Typeset QC tables, in the shape a manuscript wants them.

This is the ``gt``/``kableExtra`` half of the QC output: a Table 1 you can drop
into a paper, not a browsable dashboard. The rules are the ones typographers and
``booktabs`` agree on — horizontal rules only, never vertical; a heavy rule at
top and bottom with a light one under the headers; column groups announced by a
spanner and a partial rule beneath it; numbers right-aligned on a fixed number of
decimals so the digits form a column; footnote markers carrying the definitions
that would otherwise bloat a header.

One table specification (:class:`TypesetTable`) drives three renderers — HTML,
LaTeX and a rasterised figure — so the version a collaborator opens, the version
that compiles into the manuscript and the version that lands in a slide deck can
never disagree about the numbers.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.visualization.qc.html_report import (
    build_qc_cell_frame,
    build_rule_attribution_table,
    build_sample_qc_table,
    summarize_qc_pool,
    summarize_qc_rows,
)
from cellquorum.visualization.qc.panels import (
    GROUP_BAND,
    SMALL_CELL_TYPE,
    UNLABELLED,
    _rule_label,
    _threshold_lookup,
    order_cell_types,
    redundant_group_members,
    resolve_cell_type_keys,
)

# Rule humanisation, cell-type grouping and the subtotal tint all live with the
# panels, and the table must agree with the figures word for word and row for
# row: "Mitochondrial % > 6.1 (MAD)" in panel B and in Table 2, and the same
# cell types in the same order in Table 3 and the by-cell-type figure.
from cellquorum.visualization.qc.summarise import as_float

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Ink. Near-black rules rather than pure black, which prints harshly, and one
# grey for the subordinate text (subtitle, footnotes).
_INK = "#111418"
_SUBTLE = "#5B6470"
_BAND = GROUP_BAND  # the group-subtotal tint, achromatic by design
_MISSING = "—"  # em dash, the typesetter's "no value"


class QCTableError(CellQuorumDataError):
    """Raised when a typeset QC table cannot be built."""


@dataclass(frozen=True)
class Column:
    """One column of a typeset table.

    Args:
        key: Column name in the body frame.
        label: Header text.
        spanner: Optional column-group heading shared with adjacent columns.
        align: ``"left"`` for labels, ``"right"`` for numbers.
        fmt: One of ``"text"``, ``"int"``, ``"pct"``, ``"num"``.
        marker: Optional footnote marker appended to the header.
    """

    key: str
    label: str
    spanner: str | None = None
    align: str = "right"
    fmt: str = "int"
    marker: str | None = None


@dataclass(frozen=True)
class TypesetTable:
    """A table specification the three renderers share.

    Args:
        title: Table title, e.g. ``"Table 1. Cohort quality control"``.
        subtitle: Optional one-line description under the title.
        columns: Column specifications, in print order.
        body: Body rows. Must contain every column ``key``.
        total: Optional summary row, ruled off and set in bold.
        row_group: Optional body column whose runs become italic group headings
            (the donor, so a reader sees paired samples as one block).
        group_totals: Optional per-group subtotals, indexed by group label. When
            given, each group heading becomes a total bar carrying that group's
            own numbers, with its members indented beneath — so a cell type reads
            as one figure before it reads as a list of subtypes.
        footnotes: ``(marker, text)`` pairs, printed under the bottom rule.
        source_note: Optional final line — provenance, thresholds, timestamp.
        stem: File stem and LaTeX label suffix.
    """

    title: str
    columns: tuple[Column, ...]
    body: pd.DataFrame
    subtitle: str | None = None
    total: pd.Series | None = None
    row_group: str | None = None
    group_totals: pd.DataFrame | None = None
    footnotes: tuple[tuple[str, str], ...] = ()
    source_note: str | None = None
    stem: str = "qc_table"

    def __post_init__(self) -> None:
        missing = [column.key for column in self.columns if column.key not in self.body.columns]
        if missing:
            raise QCTableError(
                f"Table {self.stem!r} declares columns absent from the body: {missing}. "
                f"Body has: {sorted(self.body.columns)}"
            )
        if self.group_totals is not None and self.row_group is None:
            raise QCTableError(
                f"Table {self.stem!r} supplies group_totals but no row_group, so there "
                "are no group headings for the subtotals to sit on."
            )

    def group_total(self, group: str) -> pd.Series | None:
        """Return the subtotal row for a group heading, if one was supplied."""

        if self.group_totals is None or group not in self.group_totals.index:
            return None
        row = self.group_totals.loc[group]
        # A duplicated group label would make `.loc` return a frame rather than a row, which no
        # caller handles. Reporting it beats silently rendering the first match.
        if isinstance(row, pd.DataFrame):
            raise QCTableError(
                f"Group {group!r} appears {len(row)} times in the subtotal table; group labels "
                f"must be unique."
            )
        return row


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt(value: object, kind: str) -> str:
    """Format one cell, rendering absent values as an em dash."""

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return _MISSING
    if kind == "text":
        text = str(value)
        if text == "":
            # A deliberately blank label — the total row's donor cell — stays blank.
            return ""
        return _MISSING if text in {"nan", "None"} else text
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return _MISSING
    if kind == "int":
        return f"{number:,.0f}"
    if kind == "pct":
        return f"{number:.1f}"
    if kind == "num":
        # Three significant figures, but never in exponent form: a threshold
        # printed as "6.1e+00" is a threshold nobody can check.
        if abs(number) >= 1000:
            return f"{number:,.0f}"
        return f"{number:,.3g}"
    return str(value)


def _row_groups(body: pd.DataFrame, key: str | None) -> list[tuple[str | None, pd.DataFrame]]:
    """Split the body into ``(group label, rows)`` runs, preserving row order.

    Grouping is by consecutive runs rather than by value, so the caller's sort
    order survives: the body arrives ordered donor-then-condition and must stay
    that way.
    """

    if key is None or key not in body.columns or body.empty:
        return [(None, body)]
    labels = body[key].astype(str)
    run = (labels != labels.shift()).cumsum()
    return [(str(chunk[key].iat[0]), chunk) for _, chunk in body.groupby(run, sort=True)]


def _stub_index(columns: tuple[Column, ...], row_group: str | None) -> int:
    """Index of the column the group members are indented under.

    Usually column 0, but not when the grouping column is itself printed: the
    indent belongs on the label beside it, not on the group name.
    """

    for index, column in enumerate(columns):
        if column.key != row_group:
            return index
    return 0


def _bar_cells(
    table: TypesetTable, columns: tuple[Column, ...], group: str, subtotal: pd.Series
) -> list[str]:
    """Format a group subtotal row: the group name in the stub, its own numbers.

    Shared by the three renderers so a subtotal bar cannot say one thing in the
    HTML and another in the LaTeX.
    """

    stub = _stub_index(columns, table.row_group)
    cells: list[str] = []
    for index, column in enumerate(columns):
        if index == stub:
            cells.append(group)
        elif column.key == table.row_group:
            cells.append("")
        else:
            cells.append(_fmt(subtotal.get(column.key), column.fmt))
    return cells


def _spanner_runs(columns: tuple[Column, ...]) -> list[tuple[str | None, int, int]]:
    """Group columns into ``(spanner, first index, last index)`` runs."""

    runs: list[tuple[str | None, int, int]] = []
    for index, column in enumerate(columns):
        if runs and runs[-1][0] == column.spanner and column.spanner is not None:
            spanner, start, _ = runs.pop()
            runs.append((spanner, start, index))
        else:
            runs.append((column.spanner, index, index))
    return runs


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_cohort_table(
    *,
    cell_metrics: pd.DataFrame,
    cell_decisions: pd.DataFrame,
    obs: pd.DataFrame,
    sample_key: str,
    donor_key: str | None = None,
    condition_key: str | None = None,
    thresholds: pd.DataFrame | None = None,
    gene_summary: dict[str, int] | None = None,
    case_label: str | None = None,
    title: str = "Table 1. Sample-level quality control",
    stem: str = "qc_table1_cohort",
) -> TypesetTable:
    """Build the per-sample QC table: attrition, then the retained dataset.

    The two halves answer different questions and are deliberately kept in one
    table: the ``Cells`` block is what QC removed (counted pre-filter, or every
    row would read 0), and the ``Median`` block describes what survived, which is
    the dataset every downstream number comes from.

    Args:
        cell_metrics: Per-cell QC metrics, indexed by input cell.
        cell_decisions: Per-cell keep/fail decisions, indexed by input cell.
        obs: Cell metadata carrying the sample/donor/condition columns.
        sample_key: Column identifying the library.
        donor_key: Optional donor column, used for row grouping.
        condition_key: Optional condition column.
        thresholds: Optional applied-threshold table, summarised in the note.
        gene_summary: Optional ``{"n_genes", "n_genes_kept"}`` counts.
        case_label: Optional condition value marking the case arm, so each donor's
            control sample is printed first.
        title: Table title.
        stem: File stem and LaTeX label suffix.

    Returns:
        The table specification.
    """

    table = build_sample_qc_table(
        cell_metrics=cell_metrics,
        cell_decisions=cell_decisions,
        obs=obs,
        sample_key=sample_key,
        donor_key=donor_key,
        condition_key=condition_key,
        case_label=case_label,
        # A manuscript table describes the analysed cells, not the discarded ones.
        median_population="retained",
    )
    body = table.loc[table["sample"] != "TOTAL"].reset_index(drop=True)
    total_rows = table.loc[table["sample"] == "TOTAL"]
    total = total_rows.iloc[0] if len(total_rows) else None

    # With a donor column the donors become row groups, and the donor is then NOT
    # printed as a column: it would be an empty stub repeating each group heading.
    has_donor = "donor" in body.columns and body["donor"].nunique() > 1

    # Each donor heading carries that donor's own totals, pooled over their
    # samples: a within-donor imbalance is the thing this table is read for, and
    # it is invisible if the reader has to add two rows in their head.
    group_totals = None
    if has_donor:
        frame = build_qc_cell_frame(
            cell_metrics=cell_metrics,
            cell_decisions=cell_decisions,
            obs=obs,
            labels={"donor": donor_key},
        )
        group_totals = summarize_qc_rows(
            frame, label="donor", name="donor", median_population="retained"
        ).set_index("donor")
        # Text columns stay blank on a subtotal row rather than printing an em
        # dash: the donor total belongs to no single sample or condition.
        for blank in ("sample", "condition"):
            group_totals[blank] = ""
    columns: list[Column] = [Column("sample", "Sample", align="left", fmt="text")]
    if "condition" in body.columns:
        columns.append(Column("condition", "Condition", align="left", fmt="text"))

    columns += [
        Column("cells_in", "In", spanner="Cells", fmt="int", marker="a"),
        Column("cells_removed", "Removed", spanner="Cells", fmt="int"),
        Column("pct_removed", "%", spanner="Cells", fmt="pct", marker="b"),
        Column("cells_kept", "Retained", spanner="Cells", fmt="int"),
    ]
    median_spanner = "Median per retained cell"
    for key, label in (
        ("median_umi", "UMI"),
        ("median_genes", "Genes"),
        ("median_pct_mito", "% mito."),
        ("median_pct_ribo", "% ribo."),
    ):
        if key in body.columns:
            columns.append(
                Column(key, label, spanner=median_spanner, fmt="int" if "pct" not in key else "pct")
            )

    footnotes = [
        ("a", "Barcodes entering the quality-control stage for that sample."),
        (
            "b",
            "Within-sample removal. The cohort value is pooled over cells, not the "
            "mean of the per-sample percentages.",
        ),
    ]

    notes: list[str] = []
    criteria = _threshold_sentence(thresholds)
    if criteria:
        notes.append(f"Criteria applied: {criteria}")
    if gene_summary and gene_summary.get("n_genes"):
        n_genes = int(gene_summary["n_genes"])
        n_kept = int(gene_summary.get("n_genes_kept", n_genes))
        notes.append(
            f"Gene filtering: {n_kept:,} of {n_genes:,} genes retained "
            f"({100.0 * n_kept / n_genes:.1f}%)."
        )

    subtitle = None
    if total is not None:
        subtitle = (
            f"{int(total['cells_in']):,} cells in, {int(total['cells_kept']):,} retained "
            f"({float(total['pct_removed']):.1f}% removed) across {len(body):,} samples"
        )
        if has_donor:
            subtitle += f" from {body['donor'].nunique():,} donors"

    return TypesetTable(
        title=title,
        subtitle=subtitle,
        columns=tuple(columns),
        body=body,
        total=total,
        row_group="donor" if has_donor else None,
        group_totals=group_totals,
        footnotes=tuple(footnotes),
        source_note=" ".join(notes) or None,
        stem=stem,
    )


def build_cell_type_table(
    *,
    cell_metrics: pd.DataFrame,
    cell_decisions: pd.DataFrame,
    obs: pd.DataFrame,
    cell_type_key: str | None = None,
    granular_key: str | None = None,
    thresholds: pd.DataFrame | None = None,
    title: str = "Table 3. Quality control by cell type",
    stem: str = "qc_table3_cell_type",
) -> TypesetTable | None:
    """Build the per-cell-type QC table: a total bar per type, granular beneath.

    Attrition is not uniform across cell types, and that matters more than the
    per-sample view for interpretation: a filter that removes a fifth of one
    lineage and a fiftieth of another has changed the composition of the dataset
    before any analysis runs. Coarse types become total bars and their granular
    subtypes are indented under them, so the table reads at either resolution.

    Args:
        cell_metrics: Per-cell QC metrics, indexed by input cell.
        cell_decisions: Per-cell keep/fail decisions, indexed by input cell.
        obs: Cell metadata carrying the cell-type labels.
        cell_type_key: Optional explicit coarse cell-type column.
        granular_key: Optional explicit granular cell-type column.
        thresholds: Optional applied-threshold table, summarised in the note.
        title: Table title.
        stem: File stem and LaTeX label suffix.

    Returns:
        The table specification, or None when ``obs`` carries no cell-type
        labels at all.
    """

    coarse, granular = resolve_cell_type_keys(
        obs, cell_type_key=cell_type_key, granular_key=granular_key
    )
    if coarse is None and granular is None:
        return None

    frame = build_qc_cell_frame(
        cell_metrics=cell_metrics,
        cell_decisions=cell_decisions,
        obs=obs,
        labels={"coarse": coarse, "granular": granular},
    )
    # The stub is the finest label present; the coarse label groups it. With only
    # one of the two, the table is a flat list of that one.
    stub = "granular" if "granular" in frame.columns else "coarse"
    group = "coarse" if stub == "granular" and "coarse" in frame.columns else None

    rows = summarize_qc_rows(
        frame,
        label=stub,
        name="cell_type",
        carry=("coarse",) if group else (),
        median_population="retained",
    )
    if rows.empty:
        return None

    # Largest first, within group, unlabelled last — the row order shared with the
    # by-cell-type figure, so table and figure can be read against each other.
    rows = order_cell_types(rows, name="cell_type", group="coarse" if group else None)

    group_totals = None
    if group:
        subtotals = summarize_qc_rows(
            frame, label="coarse", name="coarse", median_population="retained"
        )
        group_totals = subtotals.set_index("coarse").reindex(list(dict.fromkeys(rows["coarse"])))
        # A group whose single member IS the group prints the same numbers twice,
        # once on its bar and once indented beneath it. Drop the member and keep
        # the bar; if that holds for every group the granular label adds no
        # sub-structure at all, so the table is a flat list of the coarse labels.
        # Same helper as the figure, so the two keep the same rows.
        redundant = redundant_group_members(rows, name="cell_type", group="coarse")
        if redundant.all():
            group, group_totals = None, None
        else:
            rows = rows.loc[~redundant]

    columns: list[Column] = [
        Column("cell_type", "Cell type", align="left", fmt="text"),
        Column("cells_in", "In", spanner="Cells", fmt="int", marker="a"),
        Column("cells_removed", "Removed", spanner="Cells", fmt="int"),
        Column("pct_removed", "%", spanner="Cells", fmt="pct", marker="b"),
        Column("cells_kept", "Retained", spanner="Cells", fmt="int"),
    ]
    median_spanner = "Median per retained cell"
    for key, label in (
        ("median_umi", "UMI"),
        ("median_genes", "Genes"),
        ("median_pct_mito", "% mito."),
        ("median_pct_ribo", "% ribo."),
    ):
        if key in rows.columns:
            columns.append(
                Column(key, label, spanner=median_spanner, fmt="pct" if "pct" in key else "int")
            )

    pooled = summarize_qc_pool(frame, median_population="retained")
    total = pd.Series({"cell_type": "TOTAL", **pooled})
    # One group means its bar already is the cohort total; printing both would be
    # the same row twice.
    if group_totals is not None and len(group_totals) == 1:
        total = None  # type: ignore[assignment]

    footnotes = [
        ("a", "Barcodes of that cell type entering the quality-control stage."),
        (
            "b",
            "Within-cell-type removal. Unequal values across types mean quality "
            "control changed the composition of the dataset.",
        ),
    ]

    # An unlabelled row is bookkeeping, not a population: it is counted in the
    # table and excluded from every claim the subtitle makes.
    named = rows.loc[rows["cell_type"] != UNLABELLED]
    subtitle_bits = [f"{len(named):,} cell type" + ("s" if len(named) != 1 else "")]
    n_lineages = named["coarse"].nunique() if group else 1
    if group and n_lineages > 1:
        subtitle_bits.append(f"in {n_lineages:,} lineages")
    n_kept = int(as_float(pooled["cells_kept"]))
    if n_kept and len(named):
        largest = named.loc[named["cells_kept"].idxmax()]
        share = 100.0 * as_float(largest["cells_kept"]) / n_kept
        subtitle_bits.append(f"{share:.1f}% of retained cells are {largest['cell_type']}")
    # Only worth naming when there is something to compare it against, and only
    # among populations big enough for a percentage to mean anything.
    sizeable = named.loc[named["cells_in"] >= SMALL_CELL_TYPE]
    if len(sizeable) > 1:
        hardest = sizeable.loc[sizeable["pct_removed"].idxmax()]
        subtitle_bits.append(
            f"hardest hit: {hardest['cell_type']} "
            f"({as_float(hardest['pct_removed']):.1f}% removed)"
        )
    if len(named) < len(rows):
        unlabelled = rows.loc[rows["cell_type"] == UNLABELLED, "cells_in"].sum()
        subtitle_bits.append(f"{int(unlabelled):,} cells carry no cell-type label")
    criteria = _threshold_sentence(thresholds)

    return TypesetTable(
        title=title,
        subtitle="; ".join(subtitle_bits),
        columns=tuple(columns),
        body=rows,
        total=total,
        row_group="coarse" if group else None,
        group_totals=group_totals,
        footnotes=tuple(footnotes),
        source_note=(f"Criteria applied: {criteria}" if criteria else None),
        stem=stem,
    )


def build_criteria_table(
    *,
    cell_decisions: pd.DataFrame,
    thresholds: pd.DataFrame | None = None,
    title: str = "Table 2. Contribution of each quality-control criterion",
    stem: str = "qc_table2_criteria",
) -> TypesetTable:
    """Build the per-criterion removal table.

    Criteria overlap, so the gross counts do not sum to the total removed. The
    marginal column — cells no other criterion caught — is the number that says
    whether a criterion is earning its place.

    Args:
        cell_decisions: Per-cell decisions with one boolean column per rule.
        thresholds: Optional applied-threshold table, used for readable labels.
        title: Table title.
        stem: File stem and LaTeX label suffix.

    Returns:
        The table specification.
    """

    attribution = build_rule_attribution_table(cell_decisions=cell_decisions, thresholds=thresholds)
    lookup = _threshold_lookup(thresholds)
    body = attribution.copy()
    if body.empty:
        body = pd.DataFrame(columns=["criterion", "cells_failed", "pct_of_input", "only_this_rule"])
    else:
        body["criterion"] = [_rule_label(str(rule), lookup.get(str(rule))) for rule in body["rule"]]
        body = body[["criterion", "cells_failed", "pct_of_input", "only_this_rule"]]

    n_input = int(len(cell_decisions))
    n_removed = (
        int((~cell_decisions["keep"].astype(bool)).sum())
        if "keep" in cell_decisions.columns
        else None
    )
    total = None
    if not body.empty:
        total = pd.Series(
            {
                "criterion": "Any criterion",
                "cells_failed": n_removed if n_removed is not None else np.nan,
                "pct_of_input": (
                    100.0 * n_removed / n_input if n_removed is not None and n_input else np.nan
                ),
                "only_this_rule": np.nan,
            }
        )

    return TypesetTable(
        title=title,
        subtitle=(
            f"Of {n_input:,} cells entering quality control"
            + (f", {n_removed:,} were removed" if n_removed is not None else "")
        ),
        columns=(
            Column("criterion", "Criterion (threshold applied)", align="left", fmt="text"),
            Column("cells_failed", "Cells failing", fmt="int"),
            Column("pct_of_input", "% of input", fmt="pct"),
            Column("only_this_rule", "Only this criterion", fmt="int", marker="a"),
        ),
        body=body,
        total=total,
        footnotes=(
            (
                "a",
                "Cells no other criterion would have removed. Criteria overlap, so "
                "the failing counts sum to more than the total removed.",
            ),
        ),
        source_note=(
            "MAD criteria are data-driven (median absolute deviation); fixed "
            "criteria are the configured constants."
        ),
        stem=stem,
    )


def _threshold_sentence(
    thresholds: pd.DataFrame | None,
    graded_policy: dict[str, float] | None = None,
) -> str:
    """Render the criteria applied as one readable clause, graded policy preferred.

    A threshold policy is a list of bounds and can be written as one. A graded policy cannot: its
    bars sit on a severity scale that means nothing without the sigma equivalence, and the
    operative rule is concordance across families rather than any single bound. So when the run
    was graded, the note states the policy that actually applied instead of a list of numbers
    that did not.

    Gene-level rules are dropped from the threshold form: this note sits under a table of cells,
    and "n cells by counts < 3" there reads as a cell criterion when it is the gene expression
    filter reported separately.
    """
    if graded_policy:
        from cellquorum.visualization.qc.graded import graded_policy_sentence

        return graded_policy_sentence(
            graded_policy["concern_severity"],
            graded_policy["severe_severity"],
            int(graded_policy["min_concordant_families"]),
        )

    lookup = _threshold_lookup(_cell_axis_only(thresholds))
    if not lookup:
        return ""
    parts = [_rule_label(rule, bound) for rule, bound in lookup.items()]
    return "; ".join(parts) + "."


def _cell_axis_only(thresholds: pd.DataFrame | None) -> pd.DataFrame | None:
    """Keep the cell-level rows of a threshold table when it says which is which."""

    if thresholds is None or "axis" not in thresholds.columns:
        return thresholds
    return thresholds.loc[thresholds["axis"].astype(str) == "cell"]


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_TABLE_CSS = """
:root { color-scheme: light; }
body {
  margin: 0; padding: 34px 30px 44px;
  background: #FFFFFF; color: #111418;
  font-family: "Source Serif 4", "Iowan Old Style", "Palatino Linotype",
               Palatino, Georgia, "Times New Roman", serif;
}
.tbl { margin: 0 0 40px; max-width: 1080px; }
.tbl table { border-collapse: collapse; width: 100%; }
.tbl caption { caption-side: top; text-align: left; padding: 0 0 3px; }
.ttl { font-size: 15px; font-weight: 600; letter-spacing: 0.005em; }
.sub { font-size: 12.5px; color: #5B6470; padding-top: 2px; font-style: italic; }
.tbl th, .tbl td {
  padding: 4px 12px 4px 0; font-size: 13px; line-height: 1.35;
  font-variant-numeric: tabular-nums lining-nums;
  white-space: nowrap;
}
.tbl th:last-child, .tbl td:last-child { padding-right: 0; }
.tbl thead tr.span th {
  font-size: 12px; font-weight: 600; padding-bottom: 2px; text-align: center;
}
.tbl thead tr.span th.grp { border-bottom: 1px solid #111418; }
.tbl thead tr.head th { font-weight: 600; }
.tbl thead tr.head { border-bottom: 1px solid #111418; }
.tbl thead tr.top th { border-top: 2px solid #111418; }
.tbl tbody tr.grp th {
  font-weight: 600; font-style: italic; font-size: 12.5px;
  padding: 9px 0 2px; text-align: left; color: #111418;
}
/* A group heading that carries its own totals: achromatic tint, so it reads as a
   bar in greyscale and under any colour vision. */
.tbl tbody tr.grp.bar th { background: #F1F4F6; padding: 5px 12px 5px 0; }
.tbl tbody tr.grp.bar th:last-child { padding-right: 0; }
.tbl tbody tr.grp.bar th.num { font-style: normal; text-align: right; }
.tbl tbody td.stub { padding-left: 14px; }
.tbl tbody tr.total td { border-top: 1px solid #111418; font-weight: 600; }
.tbl tfoot td { border-top: 2px solid #111418; padding-top: 7px; }
.num { text-align: right; font-variant-numeric: tabular-nums lining-nums; }
.txt { text-align: left; }
.note { font-size: 11.5px; color: #5B6470; line-height: 1.5; white-space: normal; }
.note sup { font-size: 9px; padding-right: 2px; }
.src { font-size: 11.5px; color: #5B6470; line-height: 1.5; padding-top: 4px; white-space: normal; }
"""


def render_table_html(table: TypesetTable) -> str:
    """Render one table as a ``<div>`` fragment in ``gt``/booktabs style.

    Args:
        table: Table specification.

    Returns:
        An HTML fragment. Use :func:`render_tables_page` for a whole document.
    """

    def esc(value: object) -> str:
        return _html.escape(str(value), quote=False)

    columns = table.columns
    n_columns = len(columns)
    out: list[str] = ['<div class="tbl"><table>']
    out.append(
        f'<caption><div class="ttl">{esc(table.title)}</div>'
        + (f'<div class="sub">{esc(table.subtitle)}</div>' if table.subtitle else "")
        + "</caption>"
    )

    out.append("<thead>")
    runs = _spanner_runs(columns)
    if any(spanner for spanner, _, _ in runs):
        cells = []
        for spanner, start, end in runs:
            span = end - start + 1
            if spanner is None:
                cells.append(f'<th colspan="{span}"></th>')
            else:
                cells.append(f'<th class="grp" colspan="{span}">{esc(spanner)}</th>')
        out.append('<tr class="span top">' + "".join(cells) + "</tr>")
        head_class = "head"
    else:
        head_class = "head top"

    head = []
    for column in columns:
        label = esc(column.label)
        if column.marker:
            label += f"<sup>{esc(column.marker)}</sup>"
        head.append(f'<th class="{"txt" if column.align == "left" else "num"}">{label}</th>')
    out.append(f'<tr class="{head_class}">' + "".join(head) + "</tr>")
    out.append("</thead><tbody>")

    for group, chunk in _row_groups(table.body, table.row_group):
        if group is not None:
            subtotal = table.group_total(group)
            if subtotal is None:
                out.append(f'<tr class="grp"><th colspan="{n_columns}">{esc(group)}</th></tr>')
            else:
                # The heading carries the group's own numbers: a tinted bar, so a
                # reader sees the cell type as one figure before the subtypes.
                cells = [
                    f'<th class="{"txt" if column.align == "left" else "num"}">{esc(text)}</th>'
                    for column, text in zip(
                        columns, _bar_cells(table, columns, group, subtotal), strict=True
                    )
                ]
                out.append('<tr class="grp bar">' + "".join(cells) + "</tr>")
        for _, record in chunk.iterrows():
            cells = []
            for index, column in enumerate(columns):
                # The grouping column is printed as the group heading, so its cell
                # would repeat the heading on every row.
                text = (
                    ""
                    if group is not None and column.key == table.row_group
                    else esc(_fmt(record.get(column.key), column.fmt))
                )
                classes = "txt" if column.align == "left" else "num"
                if group is not None and index == _stub_index(columns, table.row_group):
                    classes += " stub"
                cells.append(f'<td class="{classes}">{text}</td>')
            out.append("<tr>" + "".join(cells) + "</tr>")

    if table.total is not None:
        cells = []
        for column in columns:
            # The grouping column stays blank: the total belongs to no donor.
            text = (
                ""
                if column.key == table.row_group
                else esc(_fmt(table.total.get(column.key), column.fmt))
            )
            classes = "txt" if column.align == "left" else "num"
            cells.append(f'<td class="{classes}">{text}</td>')
        out.append('<tr class="total">' + "".join(cells) + "</tr>")

    out.append("</tbody>")
    if table.footnotes or table.source_note:
        out.append("<tfoot>")
        notes = []
        for marker, text in table.footnotes:
            notes.append(f'<div class="note"><sup>{esc(marker)}</sup>{esc(text)}</div>')
        if table.source_note:
            notes.append(f'<div class="src">{esc(table.source_note)}</div>')
        out.append(f'<tr><td colspan="{n_columns}">' + "".join(notes) + "</td></tr>")
        out.append("</tfoot>")
    out.append("</table></div>")
    return "\n".join(out)


def render_tables_page(tables: list[TypesetTable], *, project: str | None = None) -> str:
    """Render several tables as one self-contained HTML document.

    Args:
        tables: Table specifications, in print order.
        project: Optional project name for the document title.

    Returns:
        A complete HTML document with inline styling and no external assets.
    """

    heading = f"QC tables — {project}" if project else "QC tables"
    fragments = "\n".join(render_table_html(table) for table in tables)
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{_html.escape(heading, quote=False)}</title>"
        f"<style>{_TABLE_CSS}</style></head><body>\n{fragments}\n</body></html>\n"
    )


# ---------------------------------------------------------------------------
# LaTeX renderer
# ---------------------------------------------------------------------------

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # Text-mode "<" and ">" print as inverted punctuation in the OT1 encoding, so
    # "Genes per cell < 200" would silently become "Genes per cell ¡ 200".
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "—": r"---",
    "–": r"--",
}


def _tex(value: object) -> str:
    """Escape LaTeX specials in a cell or label."""

    return "".join(_LATEX_ESCAPES.get(character, character) for character in str(value))


def render_table_latex(table: TypesetTable) -> str:
    """Render one table as a ``booktabs`` LaTeX float.

    Requires ``\\usepackage{booktabs}`` in the document preamble; nothing else.

    Args:
        table: Table specification.

    Returns:
        A LaTeX ``table`` environment, ready to ``\\input``.
    """

    columns = table.columns
    spec = "".join("l" if column.align == "left" else "r" for column in columns)
    lines = [
        "% Requires \\usepackage{booktabs} in the preamble.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{_tex(table.title.split('. ', 1)[-1])}}}",
        f"\\label{{tab:{table.stem}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
    ]

    runs = _spanner_runs(columns)
    if any(spanner for spanner, _, _ in runs):
        cells, rules = [], []
        for spanner, start, end in runs:
            span = end - start + 1
            if spanner is None:
                cells.append("" if span == 1 else f"\\multicolumn{{{span}}}{{c}}{{}}")
            else:
                cells.append(f"\\multicolumn{{{span}}}{{c}}{{{_tex(spanner)}}}")
                rules.append(f"\\cmidrule(lr){{{start + 1}-{end + 1}}}")
        lines.append(" & ".join(cells) + " \\\\")
        if rules:
            lines.append("".join(rules))

    head = []
    for column in columns:
        label = _tex(column.label)
        if column.marker:
            label += f"\\textsuperscript{{{_tex(column.marker)}}}"
        head.append(label)
    lines.append(" & ".join(head) + " \\\\")
    lines.append("\\midrule")

    stub = _stub_index(columns, table.row_group)
    for group, chunk in _row_groups(table.body, table.row_group):
        if group is not None:
            lines.append("\\addlinespace[2pt]")
            subtotal = table.group_total(group)
            if subtotal is None:
                lines.append(
                    f"\\multicolumn{{{len(columns)}}}{{l}}{{\\textit{{{_tex(group)}}}}} \\\\"
                )
            else:
                # A ruled subtotal bar rather than a shaded one: shading needs
                # colortbl, and this file must compile with booktabs alone.
                cells = [
                    f"\\textit{{{_tex(text)}}}" if index == stub else _tex(text)
                    for index, text in enumerate(_bar_cells(table, columns, group, subtotal))
                ]
                lines.append(" & ".join(cells) + " \\\\")
                lines.append(f"\\cmidrule(l){{{stub + 1}-{len(columns)}}}")
        for _, record in chunk.iterrows():
            cells = []
            for index, column in enumerate(columns):
                if group is not None and column.key == table.row_group:
                    cells.append("")
                    continue
                text = _tex(_fmt(record.get(column.key), column.fmt))
                if group is not None and index == stub:
                    # Indentation is the only cue LaTeX has that these rows are
                    # members of the group above.
                    text = f"\\hspace{{1em}}{text}"
                cells.append(text)
            lines.append(" & ".join(cells) + " \\\\")

    if table.total is not None:
        lines.append("\\midrule")
        cells = []
        for column in columns:
            if column.key == table.row_group:
                cells.append("")
                continue
            cells.append(f"\\textbf{{{_tex(_fmt(table.total.get(column.key), column.fmt))}}}")
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]

    if table.footnotes or table.source_note:
        lines.append("\\begin{minipage}{\\linewidth}\\vspace{3pt}\\footnotesize")
        for marker, text in table.footnotes:
            lines.append(f"\\textsuperscript{{{_tex(marker)}}}{_tex(text)}\\par")
        if table.source_note:
            lines.append(f"{_tex(table.source_note)}\\par")
        lines.append("\\end{minipage}")

    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figure renderer
# ---------------------------------------------------------------------------

# Typographic metrics. Point sizes and inch measures set here rather than inline
# so the proportions can be compared side by side and stay deliberate.
_TITLE_PT = 10.6
_SUBTITLE_PT = 8.4
_SPANNER_PT = 8.0
_HEAD_PT = 8.2
_BODY_PT = 8.2
_NOTE_PT = 6.8
_MARKER_PT = 5.8

_ROW_H = 0.205
_GROUP_H = 0.250
_COL_GAP = 0.145
_STUB_INDENT = 0.11
_PAD_X = 0.16
_PAD_Y = 0.13
_NOTE_LEAD = 0.135


class _TextWidth(Protocol):
    """A cached text-width measurer in inches, holding a scratch figure open.

    Declared as a Protocol because the object is a function with a ``close``
    attached: a plain ``Callable`` cannot express the attribute, and the helpers
    that receive it need both halves.
    """

    def __call__(self, text: str, size: float, weight: str = "normal") -> float:
        """Return the rendered width of ``text`` in inches."""

    def close(self) -> None:
        """Close the scratch figure used for measurement."""


def _measure(dpi: int = 200) -> _TextWidth:
    """Return a text-width measuring function, in inches.

    Column widths are measured with the real font metrics rather than estimated
    from a character count: a serif "1" and "W" differ by a factor of two, and a
    guessed average is what makes headers collide with their neighbours.
    """

    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    scratch = plt.figure(figsize=(1.0, 1.0), dpi=dpi)
    # `get_renderer` is an Agg-canvas method, not part of the canvas base class, so it is
    # fetched defensively: a non-Agg backend would otherwise raise here while measuring text.
    get_renderer = getattr(scratch.canvas, "get_renderer", None)
    if get_renderer is None:  # pragma: no cover - Agg is the configured backend
        raise QCTableError(
            "Measuring table text needs an Agg canvas; the active matplotlib backend "
            f"({type(scratch.canvas).__name__}) does not provide get_renderer()."
        )
    renderer = get_renderer()
    cache: dict[tuple[str, float, str], float] = {}

    def width(text: str, size: float, weight: str = "normal") -> float:
        if not text:
            return 0.0
        key = (text, size, weight)
        if key not in cache:
            properties = FontProperties(family="serif", size=size, weight=weight)
            pixels, _, _ = renderer.get_text_width_height_descent(text, properties, ismath=False)
            cache[key] = pixels / dpi
        return cache[key]

    width.close = lambda: plt.close(scratch)  # type: ignore[attr-defined]
    return cast(_TextWidth, width)


def render_table_figure(table: TypesetTable) -> Figure:
    """Render one table as a matplotlib figure, typeset like the LaTeX version.

    Drawn by hand rather than with ``Axes.table``, which centres everything, draws
    a full grid and cannot do spanners — three of the four things that separate a
    typeset table from a spreadsheet screenshot. Positions are in inches with y
    counting downward, the direction a page is set in.

    Args:
        table: Table specification.

    Returns:
        A ``matplotlib.figure.Figure`` sized to its content.
    """

    import matplotlib.pyplot as plt

    columns = table.columns
    groups = _row_groups(table.body, table.row_group)
    measure = _measure()
    try:
        return _draw_table(table, columns, groups, measure, plt)
    finally:
        measure.close()


def _draw_table(
    table: TypesetTable,
    columns: tuple[Column, ...],
    groups: list[tuple[str | None, pd.DataFrame]],
    measure: _TextWidth,
    plt: ModuleType,
) -> Figure:
    """Lay out and draw a table. Split out so the measuring figure is always closed."""

    # Formatted body first: column widths depend on the rendered strings, not the
    # underlying values.
    rows: list[list[str]] = []
    for _, chunk in groups:
        for _, record in chunk.iterrows():
            rows.append([_fmt(record.get(column.key), column.fmt) for column in columns])
    total_row = (
        [_fmt(table.total.get(column.key), column.fmt) for column in columns]
        if table.total is not None
        else None
    )

    # Rows inside a group are indented, so the stub column must be that much wider
    # or the longest label runs into its neighbour.
    stub_column = _stub_index(columns, table.row_group)
    indent = _STUB_INDENT if any(label is not None for label, _ in groups) else 0.0

    # Group subtotal rows are as wide as any body row and set in a heavier weight,
    # so they have to be measured too or a group total collides with its neighbour.
    bar_rows: list[list[str]] = []
    for label, _ in groups:
        if label is None:
            continue
        subtotal = table.group_total(label)
        if subtotal is None:
            continue
        bar_rows.append(_bar_cells(table, columns, label, subtotal))

    widths: list[float] = []
    for index, column in enumerate(columns):
        offset = indent if index == stub_column else 0.0
        candidates = [measure(row[index], _BODY_PT) + offset for row in rows]
        candidates += [measure(row[index], _BODY_PT, "semibold") for row in bar_rows]
        if total_row is not None:
            candidates.append(measure(total_row[index], _BODY_PT, "semibold"))
        header = measure(column.label, _HEAD_PT, "semibold")
        if column.marker:
            header += measure(column.marker, _MARKER_PT, "semibold") + 0.012
        widths.append(max([header, *candidates, 0.12]))

    x_edges = [_PAD_X]
    for width in widths:
        x_edges.append(x_edges[-1] + width + _COL_GAP)
    table_width = x_edges[-1] - _COL_GAP
    total_width = table_width + _PAD_X

    # A spanner may be wider than the columns it covers; widen the last column of
    # the run rather than letting the label run into its neighbour.
    runs = _spanner_runs(columns)
    for spanner, start, end in runs:
        if spanner is None:
            continue
        span_width = x_edges[end] + widths[end] - x_edges[start]
        needed = measure(spanner, _SPANNER_PT, "semibold") + 0.10
        if needed > span_width:
            widths[end] += needed - span_width
            x_edges = [_PAD_X]
            for width in widths:
                x_edges.append(x_edges[-1] + width + _COL_GAP)
            table_width = x_edges[-1] - _COL_GAP
            total_width = table_width + _PAD_X

    has_spanner = any(spanner for spanner, _, _ in runs)
    n_groups = sum(1 for label, _ in groups if label is not None)

    # Notes wrap to the table width, measured the same way, or a long note runs
    # off the edge of the figure.
    note_lines: list[str] = []
    for _marker, text in table.footnotes:
        note_lines += _wrap_to_width(text, table_width - _PAD_X - 0.10, measure, _NOTE_PT)
    if table.source_note:
        note_lines += _wrap_to_width(table.source_note, table_width - _PAD_X, measure, _NOTE_PT)

    header_h = 0.30 + (0.20 if table.subtitle else 0.0) + (0.21 if has_spanner else 0.0) + 0.25
    body_h = len(rows) * _ROW_H + n_groups * _GROUP_H
    total_h = (0.06 + _ROW_H) if total_row is not None else 0.0
    notes_h = (0.10 + len(note_lines) * _NOTE_LEAD) if note_lines else 0.0
    total_height = _PAD_Y + header_h + body_h + total_h + notes_h + _PAD_Y

    figure = plt.figure(figsize=(total_width, total_height))
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_axis_off()
    axes.set_xlim(0.0, total_width)
    axes.set_ylim(total_height, 0.0)

    serif = {"family": "serif", "color": _INK}

    def rule(y: float, x0: float, x1: float, lw: float) -> None:
        axes.plot([x0, x1], [y, y], lw=lw, color=_INK, solid_capstyle="butt", zorder=2)

    def cell(index: int, y: float, text: str, size: float, weight: str, dx: float = 0.0) -> None:
        """Draw one cell, right-aligned on its column edge unless it is a label."""
        if columns[index].align == "left":
            axes.text(
                x_edges[index] + dx,
                y,
                text,
                fontsize=size,
                fontweight=weight,
                ha="left",
                va="center",
                **serif,
            )
        else:
            axes.text(
                x_edges[index] + widths[index],
                y,
                text,
                fontsize=size,
                fontweight=weight,
                ha="right",
                va="center",
                **serif,
            )

    y = _PAD_Y
    axes.text(_PAD_X, y, table.title, fontsize=_TITLE_PT, fontweight="semibold", va="top", **serif)
    y += 0.28
    if table.subtitle:
        axes.text(
            _PAD_X,
            y,
            table.subtitle,
            fontsize=_SUBTITLE_PT,
            va="top",
            style="italic",
            family="serif",
            color=_SUBTLE,
        )
        y += 0.20

    rule(y, _PAD_X, table_width, 1.25)
    y += 0.06

    if has_spanner:
        for spanner, start, end in runs:
            if spanner is None:
                continue
            x0, x1 = x_edges[start], x_edges[end] + widths[end]
            axes.text(
                (x0 + x1) / 2.0,
                y + 0.10,
                spanner,
                fontsize=_SPANNER_PT,
                fontweight="semibold",
                ha="center",
                va="center",
                **serif,
            )
            rule(y + 0.185, x0, x1, 0.7)
        y += 0.215

    baseline = y + 0.115
    for index, column in enumerate(columns):
        label_width = measure(column.label, _HEAD_PT, "semibold")
        if column.align == "left":
            x_label = x_edges[index]
        else:
            marker_width = (
                measure(column.marker, _MARKER_PT, "semibold") + 0.012 if column.marker else 0.0
            )
            x_label = x_edges[index] + widths[index] - marker_width - label_width
        axes.text(
            x_label,
            baseline,
            column.label,
            fontsize=_HEAD_PT,
            fontweight="semibold",
            ha="left",
            va="center",
            **serif,
        )
        if column.marker:
            # Drawn as its own raised text rather than mathtext: a "$^{a}$" makes
            # matplotlib parse the whole header as maths and set it in italics.
            axes.text(
                x_label + label_width + 0.012,
                baseline - 0.045,
                column.marker,
                fontsize=_MARKER_PT,
                fontweight="semibold",
                ha="left",
                va="center",
                **serif,
            )
    y += 0.235
    rule(y, _PAD_X, table_width, 0.85)
    y += 0.04

    stub = _stub_index(columns, table.row_group)
    for group, chunk in groups:
        if group is not None:
            subtotal = table.group_total(group)
            if subtotal is not None:
                # Tinted bar behind the row, achromatic so it survives greyscale
                # print and any colour vision deficiency.
                axes.add_patch(
                    plt.Rectangle(
                        (_PAD_X - 0.05, y + 0.012),
                        table_width - _PAD_X + 0.05,
                        _GROUP_H - 0.024,
                        facecolor=_BAND,
                        edgecolor="none",
                        zorder=0,
                    )
                )
            axes.text(
                x_edges[stub],
                y + _GROUP_H / 2.0,
                group,
                fontsize=_HEAD_PT,
                style="italic",
                fontweight="semibold",
                ha="left",
                va="center",
                **serif,
            )
            if subtotal is not None:
                for index, text in enumerate(_bar_cells(table, columns, group, subtotal)):
                    if index != stub:
                        cell(index, y + _GROUP_H / 2.0, text, _BODY_PT, "semibold")
            y += _GROUP_H
        for _, record in chunk.iterrows():
            for index, column in enumerate(columns):
                if group is not None and column.key == table.row_group:
                    continue
                cell(
                    index,
                    y + _ROW_H / 2.0,
                    _fmt(record.get(column.key), column.fmt),
                    _BODY_PT,
                    "normal",
                    dx=_STUB_INDENT if (group is not None and index == stub) else 0.0,
                )
            y += _ROW_H

    if total_row is not None:
        y += 0.06
        rule(y, _PAD_X, table_width, 0.7)
        for index, column in enumerate(columns):
            if column.key == table.row_group:
                continue
            cell(index, y + _ROW_H / 2.0 + 0.02, total_row[index], _BODY_PT, "semibold")
        y += _ROW_H

    y += 0.05
    rule(y, _PAD_X, table_width, 1.25)
    y += 0.11
    marker_lines: dict[int, tuple[str | None, str]] = {}
    for marker, text in table.footnotes:
        for offset, line in enumerate(
            _wrap_to_width(text, table_width - _PAD_X - 0.10, measure, _NOTE_PT)
        ):
            marker_lines[len(marker_lines)] = (marker if offset == 0 else None, line)
    for _, (line_marker, line) in sorted(marker_lines.items()):
        if line_marker:
            axes.text(
                _PAD_X,
                y - 0.018,
                line_marker,
                fontsize=_MARKER_PT - 0.6,
                va="top",
                family="serif",
                color=_SUBTLE,
            )
        axes.text(
            _PAD_X + 0.10, y, line, fontsize=_NOTE_PT, va="top", family="serif", color=_SUBTLE
        )
        y += _NOTE_LEAD
    if table.source_note:
        for line in _wrap_to_width(table.source_note, table_width - _PAD_X, measure, _NOTE_PT):
            axes.text(_PAD_X, y, line, fontsize=_NOTE_PT, va="top", family="serif", color=_SUBTLE)
            y += _NOTE_LEAD

    return figure


def _wrap_to_width(text: str, width: float, measure: _TextWidth, size: float) -> list[str]:
    """Wrap a note to a width in inches, measured in the font it will be set in."""

    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if measure(candidate, size) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_qc_publication_tables(
    output_path: Path,
    *,
    cell_metrics: pd.DataFrame,
    cell_decisions: pd.DataFrame,
    obs: pd.DataFrame,
    sample_key: str,
    donor_key: str | None = None,
    condition_key: str | None = None,
    thresholds: pd.DataFrame | None = None,
    gene_summary: dict[str, int] | None = None,
    case_label: str | None = None,
    cell_type_key: str | None = None,
    granular_cell_type_key: str | None = None,
    project: str | None = None,
    formats: tuple[str, ...] = ("html", "tex", "png"),
    dpi: int = 400,
) -> list[Path]:
    """Write the typeset QC tables in every requested format.

    Args:
        output_path: Directory to write into. Created if absent.
        cell_metrics: Per-cell QC metrics, indexed by input cell.
        cell_decisions: Per-cell keep/fail decisions, indexed by input cell.
        obs: Cell metadata carrying sample/donor/condition columns.
        sample_key: Column identifying the library.
        donor_key: Optional donor column.
        condition_key: Optional condition column.
        thresholds: Optional applied-threshold table.
        gene_summary: Optional ``{"n_genes", "n_genes_kept"}`` counts.
        case_label: Optional condition value marking the case arm.
        cell_type_key: Optional coarse cell-type column. Auto-detected when None.
        granular_cell_type_key: Optional granular cell-type column, printed as
            sub-rows under each coarse type. Auto-detected when None.
        project: Optional project name for the HTML document title.
        formats: Any of ``"html"``, ``"tex"``, ``"png"``, ``"pdf"``, ``"svg"``.
        dpi: Raster resolution for ``png``.

    Returns:
        Paths written, in the order produced.
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    tables = [
        build_cohort_table(
            cell_metrics=cell_metrics,
            cell_decisions=cell_decisions,
            obs=obs,
            sample_key=sample_key,
            donor_key=donor_key,
            condition_key=condition_key,
            thresholds=thresholds,
            gene_summary=gene_summary,
            case_label=case_label,
        ),
        build_criteria_table(cell_decisions=cell_decisions, thresholds=thresholds),
    ]

    # Only when the object carries cell-type labels: attrition is not uniform
    # across populations, and an unannotated object cannot say so.
    by_cell_type = build_cell_type_table(
        cell_metrics=cell_metrics,
        cell_decisions=cell_decisions,
        obs=obs,
        cell_type_key=cell_type_key,
        granular_key=granular_cell_type_key,
        thresholds=thresholds,
    )
    if by_cell_type is not None:
        tables.append(by_cell_type)

    written: list[Path] = []
    if "html" in formats:
        # One page for both tables: the reader opens a single file and reads the
        # cohort table with its criteria breakdown directly beneath.
        path = output_path / "qc_tables.html"
        path.write_text(render_tables_page(tables, project=project), encoding="utf-8")
        written.append(path)

    for table in tables:
        if "tex" in formats:
            path = output_path / f"{table.stem}.tex"
            path.write_text(render_table_latex(table), encoding="utf-8")
            written.append(path)
        raster = [suffix for suffix in ("png", "pdf", "svg") if suffix in formats]
        if raster:
            import matplotlib.pyplot as plt

            from cellquorum.visualization.figstyle import atomic_savefig

            figure = render_table_figure(table)
            for suffix in raster:
                path = output_path / f"{table.stem}.{suffix}"
                # atomic_savefig rather than a bare savefig so a renderer that dies
                # mid-write leaves no truncated file to be mistaken for a figure.
                # pad_inches stays at 0.06: a typeset table wants a hairline margin,
                # not save_figure's default, which is why this loop cannot use it.
                atomic_savefig(
                    figure,
                    path,
                    dpi=dpi if suffix == "png" else None,
                    bbox_inches="tight",
                    pad_inches=0.06,
                    facecolor="white",
                )
                written.append(path)
            plt.close(figure)

    return written


__all__ = [
    "Column",
    "QCTableError",
    "TypesetTable",
    "build_cell_type_table",
    "build_cohort_table",
    "build_criteria_table",
    "resolve_cell_type_keys",
    "render_table_figure",
    "render_table_html",
    "render_table_latex",
    "render_tables_page",
    "write_qc_publication_tables",
]
