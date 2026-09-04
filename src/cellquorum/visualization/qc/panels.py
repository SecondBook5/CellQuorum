"""Publication QC panels built around one question: what did QC do to this cohort?

The pre-existing QC figures answer "what do the metrics look like" — a wall of
per-sample boxplots, one metric each. That layout has three problems this module
is written to avoid:

* It never shows the filter. A cohort can lose 13% of its cells and no panel
  changes, which is exactly how the 2026-09-01 VEC run shipped a "100% pass"
  barplot.
* It hides the paired design. Sample labels rotated 45 degrees into an
  unreadable jam put ``P1_Normal`` and ``P1_LE`` next to each other by accident
  rather than comparing them on purpose.
* It draws empty axes. "No UMAP embedding" and "No PCA variance" placeholders
  consumed 40% of the supplementary sheet on every run, because QC runs before
  either exists.

So: horizontal layout everywhere (sample names read left-to-right), donor
pairing as the organising principle, exclusion regions shaded rather than hinted
at with a dashed line, and panels omitted rather than stubbed when the data is
not there.

Colour is three values — Normal, case, and removed — separated in both hue and
lightness, so the panels survive deuteranopia and greyscale print. Continuous
scales are monotonic-lightness (``magma_r``) or symmetric diverging
(``RdBu_r``) for the same reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc.mixture import MIQC_POSTERIOR_COLUMN
from cellquorum.visualization.figstyle import (
    LE_RED,
    NORMAL_BLUE,
    TEXT,
    save_figure,
    set_publication_style,
    two_group_test_on_donor_medians,
)

# Removed cells are grey: desaturated so they read as background against either
# condition colour, dark enough to stay visible when printed.
REMOVED_GREY = "#9AA1A8"
REMOVED_GREY_DARK = "#6E757C"
KEPT_GREEN = "#3F8A63"
RULE_FILL = "#D9DEE3"
RULE_UNIQUE = "#B5484F"
GRID_GREY = "#E6EAED"

# The group-subtotal tint. Achromatic on purpose — it marks a row's ROLE, and
# spending a hue on that would collide with the condition palette. Shared with
# the typeset tables so a subtotal row looks the same in the figure and the table.
GROUP_BAND = "#F1F4F6"

# What a cell with no group label is called. Never dropped: a cohort of 2,125
# cells whose per-cell-type table sums to 1,809 is a table nobody can reconcile.
UNLABELLED = "Unlabelled"

# Metric display names. Keyed on the raw metric column so the same label is used
# by an axis, a rule description and a matrix row without drifting.
METRIC_LABELS: dict[str, str] = {
    "total_counts": "UMIs per cell",
    "n_genes_by_counts": "Genes per cell",
    "log1p_total_counts": "UMIs per cell",
    "log1p_n_genes_by_counts": "Genes per cell",
    "pct_counts_mito": "Mitochondrial %",
    "pct_counts_ribo": "Ribosomal %",
    "pct_counts_hemoglobin": "Hemoglobin %",
    "pct_counts_in_top_20_genes": "Top-20 gene %",
    "doublet_score": "Doublet score",
}

# Short forms for the matrix panel, where column width is scarce.
METRIC_SHORT: dict[str, str] = {
    "total_counts": "UMI",
    "n_genes_by_counts": "Genes",
    "pct_counts_mito": "%mito",
    "pct_counts_ribo": "%ribo",
    "pct_counts_hemoglobin": "%hb",
    "pct_counts_in_top_20_genes": "Top-20%",
    "doublet_score": "Doublet",
}

MATRIX_METRICS: tuple[str, ...] = (
    "total_counts",
    "n_genes_by_counts",
    "pct_counts_mito",
    "pct_counts_ribo",
    "pct_counts_in_top_20_genes",
    "doublet_score",
)

_KEEP_COLUMN = "cellquorum_qc_keep"

#: Graded QC columns, preferred over the threshold verdict. ``qc_fit_manifold`` is the
#: consequential one: it is what every fitting stage actually reads, so "kept" in a panel
#: should mean "allowed to define the biological reference", not "survived a threshold".
#: The threshold verdict remains a fallback for objects written before graded QC existed.
_GRADED_FIT_COLUMN = "qc_fit_manifold"
_GRADED_STATE_COLUMN = "qc_state_initial"
_RULE_PREFIX = "cellquorum_qc_"
_RESERVED_RULES = {"keep", "fail_any_qc", "failed_rules"}


class QCPanelError(CellQuorumDataError):
    """Report failures building QC panels."""


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


# Conventional CellQuorum annotation columns, coarse then granular, searched in
# order when a caller does not name one. Both lists live here rather than in the
# table module because the figures and the tables must group identically.
COARSE_CELL_TYPE_KEYS: tuple[str, ...] = (
    "cell_type",
    "cell_type_coarse",
    "lineage",
    "compartment",
    "ref_state",
)
GRANULAR_CELL_TYPE_KEYS: tuple[str, ...] = (
    "cell_type_granular",
    "cell_subtype",
    "cell_type_fine",
    "cell_state",
    "subcluster",
)


def resolve_cell_type_keys(
    obs: pd.DataFrame,
    *,
    cell_type_key: str | None = None,
    granular_key: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve the coarse and granular cell-type obs columns.

    Named columns win; otherwise the conventional CellQuorum annotation columns
    are searched. Returning ``(None, None)`` is normal — QC runs on unannotated
    objects too, and the caller then skips the by-cell-type output rather than
    inventing a grouping.

    Args:
        obs: Cell metadata to search.
        cell_type_key: Optional explicit coarse column.
        granular_key: Optional explicit granular column.

    Returns:
        ``(coarse, granular)``, either of which may be None.
    """

    def first(explicit: str | None, candidates: tuple[str, ...]) -> str | None:
        if explicit:
            return explicit if explicit in obs.columns else None
        return next((name for name in candidates if name in obs.columns), None)

    coarse = first(cell_type_key, COARSE_CELL_TYPE_KEYS)
    granular = first(granular_key, GRANULAR_CELL_TYPE_KEYS)
    if coarse is not None and granular == coarse:
        granular = None
    return coarse, granular


def label_series(values: pd.Series, index: pd.Index) -> pd.Series:
    """Align a label column to an index, naming the gaps rather than losing them.

    Reindexing a post-filter label column onto the pre-filter population leaves
    a hole for every removed cell. ``astype(str)`` would spell those holes "nan"
    and they would then sort in as a cell type called nan; naming them once, here,
    is what keeps them countable and keeps them out of the legend as a fake group.
    """

    aligned = values.reindex(index).astype("object")
    return aligned.where(aligned.notna(), UNLABELLED).astype(str)


def assemble_qc_frame(
    *,
    obs: pd.DataFrame,
    cell_metrics: pd.DataFrame | None = None,
    cell_decisions: pd.DataFrame | None = None,
    sample_key: str | None = None,
    donor_key: str | None = None,
    condition_key: str | None = None,
    cell_type_key: str | None = None,
    granular_key: str | None = None,
) -> pd.DataFrame:
    """
    Assemble the tidy per-cell frame every panel in this module reads.

    Accepts either an annotated ``obs`` alone (the in-pipeline path, where the
    stage has already written ``cellquorum_qc_*`` columns) or ``obs`` plus the
    raw metric and decision tables (the replot path, reading CSVs back off a
    finished run). Either way the result is indexed by every INPUT cell, because
    a frame containing only survivors cannot describe the filter.

    Args:
        obs: Cell metadata. May already carry ``cellquorum_qc_*`` columns.
        cell_metrics: Optional per-cell metric table to merge in.
        cell_decisions: Optional per-cell decision table to merge in.
        sample_key: Column identifying the library/sample.
        donor_key: Column identifying the donor.
        condition_key: Column identifying the condition.
        cell_type_key: Column carrying the coarse cell-type label. Auto-detected
            from the conventional annotation columns when both cell-type
            arguments are omitted.
        granular_key: Column carrying the granular cell-type label.

    Returns:
        A frame with a boolean ``keep`` column, the grouping columns present
        under the fixed names ``sample``/``donor``/``condition``/``cell_type``/
        ``cell_type_granular``, one column per available metric, and one boolean
        column per QC rule prefixed ``rule:``.

    Raises:
        QCPanelError: If no keep/fail decision can be resolved.
    """

    frame = pd.DataFrame(index=obs.index)

    # Resolve keep/fail, graded first. Under graded QC the meaningful line is eligibility to
    # fit — that is the permission every downstream stage reads — so a panel drawn from it shows
    # what the run actually did. The threshold verdict is a fallback for objects predating it,
    # and the decisions table a further fallback for the replot-from-CSV path.
    if _GRADED_FIT_COLUMN in obs.columns:
        frame["keep"] = obs[_GRADED_FIT_COLUMN].astype(bool)
        if _GRADED_STATE_COLUMN in obs.columns:
            # Carried so a panel can separate "questionable but projected" from "excluded",
            # which one boolean cannot express and which is the whole point of the graded model.
            frame["qc_state"] = obs[_GRADED_STATE_COLUMN].astype(str)
    elif cell_decisions is not None and "keep" in cell_decisions.columns:
        frame["keep"] = cell_decisions["keep"].reindex(frame.index).astype(bool)
    elif _KEEP_COLUMN in obs.columns:
        frame["keep"] = obs[_KEEP_COLUMN].astype(bool)
    else:
        raise QCPanelError(
            "QC panels need a keep/fail decision per cell. Supply an obs carrying "
            f"'{_GRADED_FIT_COLUMN}' (graded QC), cell_decisions, or '{_KEEP_COLUMN}'. "
            f"obs columns present: {sorted(obs.columns)[:20]}"
        )

    coarse_key, fine_key = resolve_cell_type_keys(
        obs, cell_type_key=cell_type_key, granular_key=granular_key
    )
    for target, key in (
        ("sample", sample_key),
        ("donor", donor_key),
        ("condition", condition_key),
        ("cell_type", coarse_key),
        ("cell_type_granular", fine_key),
    ):
        if key and key in obs.columns:
            frame[target] = label_series(obs[key], frame.index)

    # Metrics: prefer the standalone table, fall back to obs annotations.
    for metric in METRIC_LABELS:
        series: pd.Series | None = None
        if cell_metrics is not None and metric in cell_metrics.columns:
            series = cell_metrics[metric]
        elif metric in obs.columns:
            series = obs[metric]
        if series is not None:
            frame[metric] = pd.to_numeric(series.reindex(frame.index), errors="coerce")

    # The mitochondrial mixture's per-cell posterior, when a run recorded one.
    #
    # Deliberately not added to METRIC_LABELS. That mapping is the metric MENU for
    # the sample matrix and the paired dumbbells, and a posterior is a model's
    # verdict rather than a measurement of the cell, so it would be meaningless in
    # either: a donor-paired test of "mean P(compromised)" tests the filter against
    # itself. Only the mixture panel reads this column, and it reads it by name.
    for source in (cell_metrics, obs):
        if source is not None and MIQC_POSTERIOR_COLUMN in source.columns:
            frame[MIQC_POSTERIOR_COLUMN] = pd.to_numeric(
                source[MIQC_POSTERIOR_COLUMN].reindex(frame.index), errors="coerce"
            )
            break

    # Doublet scores live in obs under a couple of historical names.
    if "doublet_score" not in frame.columns:
        for candidate in ("doublet_score", "scDblFinder_score", "scrublet_score"):
            if candidate in obs.columns:
                frame["doublet_score"] = pd.to_numeric(
                    obs[candidate].reindex(frame.index), errors="coerce"
                )
                break

    # Rules, from whichever source carries them.
    rule_source: pd.DataFrame | None = None
    rule_columns: list[str] = []
    if cell_decisions is not None:
        rule_source = cell_decisions
        rule_columns = [c for c in cell_decisions.columns if c not in _RESERVED_RULES]
    else:
        rule_source = obs
        rule_columns = [
            c
            for c in obs.columns
            if c.startswith(_RULE_PREFIX) and c.removeprefix(_RULE_PREFIX) not in _RESERVED_RULES
        ]
    for column in rule_columns:
        values = rule_source[column].reindex(frame.index)
        # Only genuinely boolean columns are rules; `failed_rules` is a string.
        non_null = values.dropna()
        if len(non_null) and not non_null.isin([True, False, 0, 1]).all():
            continue
        name = column.removeprefix(_RULE_PREFIX)
        frame[f"rule:{name}"] = values.fillna(False).astype(bool)

    return frame


def summarize_by_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the per-cell frame to one row per sample, donor-ordered.

    Ordering by donor then condition is what makes the paired panels work: a
    donor's two samples end up adjacent by construction rather than by luck of
    alphabetical sample naming.

    Args:
        frame: Output of :func:`assemble_qc_frame`.

    Returns:
        One row per sample with cell counts, percent removed, and per-metric
        medians. Empty frame when no sample column is present.
    """

    if "sample" not in frame.columns:
        return pd.DataFrame()

    table = _summarize_qc_groups(
        frame,
        label="sample",
        name="sample",
        carry={"donor": "donor", "condition": "condition"},
        median_population="all",
    )
    order = [c for c in ("donor", "condition", "sample") if c in table.columns]
    return table.sort_values(order).reset_index(drop=True)


def order_cell_types(
    rows: pd.DataFrame,
    *,
    name: str = "cell_type",
    group: str | None = None,
) -> pd.DataFrame:
    """Order a per-cell-type table largest-first, within group.

    The reading order the by-cell-type table and its figure share. Size, not the
    alphabet: a QC table is read to find the populations that carry the dataset
    and the ones the filter hit hardest, and an alphabetical list buries both.
    Unlabelled cells sort last however many there are — they are a bookkeeping
    row, not a population.

    Args:
        rows: One row per cell type, carrying ``cells_in``.
        name: Column holding the cell-type label.
        group: Optional column holding the coarser grouping label.

    Returns:
        The table reordered, with a reset index.
    """

    ordered = rows.copy()
    keys = [name] if group is None else [group, name]
    for column in keys:
        if column not in ordered.columns:
            raise QCPanelError(
                f"order_cell_types needs a '{column}' column. "
                f"Present: {sorted(ordered.columns)}"
            )

    # Rank by size within each level, with the unlabelled row pinned last, then
    # sort on the ranks. Sorting on the labels themselves is what would put
    # "B cells" above a population fifty times its size.
    def size_order(labels: pd.Series, sizes: pd.Series) -> list[str]:
        totals = sizes.groupby(labels, observed=True).sum().sort_values(ascending=False)
        names = [str(value) for value in totals.index]
        return [n for n in names if n != UNLABELLED] + [n for n in names if n == UNLABELLED]

    sizes = ordered["cells_in"]
    if group is not None:
        ordered[group] = pd.Categorical(
            ordered[group], categories=size_order(ordered[group], sizes), ordered=True
        )
    ordered[name] = pd.Categorical(
        ordered[name], categories=size_order(ordered[name], sizes), ordered=True
    )
    # mergesort keeps the within-group size order stable under the group sort.
    ordered = ordered.sort_values(keys, kind="mergesort").reset_index(drop=True)
    for column in keys:
        ordered[column] = ordered[column].astype(str)
    return ordered


def redundant_group_members(
    rows: pd.DataFrame,
    *,
    name: str = "cell_type",
    group: str = "group",
) -> pd.Series:
    """Flag member rows that only restate the subtotal above them.

    A group holding one member whose label IS the group's label prints the same
    numbers twice — once on the bar, once indented beneath it. Shared between the
    figure and the table so both drop the same rows.

    Args:
        rows: One row per cell type.
        name: Column holding the cell-type label.
        group: Column holding the grouping label.

    Returns:
        A boolean Series aligned to ``rows``.
    """

    if group not in rows.columns or name not in rows.columns:
        return pd.Series(False, index=rows.index)
    members = rows.groupby(group, observed=True)[name].transform("size")
    return (members == 1) & (rows[name].astype(str) == rows[group].astype(str))


def summarize_by_cell_type(
    frame: pd.DataFrame,
    *,
    median_population: str = "retained",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Collapse the per-cell frame to one row per cell type, plus group subtotals.

    Attrition is rarely uniform across cell types, and that matters more than the
    per-sample view for interpretation: a filter that removes a fifth of one
    population and a fiftieth of another has changed the composition of the
    dataset before any analysis runs. The finest label available becomes the row
    and the coarse label groups it, so the same summary reads at either resolution.

    Args:
        frame: Output of :func:`assemble_qc_frame`.
        median_population: ``"retained"`` or ``"all"`` — which cells the metric
            medians describe. Retained by default, because these medians describe
            the dataset the analysis runs on. Attrition counts are always
            pre-filter regardless.

    Returns:
        ``(rows, group_totals)``. ``rows`` is one row per cell type in reading
        order; ``group_totals`` is one row per coarse group indexed by group name,
        or None when there is no coarser label to group by. Both are empty/None
        when the frame carries no cell-type labels at all.
    """

    stub = "cell_type_granular" if "cell_type_granular" in frame.columns else "cell_type"
    if stub not in frame.columns:
        return pd.DataFrame(), None
    group = "cell_type" if stub == "cell_type_granular" and "cell_type" in frame.columns else None

    rows = _summarize_qc_groups(
        frame,
        label=stub,
        name="cell_type",
        # Carried under a neutral name: the coarse column is itself called
        # cell_type when the granular label is the stub, and two columns of that
        # name would leave the table holding whichever was written last.
        carry={group: "group"} if group else None,
        median_population=median_population,
    )
    if rows.empty:
        return rows, None
    rows = order_cell_types(rows, name="cell_type", group="group" if group else None)

    group_totals = None
    if group and not redundant_group_members(rows, name="cell_type", group="group").all():
        # Skipped when every group restates itself, i.e. the granular label adds no
        # sub-structure anywhere: grouping then buys a bar per row and nothing else.
        subtotals = _summarize_qc_groups(
            frame, label=group, name="group", median_population=median_population
        )
        # Same order as the rows they head, so a caller can walk them together.
        order = list(dict.fromkeys(rows["group"]))
        group_totals = subtotals.set_index("group").reindex(order)
    return rows, group_totals


def _summarize_qc_groups(
    frame: pd.DataFrame,
    *,
    label: str,
    name: str,
    carry: dict[str, str] | None = None,
    median_population: str = "retained",
) -> pd.DataFrame:
    """Aggregate the per-cell frame into one attrition row per label value.

    Counts come off the pre-filter population always; only the metric medians
    honour ``median_population``. ``carry`` maps a source column to the name it
    takes in the result, and reports ``"mixed"`` when a group spans values.
    """

    carry = carry or {}
    metrics = [m for m in METRIC_LABELS if m in frame.columns]
    records: list[dict[str, object]] = []
    for value, chunk in frame.groupby(label, observed=True, dropna=False, sort=True):
        record: dict[str, object] = {name: str(value)}
        for source, target in carry.items():
            if source in chunk.columns:
                unique = chunk[source].dropna().unique()
                record[target] = str(unique[0]) if len(unique) == 1 else "mixed"
        n_in = int(len(chunk))
        n_keep = int(chunk["keep"].sum())
        record.update(
            cells_in=n_in,
            cells_kept=n_keep,
            cells_removed=n_in - n_keep,
            pct_removed=100.0 * (n_in - n_keep) / n_in if n_in else np.nan,
        )
        described = chunk.loc[chunk["keep"]] if median_population == "retained" else chunk
        for metric in metrics:
            # NaN when a group lost every cell: it contributed nothing to the
            # analysed dataset, which is the honest value rather than zero.
            record[metric] = float(described[metric].median()) if len(described) else np.nan
        records.append(record)
    columns = [name, *carry.values(), "cells_in", "cells_kept", "cells_removed", "pct_removed"]
    return pd.DataFrame(records, columns=None if records else columns)


def cell_type_display_rows(
    rows: pd.DataFrame,
    group_totals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Interleave group subtotal rows with their members, in draw order.

    One frame, so every panel of the by-cell-type figure draws the same rows in
    the same order as the table — a figure and a table that disagree on row order
    are read as disagreeing on the numbers.

    Args:
        rows: Output of :func:`summarize_by_cell_type`.
        group_totals: Its group subtotals, or None for a flat list.

    Returns:
        The rows in draw order with two added columns: ``label`` (what to print,
        the group name on a subtotal row) and ``is_group``.
    """

    if not len(rows):
        return rows
    if group_totals is None or "group" not in rows.columns:
        display = rows.copy()
        display["label"] = display["cell_type"]
        display["is_group"] = False
        return display.reset_index(drop=True)

    # A group whose single member IS the group would print the same numbers twice,
    # once on the bar and once indented under it; keep the bar and drop the member.
    # The table drops the same rows through the same helper, so the two agree.
    redundant = redundant_group_members(rows, name="cell_type", group="group")

    blocks: list[pd.DataFrame] = []
    for group in dict.fromkeys(rows["group"]):
        if group in group_totals.index:
            bar = group_totals.loc[[group]].reset_index(drop=True)
            bar["cell_type"] = group
            bar["group"] = group
            bar["label"] = group
            bar["is_group"] = True
            blocks.append(bar)
        members = rows.loc[(rows["group"] == group) & ~redundant].copy()
        if not len(members):
            continue
        members["label"] = members["cell_type"]
        members["is_group"] = False
        blocks.append(members)
    return pd.concat(blocks, ignore_index=True)


def summarize_rules(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Summarise each QC rule's gross and unique contribution.

    Args:
        frame: Output of :func:`assemble_qc_frame`.
        thresholds: Optional threshold table used to build readable rule labels.

    Returns:
        One row per rule with ``label``, ``n_failed`` and ``n_unique``, sorted by
        ``n_failed`` descending.
    """

    rule_columns = [c for c in frame.columns if c.startswith("rule:")]
    if not rule_columns:
        return pd.DataFrame(columns=["rule", "label", "n_failed", "n_unique"])

    flags = frame[rule_columns].astype(bool)
    n_rules_hit = flags.sum(axis=1)
    bounds = _threshold_lookup(thresholds)

    records = []
    for column in rule_columns:
        rule = column.removeprefix("rule:")
        failed = flags[column]
        records.append(
            {
                "rule": rule,
                "label": _rule_label(rule, bounds.get(rule)),
                "n_failed": int(failed.sum()),
                # Cells no other rule caught: drop this rule and they survive.
                "n_unique": int((failed & n_rules_hit.eq(1)).sum()),
            }
        )
    return pd.DataFrame(records).sort_values("n_failed", ascending=False).reset_index(drop=True)


def _threshold_lookup(
    thresholds: pd.DataFrame | None,
) -> dict[str, tuple[str, float | None, float | None]]:
    """Map rule name to (metric, lower, upper) from the threshold table."""

    lookup: dict[str, tuple[str, float | None, float | None]] = {}
    if thresholds is None or "rule_name" not in thresholds.columns:
        return lookup
    for _, record in thresholds.iterrows():
        lower = record.get("lower")
        upper = record.get("upper")
        lookup[str(record["rule_name"])] = (
            str(record.get("metric", "")),
            float(lower) if pd.notna(lower) else None,
            float(upper) if pd.notna(upper) else None,
        )
    return lookup


def _rule_family(rule: str) -> str:
    """Name the policy a rule came from, for the parenthetical in its label.

    The family is what tells a reader whether a number was chosen or estimated,
    and by what. It is read off the rule-name prefix rather than the threshold
    table's ``source`` column so that a replot of an older run, whose table may
    predate a source value, still labels correctly.
    """

    # Match the most specific prefix first, since "mad_mito_" is also "mad_".
    if rule.startswith("mixture_"):
        # Name the method, not the mechanism: a reviewer can look up miQC.
        return "miQC"
    if rule.startswith("mad_"):
        return "MAD"
    return "fixed"


def _rule_label(rule: str, bound: tuple[str, float | None, float | None] | None) -> str:
    """Turn a rule name plus its bounds into something a reader can act on.

    ``mad_mito_pct_counts_mito`` says nothing; "Mitochondrial % > 6.1 (MAD)"
    says both what was tested and where the line landed.
    """

    family = _rule_family(rule)
    if bound is None:
        return f"{rule.replace('_', ' ')} ({family})"

    metric, lower, upper = bound
    name = METRIC_LABELS.get(metric, metric.replace("_", " ") or rule)
    # MAD bounds on log1p metrics are meaningless printed in log space.
    if metric.startswith("log1p_"):
        lower = float(np.expm1(lower)) if lower is not None else None
        upper = float(np.expm1(upper)) if upper is not None else None

    # A lower bound at or below zero excludes nothing — every QC metric here is a
    # count or a percentage — and printing it ("mitochondrial % outside
    # -0.51–5.13") makes a one-sided rule look two-sided.
    if lower is not None and lower <= 0.0:
        lower = None

    if lower is not None and upper is not None:
        text = f"{name} outside {_bound_text(lower)}–{_bound_text(upper)}"
    elif upper is not None:
        text = f"{name} > {_bound_text(upper)}"
    elif lower is not None:
        text = f"{name} < {_bound_text(lower)}"
    else:
        text = name
    return f"{text} ({family})"


def _bound_text(value: float) -> str:
    """Format a threshold without exponents: "47,300", not "4.73e+04"."""

    if abs(value) >= 1000.0:
        return f"{value:,.0f}"
    return f"{value:,.3g}"


def _axis_bounds(
    thresholds: pd.DataFrame | None,
    metric: str,
) -> tuple[float | None, float | None]:
    """Tightest applied bounds for a metric, in raw (non-log) units.

    A metric can be bounded by several rules at once — a fixed floor and a MAD
    interval, say. The binding constraint is the tightest of them, which is what
    a shaded exclusion region should show.
    """

    if thresholds is None or "metric" not in thresholds.columns:
        return None, None

    lower_values: list[float] = []
    upper_values: list[float] = []
    for _, record in thresholds.iterrows():
        name = str(record.get("metric", ""))
        if name not in {metric, f"log1p_{metric}"}:
            continue
        transform = np.expm1 if name.startswith("log1p_") else (lambda v: v)
        if pd.notna(record.get("lower")):
            lower_values.append(float(transform(float(record["lower"]))))
        if pd.notna(record.get("upper")):
            upper_values.append(float(transform(float(record["upper"]))))
    return (
        max(lower_values) if lower_values else None,
        min(upper_values) if upper_values else None,
    )


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------


def _condition_colors(
    frame: pd.DataFrame,
    case_label: str | None = None,
) -> dict[str, str]:
    """Assign the two-colour condition palette in a stable order.

    Two hues only. More conditions fall back to grey rather than inventing a
    categorical palette here, which would bypass the project's palette check.
    """

    if "condition" not in frame.columns:
        return {}
    levels = [str(v) for v in pd.unique(frame["condition"].dropna())]
    if case_label and case_label in levels:
        control = [v for v in levels if v != case_label]
        ordered = ([control[0]] if control else []) + [case_label]
    else:
        ordered = sorted(levels)
    # Insertion order is control then case, because every legend built from this
    # iterates it and a legend reading case-then-control fights the design.
    palette: dict[str, str] = {}
    if len(ordered) >= 1:
        palette[ordered[0]] = NORMAL_BLUE
    if len(ordered) >= 2:
        palette[ordered[-1]] = LE_RED
    for level in levels:
        palette.setdefault(level, REMOVED_GREY)
    return palette


def order_samples(table: pd.DataFrame, *, case_label: str | None = None) -> pd.DataFrame:
    """Order a per-sample table donor-major, control before case.

    The reading order every QC figure and table shares: donors numerically
    (P2 before P10), and within a donor the control sample first so the eye moves
    control-to-case, never the reverse.

    Args:
        table: Per-sample table with any of ``donor``, ``condition``, ``sample``.
        case_label: Condition value that marks the case arm.

    Returns:
        The table reordered, with a reset index.
    """

    return _order_by_donor_then_condition(table, _condition_colors(table, case_label))


def _order_by_donor_then_condition(
    table: pd.DataFrame,
    palette: dict[str, str],
) -> pd.DataFrame:
    """Order rows donor-major, control before case within each donor.

    Sorting the condition column alphabetically is not good enough: "LE" precedes
    "Normal", so every donor's rows would read case-then-control and the eye would
    have to reverse each pair. Rank by the palette's control/case assignment
    instead, which is fixed by the design rather than by the label spelling.
    """

    ranks = {
        level: (0 if color == NORMAL_BLUE else 1 if color == LE_RED else 2)
        for level, color in palette.items()
    }
    ordered = table.copy()
    keys: list[str] = []
    if "donor" in ordered.columns:
        # Natural sort, so P2 precedes P10. Plain string sort gives P1, P10, P12,
        # P2 — which reads as a scrambled cohort on any donor list past nine.
        ordered["_donor_key"] = _natural_key(ordered["donor"])
        keys.append("_donor_key")
    if "condition" in ordered.columns and ranks:
        ordered["_condition_rank"] = ordered["condition"].astype(str).map(ranks).fillna(2)
        keys.append("_condition_rank")
    keys += [c for c in ("sample",) if c in ordered.columns]
    ordered = ordered.sort_values(keys).reset_index(drop=True)
    return ordered.drop(columns=["_donor_key", "_condition_rank"], errors="ignore")


def _natural_key(values: pd.Series) -> pd.Series:
    """Zero-pad every digit run so a lexical sort orders labels numerically."""
    return values.astype(str).str.replace(
        r"(\d+)", lambda match: match.group(1).zfill(6), regex=True
    )


def _panel_label(ax: Axes, letter: str, *, dx: float = -30.0, dy: float = 14.0) -> None:
    """Panel letter offset in POINTS from the axes' top-left corner.

    Points, not axes fractions: the offset then does not scale with panel width,
    so a narrow panel's letter cannot drift into its own title the way the old
    centred-title layout let "D" collide with "Doublet-score separation".
    """

    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
        va="center",
        ha="left",
        annotation_clip=False,
    )


# Points of title padding reserved when a subtitle sits underneath. The subtitle
# is an annotation, which the layout engine does not measure, so the title's pad
# is what reserves room for both lines. Too small and they overprint — which is
# how the old sheet ended up with "Doublet-score separation" written through its
# own panel letter.
_SUBTITLE_PAD = 17.0
_SUBTITLE_OFFSET = 4.0


def _title(ax: Axes, text: str, subtitle: str | None = None) -> None:
    """Left-aligned title, optionally with a lighter subtitle beneath it.

    Left alignment is what keeps titles clear of panel letters; a centred title
    on a narrow panel has nowhere to go but on top of the letter.
    """

    ax.set_title(
        text,
        loc="left",
        fontsize=9.5,
        fontweight="bold",
        color=TEXT,
        pad=_SUBTITLE_PAD if subtitle else 6.0,
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0, _SUBTITLE_OFFSET),
            textcoords="offset points",
            fontsize=7.5,
            color="#6B7280",
            va="bottom",
            ha="left",
            annotation_clip=False,
        )


def _xgrid(ax: Axes) -> None:
    """Light x-grid behind the data — the reference a bar length needs."""
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID_GREY, linewidth=0.6)
    ax.yaxis.grid(False)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def plot_cohort_funnel(ax: Axes, frame: pd.DataFrame) -> None:
    """
    Draw the cohort as one stacked bar: retained versus removed.

    The single most important number in a QC report, and the one the previous
    figure set reported as 100% pass on a run that dropped 503 cells.

    Args:
        ax: Target axes.
        frame: Output of :func:`assemble_qc_frame`.
    """

    n_in = int(len(frame))
    n_keep = int(frame["keep"].sum())
    n_drop = n_in - n_keep
    pct_drop = 100.0 * n_drop / n_in if n_in else 0.0

    ax.barh([0], [n_keep], color=KEPT_GREEN, height=0.55, linewidth=0)
    ax.barh([0], [n_drop], left=[n_keep], color=RULE_UNIQUE, height=0.55, linewidth=0)

    # Segments carry their own word ("Retained 3,294"), so the panel needs no
    # legend and cannot have one land on top of the bar. Text goes inside the
    # segment when it fits and outside when it does not, so a 2% drop stays
    # legible instead of being crushed into a sliver.
    for label, value, left, color in (
        ("Retained", n_keep, 0, KEPT_GREEN),
        ("Removed", n_drop, n_keep, RULE_UNIQUE),
    ):
        if not value:
            continue
        share = value / n_in if n_in else 0.0
        text = f"{label}  {value:,} ({100 * share:.1f}%)"
        inside = share > 0.30
        ax.text(
            left + (value / 2 if inside else value + n_in * 0.012),
            0,
            text,
            ha="center" if inside else "left",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white" if inside else color,
        )

    # Headroom for the outside label so it cannot run off the axes.
    ax.set_xlim(0, n_in * 1.22)
    ax.set_yticks([])
    ax.set_xlabel("Cells")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    # The subtitle carries cohort shape rather than repeating the drop, which the
    # bar segments already state.
    parts = []
    if "sample" in frame.columns:
        parts.append(f"{frame['sample'].nunique():,} samples")
    if "donor" in frame.columns:
        parts.append(f"{frame['donor'].nunique():,} donors")
    _title(
        ax,
        f"{n_in:,} cells in, {n_keep:,} retained",
        " · ".join(parts) if parts else f"{n_drop:,} removed ({pct_drop:.1f}%)",
    )


def plot_rule_attribution(ax: Axes, rule_table: pd.DataFrame, n_cells: int) -> None:
    """
    Draw per-rule failures, separating each rule's unique contribution.

    Rules overlap heavily — a dying cell trips the mito rule, the complexity rule
    and the count rule at once — so a plain per-rule bar chart implies more
    independent filtering than happened. The dark inner bar is the honest number:
    cells that rule alone caught. A rule with no dark bar is doing no work of its
    own.

    Args:
        ax: Target axes.
        rule_table: Output of :func:`summarize_rules`.
        n_cells: Input cell count, for percentage annotations.
    """

    if not len(rule_table):
        ax.set_axis_off()
        return

    table = rule_table.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(table))

    ax.barh(positions, table["n_failed"], color=RULE_FILL, height=0.68, linewidth=0)
    ax.barh(positions, table["n_unique"], color=RULE_UNIQUE, height=0.68, linewidth=0)

    limit = max(1, int(table["n_failed"].max()))
    for position, row in zip(positions, table.itertuples(), strict=True):
        pct = 100.0 * row.n_failed / n_cells if n_cells else 0.0
        ax.text(
            row.n_failed + limit * 0.02,
            position,
            f"{row.n_failed:,}  ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=7.5,
            color=TEXT,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(table["label"], fontsize=7.5)
    ax.set_xlim(0, limit * 1.28)
    ax.set_xlabel("Cells failing rule")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _xgrid(ax)
    # No legend: the subtitle already names the encoding, and a two-entry legend
    # has nowhere to go here. Above the axes it lands on the subtitle; inside, it
    # covers the short bars at the bottom of the sorted list, which are the ones
    # already hardest to read.
    _title(ax, "Attribution by rule", "dark = cells this rule alone removed")


def plot_paired_dumbbell(
    ax: Axes,
    sample_table: pd.DataFrame,
    value: str,
    *,
    case_label: str | None = None,
    thresholds: pd.DataFrame | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    show_legend: bool = True,
) -> None:
    """
    Draw one row per donor, the donor's two conditions as connected dots.

    This is the panel the old boxplot wall could not be. With a paired design the
    quantity of interest is the within-donor difference, and a dumbbell shows all
    nine of them at once: the direction of each donor's shift, the spread of
    those shifts, and any donor that goes the other way. Rows sort by the
    difference so the pattern is the shape of the panel.

    Args:
        ax: Target axes.
        sample_table: Output of :func:`summarize_by_sample`.
        value: Column to compare across conditions.
        case_label: Condition to treat as the case arm.
        thresholds: Optional threshold table. When the applied bound for ``value``
            lands where the donors are, it is drawn as a reference line — the
            answer to "which donor sat on the cut?". See the note below on units.
        xlabel: Axis label; defaults to the metric's display name.
        title: Panel title.
        subtitle: Panel subtitle.
        show_legend: Draw the per-panel condition legend. False when a caller
            tiles several of these and carries one shared legend for the figure —
            three copies of a two-item key is redundant ink, and it competes with
            the paired-test annotation for the one empty band on the axes.

    Raises:
        QCPanelError: If the table lacks donor/condition structure.
    """

    required = {"donor", "condition", value}
    missing = required.difference(sample_table.columns)
    if missing:
        raise QCPanelError(
            f"Paired dumbbell needs columns {sorted(required)}; missing {sorted(missing)}."
        )

    palette = _condition_colors(sample_table, case_label=case_label)
    levels = [level for level, color in palette.items() if color != REMOVED_GREY]
    if len(levels) != 2:
        raise QCPanelError(
            f"Paired dumbbell needs exactly two conditions, found {sorted(palette)}."
        )
    control_level = next(level for level in levels if palette[level] == NORMAL_BLUE)
    case_level = next(level for level in levels if palette[level] == LE_RED)

    wide = sample_table.pivot_table(
        index="donor", columns="condition", values=value, aggfunc="median"
    )
    wide = wide.reindex(columns=[control_level, case_level]).dropna()
    if wide.empty:
        ax.set_axis_off()
        return

    # Sort by within-donor difference: the panel's shape becomes the result.
    delta = wide[case_level] - wide[control_level]
    wide = wide.loc[delta.sort_values().index]
    positions = np.arange(len(wide))

    # Read the two columns positionally. Condition levels are arbitrary strings,
    # so attribute access on itertuples cannot be relied on to name them.
    control_values = wide[control_level].to_numpy(dtype=float)
    case_values = wide[case_level].to_numpy(dtype=float)

    for position, control_value, case_value in zip(
        positions, control_values, case_values, strict=True
    ):
        # Connector colour encodes direction, so sign reads without the axis.
        ax.plot(
            [control_value, case_value],
            [position, position],
            color=LE_RED if case_value > control_value else NORMAL_BLUE,
            linewidth=1.4,
            alpha=0.45,
            solid_capstyle="round",
            zorder=1,
        )
    # Case dot first and larger, control dot on top and smaller. A donor whose two
    # conditions agree to within a rounding error — and there is usually one —
    # otherwise loses a marker entirely to overplotting and reads as missing data.
    # Nested, it reads as a ring: same value, both arms present.
    ax.scatter(
        case_values,
        positions,
        s=58,
        color=LE_RED,
        zorder=3,
        linewidth=0,
        label=case_level,
    )
    ax.scatter(
        control_values,
        positions,
        s=30,
        color=NORMAL_BLUE,
        zorder=4,
        linewidth=0.6,
        edgecolor="white",
        label=control_level,
    )

    # The applied QC bound, when it lands where the donors are. Note the units:
    # this axis is donor MEDIANS and the bound was applied per CELL, so the line
    # does not say "these donors were removed" — it says how close the cohort's
    # centre sat to the cut, which is what a reviewer asks about a marginal donor.
    # Refused when the bound is far outside the data, for the same reason the
    # joint scatter shades after plotting: a 48,000-UMI ceiling on an axis of
    # 4,000–10,000 medians would squash every dumbbell to draw one line.
    in_view: list[float] = []
    if thresholds is not None:
        low, high = ax.get_xlim()
        span = high - low
        in_view = [
            bound
            for bound in _axis_bounds(thresholds, value)
            if bound is not None and low - 0.25 * span <= bound <= high + 0.25 * span
        ]
        for index, bound in enumerate(in_view):
            ax.axvline(
                bound,
                color=RULE_UNIQUE,
                linestyle=(0, (4, 2.5)),
                linewidth=0.9,
                zorder=2,
                # A floor and a ceiling are one legend entry, not two.
                label="QC bound" if index == 0 else None,
            )
        if in_view:
            pad = 0.03 * span
            ax.set_xlim(
                min([low, *(bound - pad for bound in in_view)]),
                max([high, *(bound + pad for bound in in_view)]),
            )

    n_up = int((case_values > control_values).sum())
    ax.set_yticks(positions)
    ax.set_yticklabels(wide.index, fontsize=8)
    # An empty band below the last donor, so the legend and the paired-test line
    # have somewhere to sit that is not on top of a dumbbell. Rows are sorted by
    # effect size, so both ends of the axes hold data and there is no naturally
    # empty corner. The band is sized to the legend: a third entry (the QC bound)
    # in a two-row band puts the box over the bottom donor, which on a metric that
    # drove attrition is exactly the donor the reader came for.
    legend_rows = 2 + (1 if (show_legend and in_view) else 0)
    y_floor = -(0.5 + 0.42 * legend_rows)
    ax.set_ylim(y_floor, len(wide) - 0.5)
    ax.set_xlabel(xlabel or METRIC_LABELS.get(value, value.replace("_", " ")))
    ax.set_ylabel("Donor")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _xgrid(ax)
    _title(
        ax,
        title or f"{METRIC_LABELS.get(value, value)} by donor",
        subtitle
        if subtitle is not None
        else f"higher in {case_level} in {n_up} of {len(wide)} donors",
    )
    # Control first, matching the design and every other legend on the sheet; the
    # draw order is the reverse of that, for the overplotting reason above.
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        # Conditions first in design order, then anything else drawn — the QC
        # bound, when this metric has one on-axis. Dropping it would leave an
        # unexplained dashed line on the panel.
        order = [labels.index(control_level), labels.index(case_level)]
        order += [i for i in range(len(labels)) if i not in order]
        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="lower right",
            # Framed, unlike the sheet's other legends: the bound is drawn at the
            # tightest applied threshold, which on a metric that drove attrition is
            # near the high end of the axis — right where this legend sits.
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.88,
            fontsize=7.5,
            handletextpad=0.2,
            borderaxespad=0.4,
        )

    # The paired test, in the empty row below the last donor. This panel is
    # already donor-level -- one point per donor per
    # arm -- so the test that belongs on it is the paired one over those donors,
    # and "higher in 5 of 9" without a p-value is the weakest form of the result.
    # Restricted to the donors actually drawn, so the number and the picture
    # cannot disagree: `wide` dropped any donor missing an arm.
    drawn = sample_table.loc[sample_table["donor"].isin(wide.index)]
    test = two_group_test_on_donor_medians(
        drawn,
        value_col=value,
        group_col="condition",
        donor_col="donor",
        group1=control_level,
        group2=case_level,
    )
    if test is not None:
        # Both tests are Wilcoxon's: signed-rank when paired, rank-sum (a.k.a.
        # Mann-Whitney) when the arms are disjoint.
        family = "signed-rank" if test.test == "wilcoxon_signed_rank" else "rank-sum"
        ax.text(
            0.0,
            y_floor + 0.30,
            f"Wilcoxon {family} p = {test.p_value:.2g}",
            transform=ax.get_yaxis_transform(),
            fontsize=7.5,
            color=TEXT,
            va="center",
            ha="left",
        )


def plot_sample_attrition(
    ax: Axes,
    sample_table: pd.DataFrame,
    *,
    case_label: str | None = None,
) -> None:
    """
    Draw percent removed per sample, donors grouped, worst sample first.

    Sample names sit on the y-axis and read horizontally. That is the whole
    reason this is a horizontal bar chart: eighteen rotated labels are
    unreadable, and the sample that lost half its cells is the one you most need
    to be able to name.

    Args:
        ax: Target axes.
        sample_table: Output of :func:`summarize_by_sample`.
        case_label: Condition to treat as the case arm.
    """

    if not len(sample_table) or "pct_removed" not in sample_table.columns:
        ax.set_axis_off()
        return

    table = sample_table.sort_values("pct_removed").reset_index(drop=True)
    palette = _condition_colors(table, case_label=case_label)
    colors = (
        [palette.get(str(c), REMOVED_GREY) for c in table["condition"]]
        if "condition" in table.columns
        else [REMOVED_GREY_DARK] * len(table)
    )
    positions = np.arange(len(table))

    ax.barh(positions, table["pct_removed"], color=colors, height=0.7, linewidth=0)

    cohort_pct = (
        100.0 * table["cells_removed"].sum() / table["cells_in"].sum()
        if table["cells_in"].sum()
        else 0.0
    )
    ax.axvline(cohort_pct, color=TEXT, linestyle=(0, (3, 2)), linewidth=0.9, zorder=4)
    limit = float(table["pct_removed"].max())
    ax.annotate(
        f"cohort {cohort_pct:.1f}%",
        xy=(cohort_pct, len(table) - 0.4),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7,
        color=TEXT,
        va="center",
        ha="left",
        zorder=5,
        path_effects=[path_effects.withStroke(linewidth=2.2, foreground="white")],
    )

    # A white halo, because the cohort reference line lands mid-text on exactly the
    # samples sitting near the cohort mean — which is most of them.
    halo = [path_effects.withStroke(linewidth=2.2, foreground="white")]
    for position, row in zip(positions, table.itertuples(), strict=True):
        ax.text(
            row.pct_removed + limit * 0.02,
            position,
            f"{row.cells_removed:,}/{row.cells_in:,}",
            va="center",
            ha="left",
            fontsize=6.8,
            color="#6B7280",
            zorder=5,
            path_effects=halo,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(table["sample"], fontsize=7.5)
    ax.set_xlim(0, limit * 1.24 if limit else 1.0)
    ax.set_xlabel("Cells removed (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _xgrid(ax)
    _title(ax, "Attrition per sample", "label = removed / input")
    if palette:
        ax.legend(
            handles=[Patch(facecolor=color, label=level) for level, color in palette.items()],
            loc="lower right",
            frameon=False,
            fontsize=7.5,
            handlelength=1.1,
            handleheight=0.9,
        )


def plot_joint_scatter(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    thresholds: pd.DataFrame | None = None,
    color_metric: str = "pct_counts_mito",
    show_title: bool = True,
) -> tuple[Axes, mpl.collections.PathCollection | None]:
    """
    Draw UMIs against genes: the core QC scatter.

    Three things the previous version lacked: removed cells are still on the plot
    (in grey, so you see what left and where it sat), the excluded range is shaded
    rather than marked with a bare dashed line, and retained cells are coloured by
    mitochondrial fraction — usually the reason a cell in the low-count corner is
    there at all.

    Args:
        ax: Target axes.
        frame: Output of :func:`assemble_qc_frame`.
        thresholds: Optional threshold table for the shaded bounds.
        color_metric: Metric colouring the retained cells.
        show_title: Whether to title this axes. False when a marginal above it
            carries the title instead.

    Returns:
        The axes and the coloured scatter (None when ``color_metric`` is absent),
        so a caller can attach a colourbar where it has room for one.
    """

    x_column, y_column = "total_counts", "n_genes_by_counts"
    data = frame.dropna(subset=[x_column, y_column])
    kept = data.loc[data["keep"]]
    removed = data.loc[~data["keep"]]

    ax.scatter(
        removed[x_column],
        removed[y_column],
        s=5,
        color=REMOVED_GREY,
        alpha=0.55,
        linewidth=0,
        zorder=1,
        rasterized=True,
    )
    scatter = None
    if color_metric in kept.columns and kept[color_metric].notna().any():
        values = kept[color_metric]
        scatter = ax.scatter(
            kept[x_column],
            kept[y_column],
            c=values,
            s=6,
            cmap="magma_r",
            # Clip at the 98th percentile so a handful of extreme cells do not
            # flatten the whole colour range.
            vmin=0.0,
            vmax=float(np.nanpercentile(values, 98)) or None,
            alpha=0.85,
            linewidth=0,
            zorder=2,
            rasterized=True,
        )
    else:
        ax.scatter(
            kept[x_column],
            kept[y_column],
            s=6,
            color=NORMAL_BLUE,
            alpha=0.7,
            linewidth=0,
            zorder=2,
            rasterized=True,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(METRIC_LABELS[x_column])
    ax.set_ylabel(METRIC_LABELS[y_column])
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID_GREY, linewidth=0.6)

    # Label the 1-2-5 decade subdivisions as plain numbers. A default log axis
    # over the ~300–30,000 range these metrics occupy prints a single "10^3" and
    # nothing else, which makes the panel unreadable as a QC reference.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mpl.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(
            # Integers with thousands separators over the usual range, but one
            # decimal below ten, or a tiny-cohort axis prints "2" three times.
            mpl.ticker.FuncFormatter(
                lambda value, _: f"{value:,.0f}" if value >= 10 else f"{value:g}"
            )
        )
        axis.set_minor_formatter(mpl.ticker.NullFormatter())

    # Shade the excluded ranges only after the points have fixed the view. A span
    # extends the data limits, so drawing "everything above the upper bound"
    # first would rescale the axis to the span's far edge and squash the cloud.
    # The bound itself gets a line; the shading only hints which side is excluded.
    # Shading alone was misleading here: the excluded ranges cover a large area of
    # metric space but hold ~13% of the cells, so a solid tint reads as though half
    # the cohort was cut.
    x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
    for span, line, column, limits in (
        (ax.axvspan, ax.axvline, x_column, x_limits),
        (ax.axhspan, ax.axhline, y_column, y_limits),
    ):
        lower, upper = _axis_bounds(thresholds, column)
        for bound, edge in ((lower, limits[0]), (upper, limits[1])):
            if bound is None or not (limits[0] < bound < limits[1]):
                continue
            span(*sorted((edge, bound)), color=RULE_UNIQUE, alpha=0.05, linewidth=0, zorder=0)
            line(bound, color=RULE_UNIQUE, linestyle=(0, (4, 2.5)), linewidth=0.9, zorder=3)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    if scatter is not None:
        # White backing under the colourbar. The corner is empty of CELLS, which
        # is why the bar goes there, but it is not empty of ink: a QC bound and
        # its shaded exclusion run straight through it, and a dashed red line
        # crossing a colour ramp's tick labels reads as a plotting error.
        ax.add_patch(
            mpl.patches.Rectangle(
                (0.015, 0.875),
                0.40,
                0.125,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="none",
                alpha=0.88,
                # Above the bounds (zorder 3) but below the inset axes, which as a
                # child axes always draws after its parent.
                zorder=4,
            )
        )
        # Colourbar as a slim inset in the plot's empty upper-left corner, kept
        # out of the layout solve: constrained layout treats an inset as a sibling
        # to place, and a 2.5%-height axes collapses the solve for the whole figure.
        cax = ax.inset_axes((0.04, 0.94, 0.30, 0.022))
        cax.set_in_layout(False)
        bar = ax.figure.colorbar(scatter, cax=cax, orientation="horizontal")
        bar.set_label(METRIC_LABELS.get(color_metric, color_metric), fontsize=7, labelpad=2)
        bar.ax.xaxis.set_label_position("top")
        bar.ax.tick_params(labelsize=6.2, length=2, pad=1)
        bar.outline.set_visible(False)

    # Take the retained swatch from the middle of the colour map actually in use,
    # so the key points at the dots on the plot rather than at some other colour.
    kept_swatch = NORMAL_BLUE if scatter is None else mpl.colormaps["magma_r"](0.6)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=4.5,
                color=kept_swatch,
                label=f"Retained (n={len(kept):,})",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=4.5,
                color=REMOVED_GREY,
                label=f"Removed (n={len(removed):,})",
            ),
            Line2D(
                [], [], color=RULE_UNIQUE, linestyle=(0, (4, 2.5)), linewidth=0.9, label="QC bound"
            ),
        ],
        loc="lower right",
        # Framed, unlike the other legends on the sheet: the lower-right corner
        # holds the low-count exclusion shading and usually an upper QC bound, and
        # a dashed line through "Retained (n=1,864)" is unreadable. White fill,
        # no edge -- it blocks the line without drawing a box around itself.
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        fontsize=7,
        handletextpad=0.3,
        borderaxespad=0.5,
    )
    if show_title:
        _title(ax, "Library complexity", "removed cells stay on the plot, in grey")
    return ax, scatter


def plot_joint_marginals(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    size: float = 0.19,
    gap: float = 0.012,
) -> tuple[Axes, Axes]:
    """
    Attach marginal histograms directly above and right of a joint scatter.

    Built as insets pinned to the scatter's own axes rather than as gridspec
    siblings. Under constrained layout, ``hspace`` is a fraction of axes height,
    so a marginal in its own gridspec row ends up separated from the scatter by a
    gap as tall as the marginal itself — the two stop reading as one plot. Insets
    with an explicit gap keep them joined at any figure size.

    Histograms, not KDEs: a KDE on a bounded quantity smears mass past the bound,
    which is how the previous violin panels implied cells with negative UMIs.

    Args:
        ax: The joint scatter axes to attach to.
        frame: Output of :func:`assemble_qc_frame`.
        size: Marginal depth as a fraction of the scatter's extent.
        gap: Gap between scatter and marginal, as a fraction of the extent.

    Returns:
        The top and right marginal axes.
    """

    x_column, y_column = "total_counts", "n_genes_by_counts"
    data = frame.dropna(subset=[x_column, y_column])
    kept = data.loc[data["keep"]]
    removed = data.loc[~data["keep"]]

    # Left in the layout deliberately: `set_in_layout(False)` would also drop them
    # from the tight bounding box at save time, and the right marginal would be
    # cropped off the edge of the file.
    ax_top = ax.inset_axes((0.0, 1.0 + gap, 1.0, size), sharex=ax)
    ax_right = ax.inset_axes((1.0 + gap, 0.0, size, 1.0), sharey=ax)

    for marginal, column, orientation in (
        (ax_top, x_column, "vertical"),
        (ax_right, y_column, "horizontal"),
    ):
        finite = data[column].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if finite.size == 0:
            marginal.set_axis_off()
            continue
        bins = np.logspace(np.log10(finite.min()), np.log10(finite.max()), 48)
        for subset, color, alpha in ((kept, KEPT_GREEN, 0.8), (removed, REMOVED_GREY, 0.9)):
            if not len(subset):
                continue
            marginal.hist(
                subset[column],
                bins=bins,
                orientation=orientation,
                color=color,
                alpha=alpha,
                linewidth=0,
            )
        marginal.set_axis_off()

    return ax_top, ax_right


def plot_mito_mixture(
    ax: Axes,
    frame: pd.DataFrame,
    models: pd.DataFrame,
    *,
    mito_metric: str = "pct_counts_mito",
    complexity_metric: str = "n_genes_by_counts",
    posterior_column: str = MIQC_POSTERIOR_COLUMN,
    posterior_cutoff: float = 0.75,
    ceiling: float | None = None,
    show_title: bool = True,
) -> None:
    """Draw the fitted mitochondrial mixture that produced the cut.

    Every other panel on the sheet shows what the filter DID. This one shows why,
    and it is the only panel that can: an adaptive mitochondrial threshold is not
    a number a reader can check against a distribution, it is a model, and the
    question "would I have drawn this cut?" is answerable only against the model's
    own two components.

    Three things are drawn on one pair of axes, and the distance between them is
    the point:

    * The two fitted regression lines. Damaged cells leak cytoplasmic mRNA, so
      their mitochondrial fraction rises as complexity falls; the compromised
      component is the steeper, higher-intercept line. Seeing both lines is what
      shows the cut is complexity-aware rather than flat.
    * The model's own decision boundary, contoured where the posterior crosses
      the cutoff. Curved, because the components have different variances.
    * The bound actually applied, as a horizontal line. When the monotone
      projection is on, the engine reduces that curve to its most permissive
      mitochondrial value so that no cell is removed while a dirtier one at the
      same complexity survives. The gap between the curve and the line is the
      price of that guarantee, and a reader who cannot see it has to take the
      projection on trust.

    Args:
        ax: Target axes.
        frame: Output of :func:`assemble_qc_frame`, carrying the two metrics and,
            when the run recorded it, the per-cell posterior.
        models: The fitted model table, one row per group.
        mito_metric: Mitochondrial column, the regression response.
        complexity_metric: Complexity column, the regression predictor.
        posterior_column: Per-cell compromised posterior, when present.
        posterior_cutoff: Posterior above which a cell was called compromised.
        ceiling: The applied mitochondrial bound, when one was derived.
        show_title: Draw the panel title.

    Raises:
        QCPanelError: If neither metric is available to plot.
    """

    missing = [c for c in (mito_metric, complexity_metric) if c not in frame.columns]
    if missing:
        raise QCPanelError(
            f"The mixture panel needs columns {sorted(missing)}. "
            f"Present: {sorted(frame.columns)[:20]}"
        )

    data = frame.dropna(subset=[mito_metric, complexity_metric])
    if data.empty:
        ax.set_axis_off()
        return

    x = data[complexity_metric].to_numpy(dtype=float)
    y = data[mito_metric].to_numpy(dtype=float)

    # Colour by posterior when the run recorded it, and fall back to the
    # keep/remove split when it did not. The fallback matters: runs that predate
    # the posterior being published still have a fitted model worth drawing, and
    # a panel that refuses to render is worse than one drawn in two colours.
    posterior = (
        pd.to_numeric(data[posterior_column], errors="coerce").to_numpy(dtype=float)
        if posterior_column in data.columns
        else None
    )
    if posterior is not None and np.isfinite(posterior).any():
        points = ax.scatter(
            x,
            y,
            c=posterior,
            cmap="magma_r",
            vmin=0.0,
            vmax=1.0,
            s=5,
            linewidth=0,
            alpha=0.85,
            rasterized=True,
        )
        bar = ax.figure.colorbar(points, ax=ax, pad=0.02, fraction=0.045)
        bar.set_label("P(compromised)", fontsize=7.5)
        bar.ax.tick_params(labelsize=7)
        bar.outline.set_visible(False)
    else:
        for keep, color, label in (
            (True, KEPT_GREEN, "Retained"),
            (False, REMOVED_GREY, "Removed"),
        ):
            subset = data.loc[data["keep"] == keep]
            if not len(subset):
                continue
            ax.scatter(
                subset[complexity_metric],
                subset[mito_metric],
                color=color,
                s=5,
                linewidth=0,
                alpha=0.85,
                label=label,
                rasterized=True,
            )

    # The fitted lines, one pair per model. Drawn only while they can still be
    # told apart: a per-sample fit on a 40-sample cohort is 80 lines, which
    # obscures the very scatter they are meant to explain.
    grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
    drawable = models.head(3) if len(models) <= 3 else models.iloc[:0]
    for position, (_, model) in enumerate(drawable.iterrows()):
        for prefix, color, style, name in (
            ("intact", NORMAL_BLUE, "-", "Intact"),
            ("compromised", RULE_UNIQUE, (0, (5, 2)), "Compromised"),
        ):
            intercept = model.get(f"{prefix}_intercept")
            slope = model.get(f"{prefix}_slope")
            if not (np.isfinite(intercept) and np.isfinite(slope)):
                continue
            ax.plot(
                grid,
                intercept + slope * grid,
                color=color,
                linestyle=style,
                linewidth=1.5,
                zorder=4,
                label=f"{name} component" if position == 0 else None,
                path_effects=[
                    path_effects.Stroke(linewidth=3.0, foreground="white"),
                    path_effects.Normal(),
                ],
            )

    # The model's own boundary, contoured rather than solved. Equating the
    # posterior to the cutoff gives a quadratic in the mitochondrial percentage
    # whose root count depends on the two variances, so a contour is both shorter
    # and more honest than picking a branch. Needs a single model with both
    # variances recorded; older runs stored neither.
    if len(models) == 1:
        model = models.iloc[0]
        boundary = _mixture_boundary(
            model,
            complexity=grid,
            mito_limits=(float(np.nanmin(y)), float(np.nanmax(y))),
            posterior_cutoff=posterior_cutoff,
        )
        if boundary is not None:
            ax.contour(
                *boundary,
                levels=[posterior_cutoff],
                colors=[TEXT],
                linewidths=1.1,
                linestyles=[(0, (1, 1.6))],
                zorder=5,
            )
            ax.plot(
                [],
                [],
                color=TEXT,
                linestyle=(0, (1, 1.6)),
                linewidth=1.1,
                label=f"P = {posterior_cutoff:g} boundary",
            )

    # The bound that was actually applied.
    if ceiling is not None and np.isfinite(ceiling):
        ax.axhline(
            ceiling,
            color=RULE_UNIQUE,
            linewidth=1.2,
            zorder=6,
            label=f"Applied bound ({ceiling:.2g}%)",
        )
        ax.axhspan(ceiling, max(float(np.nanmax(y)), ceiling) * 1.05, color=RULE_UNIQUE, alpha=0.06)

    ax.set_xlabel(METRIC_LABELS.get(complexity_metric, complexity_metric))
    ax.set_ylabel(METRIC_LABELS.get(mito_metric, mito_metric))
    ax.set_ylim(0.0, float(np.nanmax(y)) * 1.05)
    _xgrid(ax)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        fontsize=7,
        handletextpad=0.4,
        borderaxespad=0.4,
    )
    if show_title:
        weight = models["compromised_weight"].mean() if "compromised_weight" in models else np.nan
        removed = int((~data["keep"]).sum())
        detail = f"{removed:,} cells removed"
        if np.isfinite(weight):
            detail = f"compromised component {weight:.0%} of cells · {detail}"
        _title(ax, "Mitochondrial mixture model", detail)


def _mixture_boundary(
    model: pd.Series,
    *,
    complexity: np.ndarray,
    mito_limits: tuple[float, float],
    posterior_cutoff: float,
    resolution: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Evaluate the compromised posterior on a grid, for contouring.

    Recomputes the model's reasoning from the recorded parameters alone, which is
    the same property the mixture's own round-trip test asserts: a record that
    cannot reproduce its posterior cannot justify its cut either.

    Args:
        model: One row of the fitted model table.
        complexity: Complexity values spanning the panel's x axis.
        mito_limits: Low and high mitochondrial percentage to span.
        posterior_cutoff: Unused in the computation; kept so a caller passing a
            cutoff outside the grid gets a boundary that still brackets it.
        resolution: Grid points along the mitochondrial axis.

    Returns:
        The x grid, y grid and posterior surface, or None when the record lacks
        the variances the posterior needs.
    """

    try:
        weight = float(model["compromised_weight"])
        variances = (float(model["compromised_variance"]), float(model["intact_variance"]))
        lines = (
            (float(model["compromised_intercept"]), float(model["compromised_slope"])),
            (float(model["intact_intercept"]), float(model["intact_slope"])),
        )
    except (KeyError, TypeError, ValueError):
        return None

    # Older records defaulted both variances to zero, which is not a fit.
    if not all(np.isfinite(v) and v > 0 for v in variances):
        return None
    if not (0.0 < weight < 1.0):
        return None

    low, high = mito_limits
    if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
        return None
    mito_grid = np.linspace(max(0.0, low), high, resolution)
    mesh_x, mesh_y = np.meshgrid(complexity, mito_grid)

    # Log density per component, then the softmax over the two. Same expression
    # the fit maximises, so the surface is the model's own posterior rather than a
    # reconstruction of it.
    weights = (weight, 1.0 - weight)
    densities = []
    for (intercept, slope), variance, component_weight in zip(
        lines, variances, weights, strict=True
    ):
        residual = mesh_y - (intercept + slope * mesh_x)
        densities.append(
            np.log(component_weight)
            - 0.5 * np.log(2.0 * np.pi * variance)
            - residual**2 / (2.0 * variance)
        )
    stacked = np.stack(densities)
    surface = np.exp(stacked[0] - stacked.max(axis=0)) / np.exp(stacked - stacked.max(axis=0)).sum(
        axis=0
    )
    return mesh_x, mesh_y, surface


def plot_sample_matrix(
    ax: Axes,
    sample_table: pd.DataFrame,
    *,
    metrics: Sequence[str] = MATRIX_METRICS,
    case_label: str | None = None,
) -> None:
    """
    Draw samples against metrics as a robust z-scored heatmap.

    Eighteen samples times six metrics is 108 numbers. As a wall of boxplots that
    is six figures nobody cross-references; as one matrix it is a single glance,
    and an outlier sample shows up as a coloured row rather than as one tall
    whisker in panel four. Scaling is median/IQR per metric so a single extreme
    sample cannot wash the map out, and the condition strip on the left replaces
    a legend the eye would have to keep re-checking.

    Args:
        ax: Target axes.
        sample_table: Output of :func:`summarize_by_sample`.
        metrics: Metric columns to include, in display order.
        case_label: Condition to treat as the case arm.
    """

    usable = [
        m
        for m in metrics
        if m in sample_table.columns
        and sample_table[m].notna().any()
        # A metric that is constant across samples has nothing to show; this is
        # what keeps a blank hemoglobin panel off the sheet.
        and float(sample_table[m].std(ddof=0) or 0.0) > 0
    ]
    if not len(sample_table) or not usable:
        ax.set_axis_off()
        return

    palette = _condition_colors(sample_table, case_label=case_label)
    table = _order_by_donor_then_condition(sample_table, palette)
    strip = (
        [palette.get(str(c), REMOVED_GREY) for c in table["condition"]]
        if palette and "condition" in table.columns
        else None
    )

    _plot_metric_matrix(
        ax,
        table,
        metrics=usable,
        row_labels=[str(v) for v in table["sample"]],
        strip_colors=strip,
    )
    strip_note = " · left strip = condition" if strip else ""
    _title(
        ax,
        "Per-sample metric profile",
        f"median per sample, robustly scaled within metric{strip_note}",
    )


def _plot_metric_matrix(
    ax: Axes,
    table: pd.DataFrame,
    *,
    metrics: Sequence[str],
    row_labels: Sequence[str],
    strip_colors: Sequence[str] | None = None,
    label_weights: Sequence[str] | None = None,
    scale_mask: Sequence[bool] | None = None,
) -> None:
    """Draw rows against metrics as a robust z-scored heatmap.

    Scaling is median/IQR per metric so a single extreme row cannot wash the map
    out, which is the whole reason this replaces a wall of per-metric boxplots.
    Callers own the title, because what a row means differs between them.

    ``scale_mask`` selects the rows the scale is computed FROM while every row is
    still drawn: a subtotal row is an aggregate of the rows around it, so letting
    it into the median would flatten the contrast the panel exists to show.
    """

    # Robust z-score: (x - median) / (IQR/1.349) matches a standard deviation for
    # normal data but ignores the outliers this panel exists to reveal.
    reference = np.ones(len(table), dtype=bool) if scale_mask is None else np.asarray(scale_mask)
    matrix = np.empty((len(table), len(metrics)), dtype=float)
    for index, metric in enumerate(metrics):
        values = table[metric].to_numpy(dtype=float)
        basis = values[reference]
        if not np.isfinite(basis).any():
            basis = values
        median = np.nanmedian(basis)
        iqr = np.nanpercentile(basis, 75) - np.nanpercentile(basis, 25)
        scale = (iqr / 1.349) if iqr > 0 else (np.nanstd(basis) or 1.0)
        matrix[:, index] = (values - median) / scale

    limit = float(np.nanpercentile(np.abs(matrix), 98)) or 1.0
    # A missing value is transparent by default, which leaves a white hole that
    # reads as a rendering fault. Filled grey it reads as what it is: a row with
    # nothing to summarise, e.g. a cell type QC removed entirely, which has no
    # retained cells to take a median over.
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad=GRID_GREY)
    image = ax.imshow(
        matrix, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest"
    )

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([METRIC_SHORT.get(m, m) for m in metrics], fontsize=7.5)
    ax.set_yticks(np.arange(len(table)))
    texts = ax.set_yticklabels(row_labels, fontsize=7)
    if label_weights is not None:
        for text, weight in zip(texts, label_weights, strict=True):
            text.set_fontweight(weight)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Hairline separators instead of a box: the grid reads as cells, not as a plot.
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)

    # Identity strip: colour carried alongside the row label, tight against the
    # matrix so it reads as part of the row rather than a stray bar.
    if strip_colors is not None:
        for position, color in enumerate(strip_colors):
            ax.add_patch(
                plt.Rectangle(
                    (-0.66, position - 0.5),
                    0.14,
                    1.0,
                    color=color,
                    clip_on=False,
                    linewidth=0,
                )
            )
        ax.set_xlim(-0.68, len(metrics) - 0.5)

    bar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02, aspect=14)
    bar.set_label("Robust z\nvs cohort", fontsize=7)
    bar.ax.tick_params(labelsize=6.5, length=2)
    bar.outline.set_visible(False)


# Below this many input cells a percentage is a ratio of a handful of barcodes.
# The bar is still drawn — the population exists — but faded, so "80% removed"
# from five cells cannot be read as a finding.
SMALL_CELL_TYPE = 20


def _cell_type_axis(
    ax: Axes,
    display: pd.DataFrame,
    *,
    labels: bool = True,
    fontsize: float = 7.2,
) -> np.ndarray:
    """Lay out the shared row axis of the by-cell-type panels.

    Every panel of the figure draws the same rows in the same order, so the row
    identity is set in one place: subtotal rows carry a tinted band and a bold
    label, members are indented under them, and row zero is at the TOP because
    that is where the largest population belongs.

    Args:
        ax: Target axes.
        display: Output of :func:`cell_type_display_rows`.
        labels: Whether to print the row labels. False on every panel but the
            leftmost, which owns the shared label column.
        fontsize: Row label size.

    Returns:
        The y positions, in row order.
    """

    positions = np.arange(len(display), dtype=float)
    ax.set_yticks(positions)
    if labels:
        text = [
            str(row.label) if row.is_group else f"   {row.label}" for row in display.itertuples()
        ]
        for label, is_group in zip(
            ax.set_yticklabels(text, fontsize=fontsize), display["is_group"], strict=True
        ):
            label.set_fontweight("semibold" if is_group else "normal")
    else:
        ax.set_yticklabels([])
    # Explicit inverted limits rather than invert_yaxis(): the latter compounds
    # when a caller shares an axis, silently flipping the second panel back.
    ax.set_ylim(len(display) - 0.5, -0.5)
    for position, is_group in zip(positions, display["is_group"], strict=True):
        if is_group:
            ax.axhspan(position - 0.5, position + 0.5, color=GROUP_BAND, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return positions


def _cell_type_bar_colors(
    display: pd.DataFrame, *, group: str, member: str, small: str
) -> list[str]:
    """Colour each row by its role, then fade the rows too small to read as rates."""
    return [
        group if row.is_group else (small if row.cells_in < SMALL_CELL_TYPE else member)
        for row in display.itertuples()
    ]


def plot_cell_type_composition(
    ax: Axes,
    display: pd.DataFrame,
    *,
    labels: bool = True,
    log: bool = True,
) -> None:
    """
    Draw how many cells each cell type contributed, and how many survived.

    Populations in an annotated object span orders of magnitude — 1,788 LECs
    against a 3-cell contaminant — so the count axis is logarithmic by default. A
    linear axis here is not a neutral choice: it renders every minority population
    as a hairline, and the minority populations are exactly where a QC filter does
    its most uneven damage.

    Args:
        ax: Target axes.
        display: Output of :func:`cell_type_display_rows`.
        labels: Whether this panel prints the shared row labels.
        log: Whether to use a logarithmic count axis.
    """

    if not len(display):
        ax.set_axis_off()
        return

    positions = _cell_type_axis(ax, display, labels=labels)
    counts = display["cells_in"].to_numpy(dtype=float)
    kept = display["cells_kept"].to_numpy(dtype=float)

    ax.barh(positions, counts, color=RULE_FILL, height=0.74, linewidth=0, zorder=1)
    ax.barh(positions, kept, color=KEPT_GREEN, height=0.74, linewidth=0, zorder=2)

    top = float(np.nanmax(counts)) if len(counts) else 1.0
    if log and (counts > 0).all():
        ax.set_xscale("log")
        # Room on the right for the count label, in log space.
        ax.set_xlim(0.7, top * 3.2)
    else:
        ax.set_xlim(0, top * 1.32 if top else 1.0)

    on_log = ax.get_xscale() == "log"
    halo = [path_effects.withStroke(linewidth=2.2, foreground="white")]
    for position, row in zip(positions, display.itertuples(), strict=True):
        # A gap the label can sit in: multiplicative on a log axis, additive on a
        # linear one. Using one rule for both puts the small rows' labels either
        # inside their own bar or off the axis.
        end = max(float(row.cells_in), 0.7)
        ax.text(
            end * 1.28 if on_log else end + top * 0.02,
            position,
            f"{row.cells_in:,}",
            va="center",
            ha="left",
            fontsize=6.8,
            fontweight="semibold" if row.is_group else "normal",
            color=TEXT if row.is_group else "#6B7280",
            zorder=5,
            path_effects=halo,
        )

    ax.set_xlabel("Cells entering QC" + (" (log)" if on_log else ""))
    _xgrid(ax)
    _title(ax, "Population size", "green = retained, grey = removed")
    ax.legend(
        handles=[
            Patch(facecolor=KEPT_GREEN, label="retained"),
            Patch(facecolor=RULE_FILL, label="removed"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=7.5,
        handlelength=1.1,
        handleheight=0.9,
    )


def plot_cell_type_attrition(
    ax: Axes,
    display: pd.DataFrame,
    *,
    labels: bool = True,
    cohort_pct: float | None = None,
) -> None:
    """
    Draw percent removed per cell type, subtotal bars over their subtypes.

    This is the panel that says whether QC changed the composition of the dataset.
    Uniform bars mean the filter took a flat tax and downstream proportions are
    unaffected; a bar at four times the cohort line means that population entered
    the analysis depleted, and every abundance comparison inherits it.

    Args:
        ax: Target axes.
        display: Output of :func:`cell_type_display_rows`.
        labels: Whether this panel prints the shared row labels.
        cohort_pct: Cohort-wide percent removed for the reference line. Pooled
            from ``display`` when omitted.
    """

    if not len(display) or "pct_removed" not in display.columns:
        ax.set_axis_off()
        return

    positions = _cell_type_axis(ax, display, labels=labels)
    values = display["pct_removed"].fillna(0.0).to_numpy(dtype=float)
    colors = _cell_type_bar_colors(
        display, group=REMOVED_GREY_DARK, member=REMOVED_GREY, small=RULE_FILL
    )
    ax.barh(positions, values, color=colors, height=0.74, linewidth=0, zorder=2)

    if cohort_pct is None:
        members = display.loc[~display["is_group"]] if "is_group" in display else display
        total_in = float(members["cells_in"].sum())
        cohort_pct = 100.0 * float(members["cells_removed"].sum()) / total_in if total_in else 0.0

    limit = float(np.nanmax(values)) if len(values) else 0.0
    ax.set_xlim(0, max(limit * 1.34, 1.0))
    ax.axvline(cohort_pct, color=TEXT, linestyle=(0, (3, 2)), linewidth=0.9, zorder=4)

    halo = [path_effects.withStroke(linewidth=2.2, foreground="white")]
    # Labelled at the FOOT of the line: the top row is the largest group's
    # subtotal bar, whose own count label sits exactly where a top-anchored
    # annotation lands, and the two overprint each other.
    ax.annotate(
        f"cohort {cohort_pct:.1f}%",
        xy=(cohort_pct, len(display) - 0.5),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=7,
        color=TEXT,
        va="bottom",
        ha="left",
        zorder=5,
        path_effects=halo,
    )
    for position, row in zip(positions, display.itertuples(), strict=True):
        pct = 0.0 if pd.isna(row.pct_removed) else float(row.pct_removed)
        ax.text(
            pct + max(limit, 1.0) * 0.02,
            position,
            f"{row.cells_removed:,}/{row.cells_in:,}",
            va="center",
            ha="left",
            fontsize=6.8,
            fontweight="semibold" if row.is_group else "normal",
            color=TEXT if row.is_group else "#6B7280",
            zorder=5,
            path_effects=halo,
        )

    ax.set_xlabel("Cells removed (%)")
    _xgrid(ax)
    faded = (
        " · faded = " + f"<{SMALL_CELL_TYPE} cells in"
        if (display["cells_in"] < SMALL_CELL_TYPE).any()
        else ""
    )
    _title(ax, "Attrition by cell type", f"label = removed / input{faded}")


def plot_cell_type_matrix(
    ax: Axes,
    display: pd.DataFrame,
    *,
    metrics: Sequence[str] = MATRIX_METRICS,
    labels: bool = True,
) -> None:
    """
    Draw the retained cells' metric profile per cell type.

    Attrition says how much QC took; this says what it left behind. A population
    that survives the filter with half the cohort's UMI depth is still in the
    dataset, and every clustering and marker test downstream will feel it.

    Args:
        ax: Target axes.
        display: Output of :func:`cell_type_display_rows`.
        metrics: Metric columns to include, in display order.
        labels: Whether this panel prints the shared row labels.
    """

    usable = [
        m
        for m in metrics
        if m in display.columns
        and display[m].notna().any()
        and float(display[m].std(ddof=0) or 0.0) > 0
    ]
    if not len(display) or not usable:
        ax.set_axis_off()
        return

    members = ~display["is_group"].to_numpy(dtype=bool)
    _plot_metric_matrix(
        ax,
        display,
        metrics=usable,
        row_labels=[
            str(row.label) if row.is_group else f"   {row.label}" for row in display.itertuples()
        ]
        if labels
        else [""] * len(display),
        label_weights=["semibold" if is_group else "normal" for is_group in display["is_group"]]
        if labels
        else None,
        # Scale on the granular rows only: the subtotals are aggregates of them.
        scale_mask=members if members.sum() >= 3 else None,
    )
    subtitle = "median per cell type, scaled within metric"
    if display[usable].isna().all(axis=1).any():
        # Say it, rather than leaving the reader to guess why a row is blank.
        subtitle += " · grey = no cells retained"
    _title(ax, "Metric profile of retained cells", subtitle)


# ---------------------------------------------------------------------------
# Figure writers
# ---------------------------------------------------------------------------


def write_qc_overview_figure(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    thresholds: pd.DataFrame | None = None,
    case_label: str | None = None,
    stem: str = "qc_overview",
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
    title: str | None = None,
) -> list[Path]:
    """
    Write the composite QC overview: the one figure that tells the whole story.

    Panels are added only when the data supports them, so the sheet has no
    "No UMAP embedding" placeholders — QC runs before any embedding exists, and
    reserving space for one wasted 40% of the previous supplementary sheet.

    Args:
        frame: Output of :func:`assemble_qc_frame`.
        output_dir: Directory to write into.
        thresholds: Optional threshold table.
        case_label: Condition to treat as the case arm.
        stem: Output filename stem.
        formats: Formats to write.
        dpi: Raster resolution.
        title: Optional figure suptitle.

    Returns:
        Written paths.
    """

    set_publication_style(dpi=dpi)
    sample_table = summarize_by_sample(frame)
    rule_table = summarize_rules(frame, thresholds)

    has_samples = len(sample_table) > 1
    has_pairs = (
        has_samples
        and {"donor", "condition"}.issubset(sample_table.columns)
        and sample_table["condition"].nunique() == 2
        and sample_table.groupby("donor")["condition"].nunique().eq(2).any()
    )

    # Every panel in the right column is per-sample. Without a sample column that
    # column would hold two axes with nothing in it, which is both the placeholder
    # this module exists to avoid and enough to collapse the layout solve, so the
    # sheet drops to a single column instead.
    n_rows = 2 + int(has_samples)
    n_cols = 2 if has_samples else 1
    # Floor the height: the funnel is a deliberately short row, and below about
    # seven inches of sheet its share of row 0 is smaller than its own title and
    # axis label, which collapses the layout solve for the whole figure.
    height = max(
        7.0,
        5.4 + (4.2 if has_pairs else 0.0) + (0.22 * len(sample_table) if has_samples else 0.0),
    )
    fig = plt.figure(
        figsize=(13.0 if has_samples else 7.4, min(height, 22.0)), layout="constrained"
    )
    # Generous vertical padding: the subtitles are annotations the layout engine
    # cannot measure, so rows need slack the engine would not otherwise leave.
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.14, wspace=0.05, hspace=0.10)
    outer = fig.add_gridspec(
        n_rows,
        n_cols,
        # The left column carries long rule descriptions as tick labels, which eat
        # plot width; without the extra share its bars end up half the size of the
        # right column's.
        width_ratios=[1.18, 1.0] if n_cols == 2 else None,
        height_ratios=[1.0, 1.35] + ([1.05] if has_samples else []),
    )

    # Row 0, left: cohort funnel above rule attribution. No explicit hspace here —
    # under constrained layout hspace is a fraction of AXES height, so 0.55 on a
    # deliberately short funnel row demands more gap than the row has and
    # collapses the whole solve. The engine's figure-level hspace is correct.
    # Letters are consumed as panels are drawn, not assigned per slot. A sheet that
    # omits the per-sample panels must still read A, B, C — not A, B, D.
    letters = iter("ABCDEFGH")

    left = outer[0, 0].subgridspec(2, 1, height_ratios=[0.40, 1.0])
    ax_funnel = fig.add_subplot(left[0])
    plot_cohort_funnel(ax_funnel, frame)
    _panel_label(ax_funnel, next(letters), dy=22.0)

    ax_rules = fig.add_subplot(left[1])
    plot_rule_attribution(ax_rules, rule_table, n_cells=len(frame))
    _panel_label(ax_rules, next(letters), dx=-150.0, dy=16.0)

    # Row 0, right: paired donor comparison of the metric driving attrition.
    if has_samples:
        ax_pair = fig.add_subplot(outer[0, 1])
        if has_pairs and "pct_counts_mito" in sample_table.columns:
            plot_paired_dumbbell(
                ax_pair,
                sample_table,
                "pct_counts_mito",
                case_label=case_label,
                thresholds=thresholds,
                title="Mitochondrial % by donor",
                subtitle=None,
            )
        else:
            plot_sample_attrition(ax_pair, sample_table, case_label=case_label)
        if ax_pair.has_data():
            _panel_label(ax_pair, next(letters), dx=-40.0, dy=16.0)

    # Row 1: the joint scatter, plus per-sample attrition. No marginals here — on
    # a shared row they would either shrink the scatter or overhang panel C.
    ax_joint = fig.add_subplot(outer[1, 0])
    plot_joint_scatter(ax_joint, frame, thresholds=thresholds)
    _panel_label(ax_joint, next(letters), dx=-46.0, dy=16.0)

    if has_samples:
        ax_attrition = fig.add_subplot(outer[1, 1])
        plot_sample_attrition(ax_attrition, sample_table, case_label=case_label)
        _panel_label(ax_attrition, next(letters), dx=-64.0, dy=16.0)

    # Row 2: the per-sample metric matrix.
    if has_samples:
        ax_matrix = fig.add_subplot(outer[2, :])
        plot_sample_matrix(ax_matrix, sample_table, case_label=case_label)
        if ax_matrix.has_data():
            _panel_label(ax_matrix, next(letters), dx=-64.0, dy=16.0)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color=TEXT, ha="left", x=0.008)

    return save_figure(fig, output_dir, stem, formats=formats, dpi=dpi)


def write_qc_cell_type_figure(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    median_population: str = "retained",
    metrics: Sequence[str] = MATRIX_METRICS,
    stem: str = "qc_cell_type",
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
) -> list[Path]:
    """
    Write the by-cell-type QC figure: size, attrition and profile on shared rows.

    The companion to the by-cell-type table, and deliberately laid out like one —
    three graphic columns against one column of row labels, subtotal rows banded
    with their subtypes indented beneath. Same rows, same order, same numbers as
    the table, because a figure and a table that disagree on row order get read as
    disagreeing on the data.

    Args:
        frame: Output of :func:`assemble_qc_frame`, carrying cell-type labels.
        output_dir: Directory to write into.
        median_population: Which cells the metric medians describe.
        metrics: Metrics for the profile panel.
        stem: Output filename stem.
        formats: Formats to write.
        dpi: Raster resolution.

    Returns:
        Written paths, empty when the frame carries no cell-type labels or only
        one row — a single bar is a number, and the table already prints it.
    """

    rows, group_totals = summarize_by_cell_type(frame, median_population=median_population)
    if not len(rows):
        return []
    display = cell_type_display_rows(rows, group_totals)
    if len(display) < 2:
        return []

    set_publication_style(dpi=dpi)
    fig = plt.figure(figsize=(12.6, min(2.0 + 0.27 * len(display), 22.0)), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.10, wspace=0.03)
    # The left panel carries the shared row labels, so it needs the width they eat
    # on top of the width its own bars need.
    grid = fig.add_gridspec(1, 3, width_ratios=[1.34, 0.95, 1.0])

    # Letters at -18pt, not the -8/-10 this figure shipped with: titles are
    # left-aligned to the axes edge, so a 12pt bold glyph starting 8pt to the left
    # of it ends ON it — "APopulation size". The letters sit above the top row, so
    # the width they borrow from the row-label gutter costs nothing.
    ax_size = fig.add_subplot(grid[0])
    plot_cell_type_composition(ax_size, display)
    _panel_label(ax_size, "A", dx=-18.0, dy=16.0)

    ax_attrition = fig.add_subplot(grid[1])
    plot_cell_type_attrition(ax_attrition, display, labels=False)
    _panel_label(ax_attrition, "B", dx=-18.0, dy=16.0)

    ax_matrix = fig.add_subplot(grid[2])
    plot_cell_type_matrix(ax_matrix, display, metrics=metrics, labels=False)
    if ax_matrix.has_data():
        _panel_label(ax_matrix, "C", dx=-18.0, dy=16.0)

    return save_figure(fig, output_dir, stem, formats=formats, dpi=dpi)


def write_qc_panels(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    thresholds: pd.DataFrame | None = None,
    case_label: str | None = None,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
    title: str | None = None,
    mixture_models: pd.DataFrame | None = None,
    mixture_ceiling: float | None = None,
    mixture_posterior_cutoff: float = 0.75,
) -> list[Path]:
    """
    Write the full advanced QC panel set: composite overview plus standalones.

    Each standalone is the same panel as on the overview at a size that stands
    alone in a supplement, so a manuscript can cite one without cropping the
    sheet.

    The mixture panel is the exception: it has no place on the overview because it
    exists only for the runs that fit a model, and reserving a slot for it would
    put back the placeholder the overview was built to avoid. It is written as a
    standalone whenever the caller supplies a fitted model.

    Args:
        frame: Output of :func:`assemble_qc_frame`.
        output_dir: Directory to write into.
        thresholds: Optional threshold table.
        case_label: Condition to treat as the case arm.
        formats: Formats to write.
        dpi: Raster resolution.
        title: Optional overview suptitle.
        mixture_models: Fitted mitochondrial-mixture table, one row per group, as
            produced by the mixture result's own ``to_dataframe``. None when the
            run used fixed or MAD thresholds, in which case no mixture figure is
            written and there is nothing to explain.
        mixture_ceiling: The mitochondrial bound the model was projected onto,
            when a single one applies to the whole object.
        mixture_posterior_cutoff: Posterior above which a cell was called
            compromised, used to place the drawn boundary.

    Returns:
        Every written path.
    """

    output_dir = Path(output_dir)
    paths: list[Path] = list(
        write_qc_overview_figure(
            frame,
            output_dir,
            thresholds=thresholds,
            case_label=case_label,
            formats=formats,
            dpi=dpi,
            title=title,
        )
    )

    set_publication_style(dpi=dpi)
    sample_table = summarize_by_sample(frame)
    rule_table = summarize_rules(frame, thresholds)

    # Attrition: funnel above rule attribution.
    fig = plt.figure(figsize=(7.2, 1.4 + 0.42 * max(len(rule_table), 3)), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, hspace=0.14)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.40, 1.0])
    ax_top = fig.add_subplot(grid[0])
    plot_cohort_funnel(ax_top, frame)
    _panel_label(ax_top, "A", dx=-34.0, dy=22.0)
    ax_bottom = fig.add_subplot(grid[1])
    plot_rule_attribution(ax_bottom, rule_table, n_cells=len(frame))
    _panel_label(ax_bottom, "B", dx=-150.0, dy=16.0)
    paths += save_figure(fig, output_dir, "qc_attrition", formats=formats, dpi=dpi)

    # Joint density on its own, with the marginals. Explicit axes placement, not
    # constrained layout: the marginals are insets that overhang the scatter, and
    # the layout engine would try to reclaim the space they need.
    fig = plt.figure(figsize=(6.6, 6.4))
    ax = fig.add_axes((0.115, 0.085, 0.70, 0.70))
    plot_joint_scatter(ax, frame, thresholds=thresholds, show_title=False)
    plot_joint_marginals(ax, frame)
    fig.text(
        0.115,
        0.985,
        "Library complexity",
        fontsize=9.5,
        fontweight="bold",
        color=TEXT,
        va="top",
        ha="left",
    )
    fig.text(
        0.115,
        0.955,
        "removed cells stay on the plot, in grey",
        fontsize=7.5,
        color="#6B7280",
        va="top",
        ha="left",
    )
    paths += save_figure(fig, output_dir, "qc_joint_density", formats=formats, dpi=dpi)

    # The mixture model that produced an adaptive mitochondrial cut, when there was
    # one. Square, and the same size as the joint density it explains: the two are
    # read side by side, one showing the cut and the other showing why.
    if mixture_models is not None and len(mixture_models):
        fig, ax = plt.subplots(figsize=(6.6, 5.6), layout="constrained")
        try:
            plot_mito_mixture(
                ax,
                frame,
                mixture_models,
                posterior_cutoff=mixture_posterior_cutoff,
                ceiling=mixture_ceiling,
            )
        except QCPanelError:
            # The mixture needs the two metrics it regressed. A run that recorded a
            # model but not the metrics is a broken artifact set, not a reason to
            # lose the rest of the sheet.
            plt.close(fig)
        else:
            paths += save_figure(fig, output_dir, "qc_mito_mixture", formats=formats, dpi=dpi)

    if len(sample_table) > 1:
        # Per-sample attrition.
        fig, ax = plt.subplots(figsize=(6.6, 1.4 + 0.28 * len(sample_table)), layout="constrained")
        plot_sample_attrition(ax, sample_table, case_label=case_label)
        paths += save_figure(fig, output_dir, "qc_sample_attrition", formats=formats, dpi=dpi)

        # Per-sample metric matrix.
        fig, ax = plt.subplots(figsize=(6.2, 1.5 + 0.24 * len(sample_table)), layout="constrained")
        plot_sample_matrix(ax, sample_table, case_label=case_label)
        if ax.has_data():
            paths += save_figure(fig, output_dir, "qc_sample_matrix", formats=formats, dpi=dpi)
        else:
            plt.close(fig)

        # Paired dumbbells, one per metric that has a within-donor contrast.
        paired = {"donor", "condition"}.issubset(sample_table.columns) and sample_table[
            "condition"
        ].nunique() == 2
        if paired:
            # One spec per panel: the column, and the wording when the metric menu
            # cannot supply it.
            candidates: list[tuple[str, str | None, str | None]] = [
                (m, None, None)
                for m in ("pct_counts_mito", "total_counts", "n_genes_by_counts", "doublet_score")
                if m in sample_table.columns and sample_table[m].notna().any()
            ]

            # Attrition last, and labelled here rather than through METRIC_LABELS.
            # This is the reviewer's question -- did QC take more from one arm,
            # donor by donor -- and it belongs beside the metrics that drove it, so
            # that "mito is higher in disease in seven of nine donors" and "removal
            # is higher in disease in seven of nine donors" are read together. It
            # stays out of the metric menu because it is not measured from the cell:
            # it is what the filter did, and the sample matrix and the metric
            # histograms would both be wrong to offer it as a cell property.
            if "pct_removed" in sample_table.columns and sample_table["pct_removed"].notna().any():
                candidates.append(
                    ("pct_removed", "Cells removed by QC (%)", "QC attrition by donor")
                )

            if candidates:
                n_donors = sample_table["donor"].nunique()
                fig, axes = plt.subplots(
                    1,
                    len(candidates),
                    figsize=(3.1 * len(candidates), 1.6 + 0.34 * n_donors),
                    layout="constrained",
                )
                axes = np.atleast_1d(axes)
                # One shared legend below the row rather than one per panel: the
                # key is identical in every panel, and the in-axes copy sat on top
                # of the paired-test p-value in the only empty band each panel has.
                shared: tuple[list, list[str]] | None = None
                for letter, axis, (metric, xlabel, panel_title) in zip(
                    "ABCDEFGH", axes, candidates, strict=False
                ):
                    try:
                        plot_paired_dumbbell(
                            axis,
                            sample_table,
                            metric,
                            case_label=case_label,
                            thresholds=thresholds,
                            xlabel=xlabel,
                            title=panel_title,
                            show_legend=False,
                        )
                    except QCPanelError:
                        axis.set_axis_off()
                        continue
                    _panel_label(axis, letter, dx=-32.0, dy=18.0)
                    # Keep harvesting until a panel offers the fullest key: only
                    # metrics whose bound lands on-axis contribute a "QC bound"
                    # entry, and that is rarely the first panel.
                    handles, labels = axis.get_legend_handles_labels()
                    if len(handles) >= 2 and (shared is None or len(handles) > len(shared[1])):
                        # Conditions are drawn case-first for overplotting reasons
                        # and shown control-first, like every other legend on the
                        # sheet; anything drawn after them keeps its own order.
                        order = [1, 0, *range(2, len(handles))]
                        shared = ([handles[i] for i in order], [labels[i] for i in order])
                if shared is not None:
                    fig.legend(
                        *shared,
                        loc="outside lower center",
                        ncol=len(shared[1]),
                        frameon=False,
                        fontsize=8,
                        handletextpad=0.2,
                        columnspacing=1.6,
                    )
                paths += save_figure(fig, output_dir, "qc_donor_paired", formats=formats, dpi=dpi)

    # The by-cell-type figure, when the object was annotated before QC ran. It is
    # its own figure rather than a row of the overview: an unannotated cohort has
    # no rows to draw, and reserving space for it would put the placeholder back.
    paths += write_qc_cell_type_figure(frame, output_dir, formats=formats, dpi=dpi)

    return paths


__all__ = [
    "COARSE_CELL_TYPE_KEYS",
    "GRANULAR_CELL_TYPE_KEYS",
    "GROUP_BAND",
    "MATRIX_METRICS",
    "METRIC_LABELS",
    "METRIC_SHORT",
    "QCPanelError",
    "SMALL_CELL_TYPE",
    "UNLABELLED",
    "assemble_qc_frame",
    "cell_type_display_rows",
    "label_series",
    "order_cell_types",
    "order_samples",
    "plot_cell_type_attrition",
    "plot_cell_type_composition",
    "plot_cell_type_matrix",
    "plot_cohort_funnel",
    "plot_joint_marginals",
    "plot_joint_scatter",
    "plot_mito_mixture",
    "plot_paired_dumbbell",
    "plot_rule_attribution",
    "plot_sample_attrition",
    "plot_sample_matrix",
    "resolve_cell_type_keys",
    "summarize_by_cell_type",
    "summarize_by_sample",
    "summarize_rules",
    "write_qc_cell_type_figure",
    "write_qc_overview_figure",
    "write_qc_panels",
]
