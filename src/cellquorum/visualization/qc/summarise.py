"""One QC summariser, shared by every renderer that reports per-group attrition.

There were two, near-identical: ``panels._summarize_qc_groups`` and
``html_report.summarize_qc_rows``. Same groupby, same ``"mixed"`` handling for carried label
columns, same ``cells_in``/``cells_kept``/``cells_removed``/``pct_removed``, same
``median_population`` semantics, same empty-frame column trick. They diverged only in how the
caller named the carried columns and which metric list they walked — which is a caller's business,
not the summariser's.

Two copies of an aggregation is worse than it looks. The numbers in the HTML report, the typeset
Table 1 and the figure panels are meant to *be the same numbers*, and a reader who finds them
disagreeing has no way to tell which is right. Keeping them equal by hand across two
implementations is a promise nobody can keep.

So the metric columns are **discovered from the frame** rather than read from a module-level list.
That is what lets one function serve both conventions: ``panels`` carries raw metric names
(``total_counts``), ``html_report`` renames at join time (``median_umi``), and the summariser does
not need to know which — it takes medians of the numeric columns the caller says are metrics.

``order_samples`` lives here too, for a smaller reason: ``html_report`` reached for it with a
lazy import inside a function body, which reads as a circular-import dodge and is not one. There
is no cycle — this module has no dependency on either renderer.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

#: Bookkeeping columns a summary always produces, so a caller can tell them from metric medians.
COUNT_COLUMNS: tuple[str, ...] = ("cells_in", "cells_kept", "cells_removed", "pct_removed")

#: Label columns that are never treated as metrics, whatever their dtype.
_NON_METRIC = frozenset(
    {"keep", "sample", "donor", "condition", "group", "cell_type", *COUNT_COLUMNS}
)


class QCSummaryError(ValueError):
    """Report a QC summary asked for on a frame that cannot support it."""


def metric_columns(frame: pd.DataFrame) -> list[str]:
    """The numeric per-cell columns in ``frame`` that a summary should take medians of.

    Discovered rather than declared, which is what lets one summariser serve callers that name
    their metric columns differently. Label and bookkeeping columns are excluded by name; anything
    else numeric is a metric.

    Args:
        frame: A per-cell QC frame.

    Returns:
        Metric column names, in frame order so the output column order is stable.
    """
    return [
        column
        for column in frame.columns
        if column not in _NON_METRIC
        # Booleans are numeric to pandas, and the per-cell frame carries a boolean column per
        # QC rule (`rule:fixed_min_genes_per_cell`, ...). Taking their median would put a
        # meaningless 0.0/1.0 "metric" in every summary table.
        and not pd.api.types.is_bool_dtype(frame[column])
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def summarize_qc_pool(
    frame: pd.DataFrame,
    *,
    median_population: str = "all",
    metrics: Sequence[str] | None = None,
) -> dict[str, object]:
    """Aggregate a per-cell QC frame into one pooled row.

    Medians pool the cells rather than averaging per-group medians, which would weight a 21-cell
    group like a 490-cell one.

    Args:
        frame: A per-cell QC frame carrying a boolean ``keep`` column.
        median_population: ``"all"`` summarises every input cell, which explains why cells were
            removed; ``"retained"`` summarises survivors, which describes the dataset the analysis
            runs on. Attrition counts are always pre-filter either way.
        metrics: Metric columns to take medians of. ``None`` discovers them from the frame.

    Returns:
        Attrition counts and metric medians, with no label column.

    Raises:
        QCSummaryError: If ``median_population`` is not a known value, or ``keep`` is absent.
    """
    if median_population not in {"all", "retained"}:
        raise QCSummaryError(
            f"median_population must be 'all' or 'retained', got {median_population!r}."
        )
    if "keep" not in frame.columns:
        raise QCSummaryError(
            "A QC summary needs a boolean 'keep' column indexed by every INPUT cell. Summarising "
            "the surviving object instead reports zero removals by construction."
        )

    n_in = int(len(frame))
    n_keep = int(frame["keep"].sum()) if n_in else 0
    row: dict[str, object] = {
        "cells_in": n_in,
        "cells_kept": n_keep,
        "cells_removed": n_in - n_keep,
        "pct_removed": 100.0 * (n_in - n_keep) / n_in if n_in else float("nan"),
    }

    described = frame.loc[frame["keep"]] if median_population == "retained" else frame
    for metric in metric_columns(frame) if metrics is None else metrics:
        if metric not in frame.columns:
            continue
        # NaN when a group lost every cell, which is the honest value: it contributed nothing to
        # the analysed dataset, and zero would read as a measurement.
        row[metric] = float(described[metric].median()) if len(described) else float("nan")
    return row


def summarize_qc_rows(
    frame: pd.DataFrame,
    *,
    label: str,
    name: str,
    carry: Sequence[str] | dict[str, str] = (),
    median_population: str = "all",
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate a per-cell QC frame into one attrition row per value of ``label``.

    Args:
        frame: A per-cell QC frame carrying ``keep`` and ``label``.
        label: Column to group by.
        name: Name the grouping column takes in the result.
        carry: Other label columns to carry through, reported as ``"mixed"`` when a group spans
            several values. A sequence keeps each column's own name; a mapping renames
            ``{source: target}``, which is what the figure panels want.
        median_population: ``"all"`` or ``"retained"`` — see :func:`summarize_qc_pool`.
        metrics: Metric columns to summarise. ``None`` discovers them from the frame.

    Returns:
        One row per group value, sorted by group value. Empty input still carries the label and
        carried columns, so a caller can build a table header from a cohort that has no groups.
    """
    carried: dict[str, str] = dict(carry) if isinstance(carry, dict) else {c: c for c in carry}
    resolved = metric_columns(frame) if metrics is None else list(metrics)

    rows: list[dict[str, object]] = []
    for value, chunk in frame.groupby(label, observed=True, dropna=False, sort=True):
        row: dict[str, object] = {name: str(value)}
        for source, target in carried.items():
            if source in chunk.columns:
                unique = chunk[source].dropna().unique()
                row[target] = str(unique[0]) if len(unique) == 1 else "mixed"
        row.update(summarize_qc_pool(chunk, median_population=median_population, metrics=resolved))
        rows.append(row)

    header = [name, *(target for source, target in carried.items() if source in frame.columns)]
    return pd.DataFrame(rows, columns=None if rows else [*header, *COUNT_COLUMNS])


def natural_sort_key(values: pd.Series) -> pd.Series:
    """Natural-order sort key for a label column, so ``P2`` precedes ``P10``.

    Zero-pads each digit run rather than returning a tuple, so the result is a plain string
    column that ``sort_values`` orders identically on every pandas version. A tuple-valued key
    sorts correctly too, but pandas compares tuples elementwise and the ordering of a mixed
    ``(str, int)`` tuple against a ``(str,)`` one is not the same as the padded-string ordering —
    which is how this returned a different row order than the panels it replaced.
    """
    return values.astype(str).str.replace(
        r"(\d+)", lambda match: match.group(1).zfill(6), regex=True
    )


def condition_rank(levels: Sequence[str], case_label: str | None = None) -> dict[str, int]:
    """Rank arms control-first, case-second, anything further last.

    Ranking by the design's control/case assignment rather than by label spelling, because
    alphabetical order puts ``"LE"`` before ``"Normal"`` — so every donor's pair would read
    case-then-control and the eye would have to reverse each one.

    Only two arms are privileged. A third is ranked last rather than folded into either, which
    matches how the two-hue condition palette treats it: the design names one control and one
    case, and a figure that pretends otherwise is inventing structure.

    Args:
        levels: Condition values present in the data.
        case_label: The arm treated as the case. ``None``, or a label not present, falls back to
            natural order — deterministic, and the figure titles say which way it reads.

    Returns:
        ``{level: rank}``, ranks 0, 1, 2.
    """
    present = [str(level) for level in dict.fromkeys(str(v) for v in levels)]
    if case_label and str(case_label) in present:
        controls = [level for level in present if level != str(case_label)]
        ordered = ([controls[0]] if controls else []) + [str(case_label)]
    else:
        ordered = sorted(present, key=lambda level: natural_key_of(level))
    ranks = {level: index for index, level in enumerate(ordered[:2])}
    return {level: ranks.get(level, 2) for level in present}


def natural_key_of(label: object) -> tuple[object, ...]:
    """Single-label natural key, so this module does not re-derive one."""
    from cellquorum.visualization.figstyle import natural_key

    return natural_key(label)


def order_samples(
    table: pd.DataFrame,
    *,
    case_label: str | None = None,
) -> pd.DataFrame:
    """Order a per-sample table donor-major, control arm first, donors naturally.

    The reading order every QC figure and table shares. Adjacent rows are a donor's paired
    samples, which is what makes a within-donor quality imbalance visible; without it a reader is
    comparing across donors and across arms at once.

    Args:
        table: A per-sample table, normally from :func:`summarize_qc_rows`.
        case_label: The condition treated as the case arm, so the control sorts first.

    Returns:
        The table reordered, with the sort keys dropped and the index reset.
    """
    if table.empty:
        return table

    ordered = table.copy()
    keys: list[str] = []

    if "donor" in ordered.columns:
        ordered["_donor_key"] = natural_sort_key(ordered["donor"].astype(str))
        keys.append("_donor_key")

    if "condition" in ordered.columns:
        ranks = condition_rank(ordered["condition"].dropna().astype(str).tolist(), case_label)
        ordered["_condition_rank"] = ordered["condition"].astype(str).map(ranks).fillna(2)
        keys.append("_condition_rank")

    if "sample" in ordered.columns:
        keys.append("sample")

    if keys:
        ordered = ordered.sort_values(keys, kind="stable")
    return ordered.drop(columns=["_donor_key", "_condition_rank"], errors="ignore").reset_index(
        drop=True
    )


__all__ = [
    "COUNT_COLUMNS",
    "QCSummaryError",
    "metric_columns",
    "natural_sort_key",
    "order_samples",
    "summarize_qc_pool",
    "summarize_qc_rows",
]
