# Pipeline step (order=20): qc — annotate helpers for the QC stage.
"""Carrying QC results onto an AnnData, and building the objects the run needs.

Three objects come out of a QC run and they are not the same object: the one that goes
downstream (optionally filtered), the one the figures render from (never filtered, so a
panel can draw what was removed), and the metric tables that stay canonical. Conflating
them is how a "100% pass" barplot ships from a run that dropped 13% of its cells.

Writing to ``obs`` is also where a recomputed metric can silently overwrite an inherited
one, so the annotation helpers compare before they replace and report any disagreement
rather than winning quietly.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.qc._errors import QCStageError
from cellquorum.stages.qc.floors import FLOOR_REASON_COLUMN, FloorResult
from cellquorum.stages.qc.metrics import QCMetricsResult

#: The keep column figure code reads. Kept under its historical name so a panel written
#: against it still works: what changed is what decides it — an absolute floor rather than a
#: threshold verdict — not that a panel needs to know which cells are in the analysis.
KEEP_COLUMN = "cellquorum_qc_keep"


def build_qc_output_adata(*, adata: ad.AnnData, floors: FloorResult) -> ad.AnnData:
    """Annotate with floor outcomes and drop what is below the floor.

    Floors always filter: a barcode with too few genes is not a cell and a gene seen in two cells
    is not measurable, so there is no flag-but-keep mode. Every judgement — damaged, may-fit,
    may-inform — belongs to graded adjudication, which never deletes.

    Args:
        adata: Input AnnData object.
        floors: Masks and reasons from :func:`cellquorum.stages.qc.floors.apply_floors`.

    Returns:
        The object with floor columns written, restricted to what cleared the floor.
    """
    if not isinstance(adata, ad.AnnData):
        raise QCStageError(
            f"build_qc_output_adata expected an AnnData object. Received: {type(adata).__name__}."
        )

    output_adata = adata.copy()
    output_adata.obs[FLOOR_REASON_COLUMN] = (
        floors.reason.reindex(output_adata.obs_names).fillna("").to_numpy()
    )
    cells = floors.cell_keep.reindex(output_adata.obs_names).fillna(True).to_numpy(dtype=bool)
    genes = floors.gene_keep.reindex(output_adata.var_names).fillna(True).to_numpy(dtype=bool)
    if cells.all() and genes.all():
        return output_adata
    return output_adata[cells, genes].copy()


def build_qc_figure_adata(
    *,
    adata: ad.AnnData,
    output_adata: ad.AnnData,
    metrics_result: QCMetricsResult,
    floors: FloorResult,
) -> ad.AnnData:
    """The object figures render from: every input cell, carrying floor outcomes.

    Never the filtered object. A panel whose job is "what did QC remove" cannot draw the removed
    cells if it is handed the survivors — which is how a "100% pass" barplot shipped from a run
    that dropped 13% of its cells.

    Args:
        adata: The unfiltered input.
        output_adata: The filtered output, used for its graded columns.
        metrics_result: Computed metrics, indexed by every input cell.
        floors: Floor masks and reasons.

    Returns:
        An unfiltered object with metrics, floor outcomes and graded columns on obs.
    """
    # obs/var only, no expression matrix. Every QC panel reads obs, var and obsm, and QC runs
    # early enough that a second copy of X on a cohort-scale object is pure cost — 11 GB on the
    # validation cohort.
    figure_adata = ad.AnnData(obs=adata.obs.copy(), var=adata.var.copy())
    figure_adata.obs[FLOOR_REASON_COLUMN] = (
        floors.reason.reindex(figure_adata.obs_names).fillna("").to_numpy()
    )
    figure_adata.obs[KEEP_COLUMN] = (
        floors.cell_keep.reindex(figure_adata.obs_names).fillna(True).to_numpy()
    )
    annotate_adata_with_qc_metrics(adata=figure_adata, metrics_result=metrics_result)

    # Everything computed after filtering — graded verdicts, doublet scores, cell-cycle phase —
    # exists only for cells that cleared the floor, so it is REINDEXED rather than assigned. A
    # sub-floor barcode gets NaN, which is the truth: nothing was computed for it. Assigning a
    # value instead would let a panel imply that a barcode nobody scored had scored well.
    for column in output_adata.obs.columns:
        if column not in figure_adata.obs.columns:
            figure_adata.obs[column] = output_adata.obs[column].reindex(figure_adata.obs_names)
    return figure_adata


def annotate_adata_with_qc_metrics(
    *,
    adata: ad.AnnData,
    metrics_result: QCMetricsResult,
) -> list[str]:
    """
    Add calculated QC metric columns to an AnnData object in place.

    QC metrics are calculated as explicit tables first. Plotting and downstream
    inspection, however, expect common cell-level metrics such as
    ``pct_counts_mito`` to be available on ``adata.obs``. This helper aligns the
    metric tables to the possibly filtered QC AnnData and stores non-conflicting
    metric columns on ``obs`` and ``var``. Pre-existing columns are preserved,
    never overwritten.

    Args:
        adata: QC AnnData to annotate.
        metrics_result: Calculated QC metrics.

    Returns:
        Human-readable warnings for any metric columns skipped because they
        already existed on ``obs``/``var``.

    Raises:
        QCStageError: If the QC AnnData axes cannot be aligned to the metric
            tables.
    """

    # Validate input types.
    if not isinstance(adata, ad.AnnData):
        raise QCStageError(
            "annotate_adata_with_qc_metrics expected an AnnData object. "
            f"Received: {type(adata).__name__}."
        )
    if not isinstance(metrics_result, QCMetricsResult):
        raise QCStageError(
            f"metrics_result must be a QCMetricsResult. Received: {type(metrics_result).__name__}."
        )

    # Align and add cell-level metrics.
    cell_metrics = align_metric_table_to_axis(
        axis_names=adata.obs_names,
        metric_table=metrics_result.cell_metrics,
        axis_label="obs",
    )
    obs_conflicts = add_metric_columns_to_axis(axis_frame=adata.obs, metrics=cell_metrics)

    # Align and add gene-level metrics.
    gene_metrics = align_metric_table_to_axis(
        axis_names=adata.var_names,
        metric_table=metrics_result.gene_metrics,
        axis_label="var",
    )
    var_conflicts = add_metric_columns_to_axis(axis_frame=adata.var, metrics=gene_metrics)

    # Report the columns whose inherited values disagreed and were replaced. Worth a
    # warning rather than a note even though QC has corrected them: an input carrying
    # QC metadata that describes a different object is a fact about the input, and on
    # a per-lineage arm it says the slice was carved from a larger object without its
    # derived columns being refreshed. Columns that already agreed are silent.
    warnings: list[str] = []
    if obs_conflicts:
        warnings.append(
            "QC recomputed obs metric columns whose inherited values described a "
            f"different object; replaced: {', '.join(obs_conflicts)}."
        )
    if var_conflicts:
        warnings.append(
            "QC recomputed var metric columns whose inherited values described a "
            "different object (gene-level metrics are aggregates over cells and do "
            f"not survive subsetting); replaced: {', '.join(var_conflicts)}."
        )
    return warnings


def align_metric_table_to_axis(
    *,
    axis_names: pd.Index,
    metric_table: pd.DataFrame,
    axis_label: str,
) -> pd.DataFrame:
    """
    Align a QC metric table to AnnData obs/var names.

    Args:
        axis_names: AnnData axis names.
        metric_table: QC metric table indexed by the original axis names.
        axis_label: Human-readable axis label for errors.

    Returns:
        Metric table aligned to ``axis_names``.

    Raises:
        QCStageError: If axis names are not present in the metric table.
    """

    # Fast path: no filtering/reordering occurred.
    if list(axis_names) == list(metric_table.index):
        return metric_table

    # Reindex supports filtered outputs while preserving the QC AnnData order.
    missing = pd.Index(axis_names).difference(metric_table.index)
    if len(missing) > 0:
        preview = ", ".join(map(str, missing[:5]))
        raise QCStageError(
            f"Cannot annotate QC {axis_label} metrics: {len(missing)} axis name(s) "
            f"are missing from the metric table. First missing: {preview}."
        )

    return metric_table.reindex(axis_names)


def _values_agree(existing: np.ndarray, fresh: np.ndarray) -> bool:
    """Do a pre-existing metric column and the freshly computed one say the same thing?

    Numeric columns compare with a tolerance and treat NaN as equal to NaN, because
    a metric recomputed by the same formula on the same matrix can differ in the last
    bits without differing in meaning.
    """
    if existing.shape != fresh.shape:
        return False
    try:
        return bool(
            np.allclose(
                existing.astype("float64"), fresh.astype("float64"), rtol=1e-9, equal_nan=True
            )
        )
    except (TypeError, ValueError):
        return bool(np.array_equal(existing, fresh))


def _describe_range(values: np.ndarray) -> str:
    """A one-number summary that makes a wrong-scale column obvious. Max, so that an
    inherited whole-atlas gene count reads as 200072 next to the arm's own 2125."""
    try:
        finite = values.astype("float64")
        finite = finite[np.isfinite(finite)]
        return f"max {finite.max():.6g}" if finite.size else "all non-finite"
    except (TypeError, ValueError):
        return f"{len(values)} non-numeric value(s)"


def add_metric_columns_to_axis(
    *,
    axis_frame: pd.DataFrame,
    metrics: pd.DataFrame,
) -> list[str]:
    """
    Add metric-table columns to an AnnData axis frame by row order.

    The freshly computed metric WINS over a pre-existing column of the same name,
    because every metric here is a pure function of the QC matrix of *this* object:
    a value that disagrees is not a competing opinion, it is a description of a
    different object. This used to be flag-not-clobber, and on a per-lineage arm
    carved out of an atlas that was actively wrong — the clean LEC input carries
    ``var['n_cells_by_counts']`` up to 200,072 on 2,125 cells, since gene-level
    metrics are aggregates OVER cells and do not survive subsetting. QC recomputed
    them correctly and then discarded them in favour of the inherited ones, so the
    final object shipped whole-atlas gene metrics (up to 100x too large) for anyone
    reading ``var`` — replotting from the run dir, or the gene-detection and
    counts-vs-genes panels of the since-deleted v1 figure module, which read
    ``n_cells_by_counts`` directly. The panel set reads no ``var`` column, so the
    figures these arms actually shipped were unaffected.

    A column whose existing values already agree with the fresh ones is left alone
    and NOT reported: per-cell metrics such as ``total_counts`` are invariant under
    cell subsetting, so on these inputs they match exactly, and reporting them was
    noise that made the genuinely stale gene-level columns look equally harmless.

    Args:
        axis_frame: AnnData obs or var DataFrame.
        metrics: Aligned metric table.

    Returns:
        Descriptions of the columns whose pre-existing values DISAGREED and were
        replaced, each naming the old and new magnitude, for the caller to surface.
    """

    # Store unprefixed metric names so plotting code and users see standard
    # Scanpy-style QC columns: total_counts, pct_counts_mito, etc.
    replaced: list[str] = []
    for column in metrics.columns:
        fresh = metrics[column].to_numpy()
        if column in axis_frame.columns:
            existing = axis_frame[column].to_numpy()
            if _values_agree(existing, fresh):
                continue
            replaced.append(f"{column} ({_describe_range(existing)} -> {_describe_range(fresh)})")
        axis_frame[column] = fresh

    return replaced
