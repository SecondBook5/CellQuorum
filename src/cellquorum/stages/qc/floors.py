# Pipeline step (order=20): qc — absolute floors, below which nothing can be modelled.
"""Absolute floors: the exclusions that are not judgements and cannot be graded.

This module replaces the fixed-and-MAD threshold path (``thresholds.py`` + ``decisions.py``,
2,090 lines) with the only part of it that graded adjudication cannot express. Everything else
that path did — deciding *which cells are bad* from a number — is what graded severity does
better, per lineage, with concordance across evidence families and an honest account of what it
could not measure.

Three things remained, and none of them is a threshold in the old sense:

**A barcode below a gene floor is not a cell.** Graded severity is a statement about a
population: "unusual for cells like this". An empty droplet has no population, so there is
nothing to be unusual against, and letting it into a lineage lets it anchor a group. The floor
is absolute — never a quantile, never a MAD — because "fewer than N genes detected" is a
statement about the assay's detection limit, not about the cohort.

**A gene detected in almost no cells is not measurable.** Graded QC scores cells, never genes,
so gene filtering had no home in it at all. It is a separate operation on a separate axis and
belongs here rather than being smuggled through a per-cell decision table.

**The mixture model.** ``fit_mito_mixture`` was reached through the threshold machinery, which
made the graded metabolic axis depend on a path it otherwise had no use for. It never needed to
be: the function takes a metric frame and a config and returns a posterior. It is now called
directly.

## Why this is not just the old thresholds renamed

The old path produced a *verdict* — ``cellquorum_qc_keep`` — from configurable numbers on five
metrics, and three places in the codebase read it, two of them figure code. These floors produce
no verdict. They remove barcodes that cannot be analysed and genes that cannot be measured, and
then hand every remaining judgement to the graded model. There is one QC system, not two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc._types import ExpressionMatrix

logger = logging.getLogger(__name__)


FLOOR_REASON_COLUMN = "qc_floor_reason"


class QCFloorError(CellQuorumDataError):
    """Report a malformed floor table."""


@dataclass(frozen=True)
class FloorResult:
    """What the floors removed, and why.

    Args:
        cell_keep: Per-barcode mask of barcodes above the floor.
        gene_keep: Per-gene mask of genes detected in enough cells.
        reason: Per-barcode reason string, empty for barcodes that passed.
        summary: Counts for provenance.
        warnings: Messages that must reach the run report rather than a log line.
    """

    cell_keep: pd.Series
    gene_keep: pd.Series
    reason: pd.Series
    summary: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def n_cells_removed(self) -> int:
        """Barcodes below the floor."""
        return int((~self.cell_keep).sum())

    def n_genes_removed(self) -> int:
        """Genes below the detection floor."""
        return int((~self.gene_keep).sum())

    def cell_table(self) -> pd.DataFrame:
        """Per-barcode outcome, for artifacts and reports."""
        return pd.DataFrame({"keep": self.cell_keep, "floor_reason": self.reason})

    def gene_table(self) -> pd.DataFrame:
        """Per-gene outcome, for artifacts and reports."""
        return pd.DataFrame({"keep": self.gene_keep})

    def to_summary_dict(self) -> dict[str, object]:
        """JSON-friendly summary, including the reasons that actually fired."""
        fired = self.reason[self.reason != ""].value_counts().to_dict()
        return {
            **self.summary,
            "floor_reasons": {str(key): int(value) for key, value in fired.items()},
            "warnings": list(self.warnings),
        }


def apply_floors(
    matrix: ExpressionMatrix,
    obs_names: pd.Index,
    var_names: pd.Index,
    *,
    min_genes_per_cell: int | None = 100,
    min_counts_per_cell: int | None = None,
    min_cells_per_gene: int | None = 3,
) -> FloorResult:
    """Identify barcodes and genes that cannot be analysed at all.

    Args:
        matrix: Raw counts, cells x genes.
        obs_names: Barcode names.
        var_names: Gene names.
        min_genes_per_cell: Genes a barcode must detect. None disables the floor.
        min_counts_per_cell: Counts a barcode must carry. None disables the floor.
        min_cells_per_gene: Cells a gene must be detected in. None disables the floor.

    Returns:
        The masks, per-barcode reasons, and counts.
    """
    from cellquorum.stages.qc.metrics import count_positive_axis, sum_axis

    if matrix.shape != (len(obs_names), len(var_names)):
        raise QCFloorError("Floor matrix shape does not match its cell and gene names.")
    return floors_from_metrics(
        pd.DataFrame(
            {
                "n_genes_by_counts": count_positive_axis(matrix, axis=1),
                "total_counts": sum_axis(matrix, axis=1),
            },
            index=obs_names,
        ),
        pd.DataFrame({"n_cells_by_counts": count_positive_axis(matrix, axis=0)}, index=var_names),
        min_genes_per_cell=min_genes_per_cell,
        min_counts_per_cell=min_counts_per_cell,
        min_cells_per_gene=min_cells_per_gene,
    )


def floors_from_metrics(
    cell_metrics: pd.DataFrame,
    gene_metrics: pd.DataFrame,
    *,
    min_genes_per_cell: int | None = 100,
    min_counts_per_cell: int | None = None,
    min_cells_per_gene: int | None = 3,
) -> FloorResult:
    """Apply floors to freshly computed QC metrics, without rescanning the counts."""
    obs_names, var_names = cell_metrics.index, gene_metrics.index
    genes_per_cell = cell_metrics["n_genes_by_counts"].to_numpy()
    counts_per_cell = cell_metrics["total_counts"].to_numpy()
    cells_per_gene = gene_metrics["n_cells_by_counts"].to_numpy()

    reason = pd.Series("", index=obs_names, dtype=object)
    cell_keep = pd.Series(True, index=obs_names, dtype=bool)

    if min_genes_per_cell is not None:
        below = genes_per_cell < int(min_genes_per_cell)
        reason.loc[below & (reason == "")] = f"fewer_than_{int(min_genes_per_cell)}_genes"
        cell_keep &= ~below
    if min_counts_per_cell is not None:
        below = counts_per_cell < int(min_counts_per_cell)
        reason.loc[below & (reason == "")] = f"fewer_than_{int(min_counts_per_cell)}_counts"
        cell_keep &= ~below

    gene_keep = pd.Series(True, index=var_names, dtype=bool)
    if min_cells_per_gene is not None:
        gene_keep = pd.Series(
            cells_per_gene >= int(min_cells_per_gene), index=var_names, dtype=bool
        )

    summary = {
        "n_cells": int(len(obs_names)),
        "n_cells_below_floor": int((~cell_keep).sum()),
        "n_genes": int(len(var_names)),
        "n_genes_below_floor": int((~gene_keep).sum()),
    }

    warnings: list[str] = []

    if summary["n_cells"] and summary["n_cells_below_floor"] / summary["n_cells"] > 0.5:
        warnings.append(
            f"QC floors removed {summary['n_cells_below_floor']:,} of {summary['n_cells']:,} "
            f"barcodes ({100 * summary['n_cells_below_floor'] / summary['n_cells']:.0f}%). A "
            f"floor removing most of a library usually means it was set for filtered data and "
            f"this input is raw, or the reverse."
        )

    logger.info(
        "QC floors: %d/%d barcodes and %d/%d genes below the floor.",
        summary["n_cells_below_floor"],
        summary["n_cells"],
        summary["n_genes_below_floor"],
        summary["n_genes"],
    )
    return FloorResult(
        cell_keep=cell_keep,
        gene_keep=gene_keep,
        reason=reason,
        summary=summary,
        warnings=warnings,
    )


def require_non_empty_qc_result(floors: FloorResult, *, n_genes: int) -> None:
    """Fail when the floors left nothing to analyse, naming the floor that did it.

    Args:
        floors: The applied floor result.
        n_genes: Genes surviving the gene floor.

    Raises:
        QCFloorError: If no cell or no gene survived.
    """
    reasons = floors.reason[floors.reason != ""].value_counts()
    culprit = (
        f" Every removal was attributed to: {', '.join(reasons.index.astype(str))}."
        if len(reasons)
        else ""
    )

    if int(floors.cell_keep.sum()) == 0:
        raise QCFloorError(
            f"The QC floors removed all {len(floors.cell_keep):,} barcodes, leaving nothing to "
            f"analyse.{culprit} A floor that removes everything is almost always set for a "
            f"different kind of input than the one supplied — `min_genes_per_cell` defaults to "
            f"200, which assumes a whole-transcriptome matrix and will empty a small panel, a "
            f"subsampled fixture, or an already-aggregated object. Lower the floors, or set them "
            f"to null to keep every barcode. Set `qc.fail_on_empty_result: false` to proceed "
            f"anyway."
        )

    if n_genes == 0:
        raise QCFloorError(
            f"The gene floor removed all {len(floors.gene_keep):,} genes, leaving nothing to "
            f"analyse. `min_cells_per_gene` is above the number of cells in which any gene is "
            f"detected here. Lower it, or set it to null. Set `qc.fail_on_empty_result: false` "
            f"to proceed anyway."
        )


def build_qc_report_table(
    cell_decisions: pd.DataFrame,
    *,
    groups: pd.Series | None = None,
    total_label: str = "TOTAL",
    group_name: str = "cell_type",
    unassigned_label: str = "unassigned",
) -> pd.DataFrame:
    """Per-group counts of cells before removal, removed, and remaining.

    Args:
        cell_decisions: Table indexed by every input cell with a boolean ``keep`` column —
            normally :meth:`FloorResult.cell_table`.
        groups: Optional per-cell group labels. Cells with a missing label are bucketed under
            ``unassigned_label`` so the group rows always sum to the total row. None collapses
            the report to a single total row.
        total_label: Label for the cohort-wide total row.
        group_name: Name of the leading group column.
        unassigned_label: Bucket label for cells whose group label is missing.

    Returns:
        One row per group in sorted order, followed by a single total row.

    Raises:
        QCFloorError: If ``cell_decisions`` lacks a ``keep`` column.
    """
    if "keep" not in cell_decisions.columns:
        raise QCFloorError(
            "build_qc_report_table needs a boolean 'keep' column indexed by every input cell. "
            f"Received columns: {sorted(cell_decisions.columns)[:20]}"
        )

    keep = cell_decisions["keep"].fillna(True).astype(bool)
    labels = (
        pd.Series(unassigned_label, index=cell_decisions.index, dtype=object)
        if groups is None
        else groups.reindex(cell_decisions.index).astype(object).fillna(unassigned_label)
    )

    rows: list[dict[str, object]] = []
    order = [] if groups is None else sorted(set(labels.astype(str)))
    for group in order:
        selected = (labels.astype(str) == group).to_numpy()
        before = int(selected.sum())
        removed = int((~keep.to_numpy())[selected].sum())
        rows.append(_report_row(group, before, removed, group_name))

    rows.append(_report_row(total_label, int(len(keep)), int((~keep).sum()), group_name))
    return pd.DataFrame(
        rows,
        columns=[group_name, "cells_before_qc", "cells_removed", "pct_removed", "cells_after_qc"],
    )


def _report_row(label: str, before: int, removed: int, group_name: str) -> dict[str, object]:
    """One report row, with the percentage computed where it is defined."""
    return {
        group_name: label,
        "cells_before_qc": before,
        "cells_removed": removed,
        "pct_removed": (100.0 * removed / before) if before else 0.0,
        "cells_after_qc": before - removed,
    }


__all__ = [
    "FLOOR_REASON_COLUMN",
    "FloorResult",
    "QCFloorError",
    "apply_floors",
    "floors_from_metrics",
    "build_qc_report_table",
    "require_non_empty_qc_result",
]
