"""NES-by-cell-type heatmap: which gene sets move, in which cell types, in which direction.

The matrices are prepared here (selection, ordering, labels) so they can be tested; the figure
itself is drawn by ``gsea_nes_heatmap.R`` through ComplexHeatmap, which gives the real
dendrograms, per-row and per-column count bars and a single stacked legend a matplotlib
rendering would have to fake.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from cellquorum.backends.script_paths import r_script_path
from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.r_method import RAnalysisMethod

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cellquorum.core.context import PipelineContext

_NES_HEATMAP_R = r_script_path("gsea_nes_heatmap.R")
_SUFFIX = "_by_cell_type.csv"
_PREFIX = "enrichment_gsea_"
_DE_BY_GROUP_CSV = "de_pseudobulk_edger_by_subcluster.csv"

#: Diverging palettes ``gsea_nes_heatmap.R`` knows by name.
PALETTES = ("teal_orange", "coolwarm", "rdbu")


def nes_matrices(
    table: pd.DataFrame,
    *,
    alpha: float,
    min_significant: int,
    max_rows: int | None,
    labels: Mapping[str, str] | None = None,
    cell_type_col: str = "cell_type",
    source_col: str = "source",
    score_col: str = "score",
    padj_col: str = "padj",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gene sets x cell types matrices of NES and adjusted p, restricted to informative rows.

    A gene set is kept when it is significant (``padj < alpha``) in at least
    ``min_significant`` cell types; ``max_rows`` then keeps the sets with the strongest
    significant score. A set a cell type could not test (too few of its genes ranked) is NaN
    there, never zero: zero would claim a measured absence of enrichment. Columns keep the
    order cell types first appear in ``table``.

    Returns:
        ``(nes, padj)``, same shape and labels.
    """
    columns = list(dict.fromkeys(table[cell_type_col].astype(str)))
    nes = table.pivot(index=source_col, columns=cell_type_col, values=score_col)
    padj = table.pivot(index=source_col, columns=cell_type_col, values=padj_col)
    nes.columns = nes.columns.astype(str)
    padj.columns = padj.columns.astype(str)
    nes, padj = nes.reindex(columns=columns), padj.reindex(index=nes.index, columns=columns)

    significant = (padj < alpha).fillna(False)
    keep = significant.sum(axis=1) >= min_significant
    nes, padj, significant = nes[keep], padj[keep], significant[keep]
    strength = nes.where(significant).abs().max(axis=1).fillna(0.0)
    order = strength.sort_values(ascending=False, kind="mergesort").index
    if max_rows is not None:
        order = order[:max_rows]
    nes, padj = nes.loc[order], padj.loc[order]
    if labels:
        rename = {c: labels.get(c, c) for c in columns}
        nes, padj = nes.rename(columns=rename), padj.rename(columns=rename)
    return nes, padj


class GseaNesHeatmapMethod(RAnalysisMethod):
    """Draw the NES-by-cell-type heatmap for every collection that has a by-cell-type table."""

    name = "gsea_nes_heatmap"
    stage_category = "enrichment_viz"
    r_package = "ComplexHeatmap"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        backend = self._resolve_rscript_backend(context, config)
        if isinstance(backend, MethodSkip):
            return backend

        results_dir = Path(context.paths.results)
        found = sorted(results_dir.glob(f"{_PREFIX}*{_SUFFIX}"))
        wanted = config.get("collections")
        tables = {
            p.name[len(_PREFIX) : -len(_SUFFIX)]: p
            for p in found
            if wanted is None or p.name[len(_PREFIX) : -len(_SUFFIX)] in set(wanted)
        }
        if not tables:
            return self._skip(
                "no by-cell-type GSEA table in results; run the enrichment method "
                "gsea_by_cell_type (set differential_expression.cell_type_col)"
            )

        palette = str(config.get("palette", "teal_orange"))
        if palette not in PALETTES:
            return self._skip(f"unknown palette '{palette}'; choose one of {list(PALETTES)}")
        alpha = float(config.get("alpha", config.get("fdr", 0.05)))
        strong = float(config.get("alpha_strong", 0.001))
        labels = config.get("cluster_labels") or {}
        # The columns are whatever the differential expression was split by (cell types, or the
        # states of one lineage); the counts and the wording must follow that, not a fixed name.
        design_de = getattr(getattr(context, "config", None), "differential_expression", None)
        group_col = config.get("cell_type_col") or getattr(design_de, "cell_type_col", None)
        group_col = str(group_col or "cell_type")
        group_label = str(config.get("group_label", "cell type"))
        scratch = Path(context.paths.scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        figures_dir = Path(context.paths.figures) / "enrichment"
        figures_dir.mkdir(parents=True, exist_ok=True)
        timeout = int(config.get("timeout_seconds", 600))

        artifacts: list[StageArtifact] = []
        notes: list[str] = []
        reasons: list[str] = []
        for collection, path in tables.items():
            table = pd.read_csv(path)
            nes, padj = nes_matrices(
                table,
                alpha=alpha,
                min_significant=int(config.get("min_significant", 1)),
                max_rows=config.get("max_rows"),
                labels=labels,
            )
            if nes.empty:
                reasons.append(f"{collection}: no gene set is significant at padj < {alpha:g}")
                continue
            footnote = self._footnote(results_dir, table, labels)
            nes.to_csv(scratch / f"nes_{collection}.csv")
            padj.to_csv(scratch / f"nes_padj_{collection}.csv")
            n_cells_csv = self._write_cell_counts(
                adata,
                group_col,
                labels,
                nes.columns,
                scratch,
                collection,
            )
            out_png = figures_dir / f"gsea_nes_heatmap_{collection}.png"
            title = config.get("title") or f"GSEA {collection.title()}: NES by {group_label}"
            try:
                proc = backend.run_script(
                    _NES_HEATMAP_R,
                    [
                        str(scratch / f"nes_{collection}.csv"),
                        str(scratch / f"nes_padj_{collection}.csv"),
                        str(out_png),
                        str(alpha),
                        str(strong),
                        title,
                        footnote,
                        palette,
                        n_cells_csv,
                        f"{group_label}s",
                    ],
                    timeout=timeout,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
                reasons.append(f"{collection}: gsea_nes_heatmap.R failed or timed out ({exc})")
                continue
            if proc.returncode != 0:
                rc = proc.returncode
                how = f"killed by {signal.Signals(-rc).name}" if rc < 0 else "failed"
                reasons.append(f"{collection}: gsea_nes_heatmap.R {how}")
                continue
            if not out_png.exists():
                reasons.append(f"{collection}: gsea_nes_heatmap.R wrote no output")
                continue
            for path_out in (out_png, out_png.with_suffix(".pdf")):
                if path_out.exists():
                    artifacts.append(
                        StageArtifact(
                            name="gsea_nes_heatmap",
                            path=path_out,
                            kind="figure",
                            description=(
                                f"GSEA {collection} NES by cell type: {nes.shape[0]} gene sets "
                                f"significant (padj < {alpha:g}) in at least one of "
                                f"{nes.shape[1]} cell types."
                            ),
                        )
                    )
            notes.append(
                f"gsea_nes_heatmap drew {nes.shape[0]} gene sets x {nes.shape[1]} cell types "
                f"({collection})."
            )
        if not artifacts:
            return self._skip("no heatmap was drawn: " + "; ".join(reasons))
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=notes,
            warnings=reasons,
            metrics={"method": self.name, "collections": sorted(tables)},
            backend="rscript",
        )

    @staticmethod
    def _write_cell_counts(
        adata: ad.AnnData,
        cell_type_col: str,
        labels: Mapping[str, str],
        columns: pd.Index,
        scratch: Path,
        collection: str,
    ) -> str:
        """Cells behind each heatmap column, so a column's reliability is visible on the figure.

        Returns the CSV path, or "" when the object has no such column (the figure is then drawn
        without the bar rather than with invented counts).
        """
        if cell_type_col not in adata.obs.columns:
            return ""
        counts = adata.obs[cell_type_col].astype(str).value_counts()
        shown = {labels.get(str(k), str(k)): int(v) for k, v in counts.items()}
        frame = pd.DataFrame({"n_cells": [shown.get(str(c), 0) for c in columns]}, index=columns)
        path = scratch / f"nes_ncells_{collection}.csv"
        frame.to_csv(path)
        return str(path)

    @staticmethod
    def _footnote(results_dir: Path, table: pd.DataFrame, labels: Mapping[str, str]) -> str:
        """Name the cell types the DE ran on but GSEA could not score, so no column vanishes."""
        de_path = results_dir / _DE_BY_GROUP_CSV
        if not de_path.exists():
            return ""
        de_groups = set(pd.read_csv(de_path, usecols=["cell_type"])["cell_type"].astype(str))
        missing = sorted(de_groups - set(table["cell_type"].astype(str)))
        if not missing:
            return ""
        names = ", ".join(labels.get(m, m) for m in missing)
        return f"Cell types not tested (too few ranked genes): {names}."


__all__ = ["PALETTES", "GseaNesHeatmapMethod", "nes_matrices"]
