"""Annotation figure methods: the marker dotplot that audits the labels."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.visualization.figio import figure_artifacts
from cellquorum.visualization.figstyle import save_figure
from cellquorum.visualization.marker_panels import (
    dendrogram_row_order,
    draw_categorized_dotplot,
    draw_lineage_marker_dotplot,
    marker_panel_statistics,
    scale_columns_unit,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cellquorum.core.context import PipelineContext


def expected_blocks(groups: Sequence[str], panel: Mapping[str, list[str]]) -> dict[str, str]:
    """Each group is expected to light up the panel block that carries its name.

    A group with no block of its name (a lineage the panel does not cover) has no expectation,
    so it gets no highlight rather than a guessed one.
    """
    return {group: group for group in groups if group in panel}


class _Resolved(NamedTuple):
    """What both dotplots read from the annotation block, resolved once."""

    panel: dict[str, list[str]]
    group_by: str
    layer: str
    groups: list[str]


def _resolve(
    adata: ad.AnnData, config: dict, skip: Callable[[str], MethodSkip]
) -> _Resolved | MethodSkip:
    """Panel, label column, layer and group order, or the reason the figure cannot be drawn."""
    panel = {str(k): list(v) for k, v in (config.get("marker_panels") or {}).items()}
    if not panel:
        return skip(
            "no marker_panels configured; set annotation.marker_panels as "
            "{cell_type: [genes]} to draw the marker dotplot"
        )
    group_by = config.get("group_by") or "cell_type"
    if group_by not in adata.obs.columns:
        return skip(f"obs column '{group_by}' not found")
    layer = config.get("layer") or "cellquorum_normalized"
    if layer not in adata.layers:
        return skip(f"layer '{layer}' not found")
    column = adata.obs[group_by]
    observed = set(column.astype(str))
    # Only a categorical marked ordered keeps its own order. Anything else, including the
    # unordered categorical an h5ad stores string labels as (alphabetical by accident), follows
    # the panel, so a panel written lineage by lineage puts each group under its own block and
    # the highlighted cells form a diagonal; labels the panel does not name come last.
    groups = (
        [str(c) for c in column.cat.categories if str(c) in observed]
        if hasattr(column, "cat") and column.cat.ordered
        else [g for g in panel if g in observed] + sorted(observed - set(panel))
    )
    return _Resolved(panel, group_by, layer, groups)


class MarkerDotplotMethod(AnalysisMethod):
    """Draw the red marker dotplot: genes in boxed panel blocks, one column per label.

    Dot colour is mean expression and dot size the percent of cells expressing it. Where the
    labels carry the names of the panel's blocks, each group's own block is banded in cream, so
    the figure answers at a glance whether every label lights up its own markers and only its own.
    """

    name = "marker_dotplot"
    stage_category = "annotation_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        resolved = _resolve(adata, config, self._skip)
        if isinstance(resolved, MethodSkip):
            return resolved
        panel, group_by, layer, groups = resolved
        highlighted = (
            expected_blocks(groups, panel) if config.get("highlight_expected", True) else {}
        )
        fig = draw_categorized_dotplot(
            adata,
            panel,
            groupby=group_by,
            layer=layer,
            categories_order=groups,
            vmax_percentile=float(config.get("vmax_percentile", 95.0)),
            title=config.get("title"),
            expected=highlighted,
        )
        if fig is None:
            return self._skip("none of the marker_panels genes is present in the data")

        figures_dir = Path(context.paths.figures) / "annotation"
        results_dir = Path(context.paths.results) / "annotation"
        paths = save_figure(
            fig,
            figures_dir,
            "marker_dotplot",
            formats=tuple(config.get("figure_formats", ["pdf", "png"])),
            dpi=int(config.get("dpi", 300)),
        )
        mean, pct = _panel_tables(adata, panel, groups, group_by=group_by, layer=layer)
        results_dir.mkdir(parents=True, exist_ok=True)
        table_artifacts = []
        for summary, frame, description in [
            ("mean", mean, "Mean normalized marker expression by cell group."),
            ("pct", pct, "Percent of cells expressing each marker by cell group."),
        ]:
            path = results_dir / f"marker_dotplot_{summary}.csv"
            frame.to_csv(path)
            table_artifacts.append(
                StageArtifact(f"marker_dotplot_{summary}", path, "csv", description)
            )
        return StageResult(
            adata=adata,
            artifacts=figure_artifacts(
                paths,
                name="marker_dotplot",
                description=(
                    f"Marker panel by {group_by}: dot colour is mean expression, dot size the "
                    "percent of cells expressing; a cream band marks each group's own block."
                ),
            )
            + table_artifacts,
            notes=[f"marker_dotplot drew {len(groups)} group(s), {len(highlighted)} highlighted."],
            warnings=[],
            metrics={
                "method": self.name,
                "group_by": group_by,
                "layer": layer,
                "highlighted": list(highlighted),
            },
            backend="python",
        )


class LineageDotplotMethod(AnalysisMethod):
    """Draw the clustered lineage dotplot: groups as rows ordered by a dendrogram, genes as columns.

    Colour is the mean scaled per gene across groups and dot size the percent of cells expressing,
    with a coloured header bar over each panel block. Where the first dotplot asks whether each
    label lights up its own markers, this one asks which labels resemble each other: groups that
    share a marker profile sit together on the tree whatever their names say.
    """

    name = "lineage_dotplot"
    stage_category = "annotation_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
        resolved = _resolve(adata, config, self._skip)
        if isinstance(resolved, MethodSkip):
            return resolved
        panel, group_by, layer, groups = resolved
        if len(groups) < 2:
            return self._skip("the clustered dotplot needs at least two groups to build a tree")
        stats = marker_panel_statistics(adata, panel, groupby=group_by, layer=layer)
        if not stats.genes:
            return self._skip("none of the marker_panels genes is present in the data")
        order, _ = dendrogram_row_order(scale_columns_unit(stats.mean))
        fig = draw_lineage_marker_dotplot(stats, title=config.get("title"))
        paths = save_figure(
            fig,
            Path(context.paths.figures) / "annotation",
            "lineage_marker_dotplot",
            formats=tuple(config.get("figure_formats", ["pdf", "png"])),
            dpi=int(config.get("dpi", 300)),
        )
        results_dir = Path(context.paths.results) / "annotation"
        results_dir.mkdir(parents=True, exist_ok=True)
        # One flat header ("block|gene"): a gene can sit in two blocks, and a two-row header breaks
        # plain CSV readers.
        columns = [f"{b}|{g}" for b, g in zip(stats.gene_group, stats.genes, strict=True)]
        pd.DataFrame(stats.mean, index=stats.groups, columns=columns).to_csv(
            results_dir / "lineage_marker_dotplot_mean.csv"
        )
        pd.DataFrame(stats.pct, index=stats.groups, columns=columns).to_csv(
            results_dir / "lineage_marker_dotplot_pct.csv"
        )
        return StageResult(
            adata=adata,
            artifacts=figure_artifacts(
                paths,
                name="lineage_marker_dotplot",
                description=(
                    f"Marker panel by {group_by}, groups ordered by hierarchical clustering of "
                    "their marker profiles."
                ),
            ),
            notes=[f"lineage_dotplot drew {len(stats.groups)} group(s)."],
            warnings=[],
            metrics={
                "method": self.name,
                "group_by": group_by,
                "layer": layer,
                "row_order": [stats.groups[i] for i in order],
            },
            backend="python",
        )


def _panel_tables(
    adata: ad.AnnData,
    panel: Mapping[str, list[str]],
    groups: Sequence[str],
    *,
    group_by: str,
    layer: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean expression and percent expressing: panel genes (rows, panel order) by group."""
    genes = [g for block in panel.values() for g in block if g in adata.var_names]
    matrix = adata.layers[layer][:, adata.var_names.get_indexer(genes)]
    matrix = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    labels = adata.obs[group_by].astype(str).to_numpy()
    mean = np.column_stack([matrix[labels == g].mean(axis=0) for g in groups])
    pct = np.column_stack([(matrix[labels == g] > 0).mean(axis=0) * 100.0 for g in groups])
    return (
        pd.DataFrame(mean, index=genes, columns=list(groups)),
        pd.DataFrame(pct, index=genes, columns=list(groups)),
    )


__all__ = ["LineageDotplotMethod", "MarkerDotplotMethod", "expected_blocks"]
