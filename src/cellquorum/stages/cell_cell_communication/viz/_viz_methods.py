"""CCC visualization methods: thin AnalysisMethod subclasses that delegate to _plots."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip
from cellquorum.stages.cell_cell_communication.viz import _plots
from cellquorum.stages.cell_cell_communication.viz._io import (
    apply_theme,
    figure_artifacts,
    load_canonical_lr_sources,
    load_curvature,
    load_mnn_prioritization,
    load_mnn_sample_activity,
    load_nichenet,
    load_tensor_factors,
    load_topology,
    relabel_clusters,
    save_figure,
)

# ─── Dotplot ───────────────────────────────────────────────────────────────


class DotplotVizMethod(AnalysisMethod):
    """Render source->target x ligand-receptor dotplots from canonical LR frames."""

    name = "dotplot_viz"
    stage_category = "ccc_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        figures_dir = Path(context.paths.figures) / "ccc"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        top_k = int(config.get("top_k", 15))
        wanted = config.get("sources")

        sources = load_canonical_lr_sources(results_dir, getattr(adata, "uns", None))
        if wanted:
            sources = [(lab, df) for lab, df in sources if lab in set(wanted)]
        if not sources:
            return self._skip("no canonical LR sources found")
        labels = config.get("cluster_labels")
        sources = [(lab, relabel_clusters(df, ["source", "target"], labels)) for lab, df in sources]

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for label, df in sources:
            try:
                fig = _plots.interaction_dotplot(df, top_k=top_k)
                paths = save_figure(fig, figures_dir, f"dotplot_{label}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"LR dotplot ({label})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"dotplot_viz: {label} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("no plottable rows in any source", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"dotplot_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


# ─── Chord ─────────────────────────────────────────────────────────────────


class ChordVizMethod(AnalysisMethod):
    """Render chord/circos figures from canonical LR frames."""

    name = "chord_viz"
    stage_category = "ccc_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        figures_dir = Path(context.paths.figures) / "ccc"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        top_k = int(config.get("top_k", 15))
        wanted = config.get("sources")

        sources = load_canonical_lr_sources(results_dir, getattr(adata, "uns", None))
        if wanted:
            sources = [(lab, df) for lab, df in sources if lab in set(wanted)]
        if not sources:
            return self._skip("no canonical LR sources found")
        labels = config.get("cluster_labels")
        sources = [(lab, relabel_clusters(df, ["source", "target"], labels)) for lab, df in sources]

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for label, df in sources:
            try:
                cts = list(set(df["source"].astype(str)) | set(df["target"].astype(str)))
                palette = _plots.celltype_palette(cts)
                fig = _plots.chord_diagram(df, palette=palette, top_k=top_k)
                paths = save_figure(fig, figures_dir, f"chord_{label}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"Chord diagram ({label})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"chord_viz: {label} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("no plottable rows in any source", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"chord_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


# ─── Sankey ────────────────────────────────────────────────────────────────


class SankeyVizMethod(AnalysisMethod):
    """Render Sankey flow diagrams from canonical LR frames."""

    name = "sankey_viz"
    stage_category = "ccc_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        figures_dir = Path(context.paths.figures) / "ccc"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        top_k = int(config.get("top_k", 15))
        wanted = config.get("sources")

        sources = load_canonical_lr_sources(results_dir, getattr(adata, "uns", None))
        if wanted:
            sources = [(lab, df) for lab, df in sources if lab in set(wanted)]
        if not sources:
            return self._skip("no canonical LR sources found")
        labels = config.get("cluster_labels")
        sources = [(lab, relabel_clusters(df, ["source", "target"], labels)) for lab, df in sources]

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for label, df in sources:
            try:
                # Build palette from union of source/ligand/receptor/target values
                vals = (
                    set(df["source"].astype(str))
                    | set(df["target"].astype(str))
                    | set(df["ligand"].astype(str))
                    | set(df["receptor"].astype(str))
                )
                palette = _plots.celltype_palette(list(vals))
                fig = _plots.sankey_flow(df, palette=palette, top_k=top_k)
                paths = save_figure(fig, figures_dir, f"sankey_{label}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"Sankey diagram ({label})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"sankey_viz: {label} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("no plottable rows in any source", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"sankey_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


# ─── Network ───────────────────────────────────────────────────────────────


class NetworkVizMethod(AnalysisMethod):
    """Render curvature-colored networks from ccc_network curvature outputs."""

    name = "network_viz"
    stage_category = "ccc_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        figures_dir = Path(context.paths.figures) / "ccc"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        top_k = int(config.get("top_k", 15))
        wanted_levels = config.get("levels")
        levels = ("cci", "gci")
        if wanted_levels:
            levels = tuple(lv for lv in levels if lv in set(wanted_levels))

        curv = load_curvature(results_dir)
        if not curv:
            return self._skip("no curvature CSVs found")
        labels = config.get("cluster_labels")
        curv = {
            key: relabel_clusters(
                df, ["node"] if key.endswith("_nodes") else ["source", "target"], labels
            )
            for key, df in curv.items()
        }

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        # Whole-cohort levels: cci/gci (edges + nodes).
        for level in levels:
            edges = curv.get(f"{level}_edges")
            nodes = curv.get(f"{level}_nodes")
            if edges is None:
                continue
            try:
                fig = _plots.curvature_network(edges, nodes, top_k=top_k)
                paths = save_figure(fig, figures_dir, f"network_{level}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"Curvature network ({level})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"network_viz: {level} failed: {str(exc)[:200]}")
        # Differential levels: color by delta_curvature.
        for level in levels:
            diff = curv.get(f"differential_{level}")
            if diff is None or diff.empty:
                continue
            try:
                fig = _plots.curvature_network(
                    diff,
                    None,
                    curvature_col="delta_curvature",
                    node_curv_col="delta_curvature",
                    top_k=top_k,
                )
                paths = save_figure(
                    fig, figures_dir, f"network_diff_{level}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths,
                    name="ccc_figure",
                    description=f"Differential curvature network ({level}).",
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"network_viz: differential {level} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("no plottable curvature levels", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"network_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


# ─── Summary ───────────────────────────────────────────────────────────────


class SummaryVizMethod(AnalysisMethod):
    """Render CCI heatmaps and topology role facets."""

    name = "summary_viz"
    stage_category = "ccc_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        figures_dir = Path(context.paths.figures) / "ccc"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        top_k = int(config.get("top_k", 15))

        sources = load_canonical_lr_sources(results_dir, getattr(adata, "uns", None))
        wanted = config.get("sources")
        if wanted:
            sources = [(lab, df) for lab, df in sources if lab in set(wanted)]
        topo = load_topology(results_dir)
        if not sources and not topo:
            return self._skip("no LR sources or topology found")
        labels = config.get("cluster_labels")
        sources = [(lab, relabel_clusters(df, ["source", "target"], labels)) for lab, df in sources]
        topo = {level: relabel_clusters(df, ["node"], labels) for level, df in topo.items()}

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for label, df in sources:
            try:
                fig = _plots.cci_heatmap(df, diverging=False)
                paths = save_figure(fig, figures_dir, f"heatmap_{label}", formats=formats, dpi=dpi)
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description=f"CCI heatmap ({label})."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"summary_viz: heatmap {label} failed: {str(exc)[:200]}")
        for level, df in topo.items():
            try:
                figs = _plots.topology_facet_figures(df, top_k=top_k)
                for metric, fig in figs.items():
                    paths = save_figure(
                        fig,
                        figures_dir,
                        f"topology_{level}_{metric.lower()}",
                        formats=formats,
                        dpi=dpi,
                    )
                    artifacts += figure_artifacts(
                        paths,
                        name="ccc_figure",
                        description=f"Topology role: {metric} ({level}).",
                    )
                    n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"summary_viz: topology {level} failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("nothing plottable", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"summary_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


class TensorFactorsVizMethod(AnalysisMethod):
    """Render the tensor-cell2cell factor-decomposition plot.

    tensor_c2c_method.py writes four real, computed factor-loading tables
    (contexts/senders/receivers/lr_pairs); nothing ever visualized them.
    """

    name = "tensor_factors_viz"
    stage_category = "ccc_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        figures_dir = Path(context.paths.figures) / "ccc"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        top_k_lr = int(config.get("tensor_top_k_lr", 200))

        factors = load_tensor_factors(results_dir)
        if factors is None:
            return self._skip("tensor_c2c factor tables not found or incomplete")
        # senders/receivers are indexed BY cluster code (not a column), so
        # relabel_clusters (column-based) doesn't apply -- relabel the index
        # directly, same display-only {code -> name} override.
        cluster_labels = config.get("cluster_labels")
        if cluster_labels:
            factors = dict(factors)
            for key in ("senders", "receivers"):
                factors[key] = factors[key].rename(
                    index=lambda v: cluster_labels.get(str(v), str(v))
                )

        # Color contexts by condition when a sample name follows the
        # <sample>_<condition> convention (e.g. "P1_LE") -- the single most
        # informative grouping for a context axis, and free to infer since
        # it costs nothing when the convention doesn't hold (falls back to
        # an uncolored bar rather than guessing wrong).
        context_palette = config.get("context_palette")
        if context_palette is None:
            conditions = {
                str(ctx): str(ctx).rsplit("_", 1)[-1]
                for ctx in factors["contexts"].index
                if "_" in str(ctx)
            }
            distinct = sorted(set(conditions.values()))
            if len(distinct) > 1:
                group_color = _plots.celltype_palette(distinct)
                context_palette = {ctx: group_color[cond] for ctx, cond in conditions.items()}

        node_palette = config.get("node_palette")
        if node_palette is None:
            node_ids = sorted(set(factors["senders"].index) | set(factors["receivers"].index))
            node_palette = _plots.celltype_palette(node_ids)

        apply_theme()
        try:
            fig = _plots.plot_tensor_factors(
                factors["contexts"],
                factors["senders"],
                factors["receivers"],
                factors["lr_pairs"],
                context_palette=context_palette,
                node_palette=node_palette,
                top_k_lr=top_k_lr,
            )
            paths = save_figure(fig, figures_dir, "tensor_factors", formats=formats, dpi=dpi)
        except Exception as exc:  # noqa: BLE001
            return self._skip(f"tensor_factors_viz failed: {str(exc)[:200]}")

        artifacts = figure_artifacts(
            paths,
            name="ccc_figure",
            description="Tensor-cell2cell factor decomposition "
            "(contexts x LR pairs x senders x receivers).",
        )
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"tensor_factors_viz rendered {factors['contexts'].shape[1]} factors."],
            metrics={"method": self.name, "n_factors": factors["contexts"].shape[1]},
            backend="python",
        )


# ─── MultiNicheNet ───────────────────────────────────────────────────────


class MultiNicheNetVizMethod(AnalysisMethod):
    """Render MultiNicheNet's own prioritization table as a top-LR lollipop.

    The circos/chord view (chord_viz on mnn_canonical_lr.csv) aggregates to
    the cell-type-pair level and can't say which ligand-receptor pair is
    driving a ribbon. This reads the native mnn_prioritization.csv directly
    and names each interaction, ranked by MultiNicheNet's own
    prioritization_score, independently per condition group.
    """

    name = "multinichenet_viz"
    stage_category = "ccc_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        figures_dir = Path(context.paths.figures) / "ccc"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        top_n = int(config.get("mnn_top_n_lr", 30))

        df = load_mnn_prioritization(results_dir)
        if df is None:
            return self._skip("no mnn_prioritization.csv found")
        labels = config.get("cluster_labels")
        group_col = "group" if "group" in df.columns else None
        groups = sorted(df[group_col].astype(str).unique()) if group_col else [None]

        key_clusters = config.get("mnn_key_clusters")
        if not key_clusters:
            key_clusters = sorted(set(df["sender"].astype(str)) | set(df["receiver"].astype(str)))
        circos_top_k = int(config.get("mnn_circos_top_k", 8))
        palette = _plots.celltype_palette(key_clusters)

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0
        for group in groups:
            subset = df[df[group_col].astype(str) == group] if group_col else df
            try:
                # One group per call (group_col omitted from the plot call):
                # a figure combining every group needs a legend entry per
                # sender->receiver route seen in ANY group, which is where
                # the circos-replacement point (name the actual LR pair) gets
                # undone by an illegible legend instead.
                fig = _plots.mnn_top_lr_lollipop(subset, top_n=top_n, cluster_labels=labels)
                suffix = f"_{group}" if group else ""
                paths = save_figure(
                    fig, figures_dir, f"multinichenet_top_lr{suffix}", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths,
                    name="ccc_figure",
                    description=(
                        f"MultiNicheNet top prioritized ligand-receptor interactions"
                        f"{f' ({group})' if group else ''}."
                    ),
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"multinichenet_viz: {group} failed: {str(exc)[:200]}")

            for cluster in key_clusters:
                cluster = str(cluster)
                touches = (subset["sender"].astype(str) == cluster) | (
                    subset["receiver"].astype(str) == cluster
                )
                if not touches.any():
                    # No blank "no data" placeholder for a cluster this
                    # group's subset never mentions -- real signal only.
                    continue
                cluster_label = (labels or {}).get(str(cluster), str(cluster))
                try:
                    fig = _plots.cluster_focused_circos(
                        subset,
                        cluster,
                        palette=palette,
                        top_k=circos_top_k,
                        cluster_labels=labels,
                        title=f"{cluster_label} ({group})" if group else cluster_label,
                    )
                    stem = f"multinichenet_circos_{cluster_label}{suffix}"
                    paths = save_figure(fig, figures_dir, stem, formats=formats, dpi=dpi)
                    if paths:
                        artifacts += figure_artifacts(
                            paths,
                            name="ccc_figure",
                            description=(
                                f"MultiNicheNet circos focused on {cluster_label}"
                                f"{f' ({group})' if group else ''}."
                            ),
                        )
                        n_figures += 1
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        f"multinichenet_viz: circos {cluster} ({group}) failed: {str(exc)[:200]}"
                    )

        sample_activity = load_mnn_sample_activity(results_dir)
        if sample_activity is not None:
            sample_df, source_df = sample_activity
            # scaled_pb_ligand/scaled_pb_receptor (celltype-specificity color
            # for the 4th panel) live in group_prioritization_tbl (df, loaded
            # above), not in group_prioritization_table_source -- confirmed
            # against a real MultiNicheNet rerun, where the two tables are
            # genuinely distinct outputs of the same R call. Merge on
            # (id, group) before plotting; a source_df is never used without
            # this merge, so plot_mnn_sample_activity can assume it's there.
            if "scaled_pb_ligand" in df.columns and "id" in source_df.columns:
                source_df = source_df.merge(
                    df[["id", "group", "scaled_pb_ligand", "scaled_pb_receptor"]],
                    on=["id", "group"],
                    how="left",
                )
            try:
                fig = _plots.plot_mnn_sample_activity(
                    sample_df,
                    source_df,
                    top_n=int(config.get("mnn_sample_activity_top_n", 20)),
                    cluster_labels=labels,
                )
                paths = save_figure(
                    fig, figures_dir, "multinichenet_sample_activity", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths,
                    name="ccc_figure",
                    description=(
                        "MultiNicheNet per-sample expression, ligand activity, and "
                        "celltype specificity for the top prioritized LR interactions."
                    ),
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"multinichenet_viz: sample_activity failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("nothing plottable", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"multinichenet_viz rendered {n_figures} figure(s)."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


# ─── NicheNet ────────────────────────────────────────────────────────────


class NicheNetVizMethod(AnalysisMethod):
    """Render NicheNet's own outputs: ligand activity, ligand-target, sender dotplot.

    nichenet_method.py already computes and writes these three tables; the
    generic canonical-LR pipeline (dotplot_viz/chord_viz/sankey_viz) only
    ever sees NicheNet's thin LR-edge projection of them, so the analysis's
    actual headline result -- which ligand best explains the receiver's DE
    geneset, and through which targets -- never reached a figure.
    """

    name = "nichenet_viz"
    stage_category = "ccc_viz"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        results_dir = Path(context.paths.results)
        figures_dir = Path(context.paths.figures) / "ccc"
        formats = tuple(config.get("figure_formats", ["pdf", "png"]))
        dpi = int(config.get("dpi", 300))
        top_n = int(config.get("nichenet_top_n_ligands", 20))

        tables = load_nichenet(results_dir)
        if not tables:
            return self._skip("no NicheNet output tables found")
        labels = config.get("cluster_labels")
        if labels and "sender_expression" in tables:
            tables["sender_expression"] = relabel_clusters(
                tables["sender_expression"], ["sender"], labels
            )

        apply_theme()
        artifacts, warnings, n_figures = [], [], 0

        if "activities" in tables:
            try:
                fig = _plots.ligand_activity_bar(tables["activities"], top_n=top_n)
                paths = save_figure(
                    fig, figures_dir, "nichenet_ligand_activity", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description="NicheNet ligand-activity ranking."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"nichenet_viz: ligand_activity failed: {str(exc)[:200]}")

        if "weights" in tables:
            try:
                fig = _plots.ligand_target_heatmap(tables["weights"])
                paths = save_figure(
                    fig, figures_dir, "nichenet_ligand_target", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths,
                    name="ccc_figure",
                    description="NicheNet ligand->target regulatory potential.",
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"nichenet_viz: ligand_target failed: {str(exc)[:200]}")

        if "sender_expression" in tables:
            try:
                fig = _plots.sender_ligand_dotplot(tables["sender_expression"])
                paths = save_figure(
                    fig, figures_dir, "nichenet_sender_dotplot", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths, name="ccc_figure", description="NicheNet sender ligand expression."
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"nichenet_viz: sender_dotplot failed: {str(exc)[:200]}")

        if "receptor_weights" in tables:
            try:
                fig = _plots.ligand_receptor_heatmap(tables["receptor_weights"])
                paths = save_figure(
                    fig, figures_dir, "nichenet_ligand_receptor", formats=formats, dpi=dpi
                )
                artifacts += figure_artifacts(
                    paths,
                    name="ccc_figure",
                    description="NicheNet ligand->receptor prior interaction potential.",
                )
                n_figures += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"nichenet_viz: ligand_receptor failed: {str(exc)[:200]}")

        if n_figures == 0:
            return self._skip("nothing plottable", warnings=warnings)
        return StageResult(
            adata=adata,
            artifacts=artifacts,
            notes=[f"nichenet_viz rendered {n_figures} figures."],
            warnings=warnings,
            metrics={"method": self.name, "n_figures": n_figures},
            backend="python",
        )


__all__ = [
    "DotplotVizMethod",
    "ChordVizMethod",
    "SankeyVizMethod",
    "NetworkVizMethod",
    "SummaryVizMethod",
    "TensorFactorsVizMethod",
    "NicheNetVizMethod",
    "MultiNicheNetVizMethod",
]
