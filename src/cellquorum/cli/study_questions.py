"""Resolve biological questions to portable, editable engine settings."""

import sys
from pathlib import Path

import pandas as pd
import typer
import yaml


def _profile(name: str) -> dict:
    for directory in (
        Path(__file__).resolve().parents[3] / "configs/profiles",
        Path(sys.prefix) / "share/cellquorum/profiles",
    ):
        path = directory / f"{name}.yaml"
        if path.is_file():
            return yaml.safe_load(path.read_text())
    raise ValueError(f"The installed CellQuorum {name} profile is missing.")


def configure_questions(
    config: dict,
    enabled: set[str],
    obs: pd.DataFrame,
    layers: list[str],
    mode: str,
    manifest: pd.DataFrame | None,
) -> None:
    from cellquorum.cli.study import _abundance, _choose, _free_layer

    questions = list(
        ("populations", "differences", "communication", "regulation", "lineage", "perturbation")
    )
    if mode == "discovery":
        selected = set(questions)
    else:
        text = typer.prompt("Study questions, separated by commas (" + ", ".join(questions) + ")")
        selected = {s.strip() for s in text.split(",")}
        if not selected or selected - set(questions):
            raise ValueError(f"Unknown study questions: {sorted(selected - set(questions))}")
    actual = selected.copy()
    cols = list(obs.columns)
    norm = config["preprocessing"]["normalization"]
    layer, counts = norm["output_layer"], norm["preserve_counts_layer"]
    sample = config["cohort"].get("sample_key")
    contrast = False
    if selected & {"differences", "communication", "regulation"}:
        contrast = typer.confirm(
            "Does this study have a replicated two-condition comparison?", default=False
        )
    if contrast:
        _abundance(config, obs, infer_clusters=True)
        celltype = config["composition"]["cell_type_key"]
    else:
        celltype = (
            _choose(
                "Column with population labels, or Enter to use new Leiden clusters",
                ["", *cols],
                *[k for k in ("cell_type", "celltype_alt") if k in cols],
                "",
            )
            or "leiden"
        )
        if "differences" in selected:
            actual.discard("differences")
            typer.echo("Condition comparisons require biological replicates and a contrast.")
    config["run"] = {"profile": "publication", "require_selected_methods": True}
    config["report"]["fail_on_report_error"] = True
    expected = {}
    method_omissions = {}

    def profile(name: str) -> None:
        for field, values in _profile(name).items():
            config.setdefault(field, {}).update(values)

    has_splice = {"spliced", "unspliced"} <= set(layers) or (
        manifest is not None and "loom_path" in manifest and manifest.loom_path.notna().all()
    )
    matrix_columns = ["raw_matrix_path", "filtered_matrix_path"]
    if manifest is not None and set(matrix_columns) <= set(manifest):
        complete = manifest[matrix_columns].notna().all().all()
        if complete and typer.confirm(
            "Estimate ambient RNA from raw droplets and carry corrected counts forward?",
            default=True,
        ):
            config["r"]["enabled"] = True
            enabled.add("ambient_correction")
            config["ambient_correction"] = {"enabled": True}
            counts = "counts"
            norm.update(input_layer=counts, preserve_counts_layer=counts)
            config["feature_selection"]["counts_layer"] = counts
            config["qc_evidence"]["metrics"]["layer"] = counts
    if has_splice:
        enabled.add("qc_splice_metrics")
        config["qc_splice_metrics"] = {"sample_col": sample or "sample_id"}
    if "populations" in selected:
        profile("standard")
        enabled.update(("population_identity", "state_scoring", "state_scoring_viz", "discovery"))
        config["population_identity"] = {"candidate_key": celltype, "annotation_keys": []}
        config["discovery"] = {"cell_type_col": celltype, "layer": layer, "max_iter": 2000}
        expected["discovery"] = [
            "results/discovery_nmf_top_genes.csv",
            "results/discovery_nmf_usage_by_celltype.csv",
        ]
        if celltype not in config["embeddings"]["color_by"]:
            config["embeddings"]["color_by"].append(celltype)
        config["state_scoring"].update(layer=layer, cell_type_col=celltype)
        config["state_scoring_viz"] = {"cluster_key": celltype, "descriptive": not contrast}
        markers = typer.prompt(
            "Tissue marker panel file (YAML mapping cell types to gene lists), or Enter to omit",
            default="",
            show_default=False,
        )
        if markers:
            parsed = yaml.safe_load(Path(markers).expanduser().read_text())
            if (
                not isinstance(parsed, dict)
                or not parsed
                or not all(isinstance(v, list) and v for v in parsed.values())
            ):
                raise ValueError(
                    "The marker panel needs a nonempty gene list for each named population."
                )
            marker_key = _free_layer("cq_marker_label", cols)
            enabled.update(("annotation", "annotation_viz"))
            config["annotation"] = {
                "method": "marker_vote",
                "marker_panels": parsed,
                "score_layer": layer,
                "key_added": marker_key,
            }
            config["annotation_viz"] = {
                "group_by": celltype,
                "methods": [{"method": "marker_dotplot"}],
            }
            config["population_identity"]["annotation_keys"] = [marker_key]
            config["embeddings"]["color_by"].append(marker_key)
            expected["annotation_viz"] = [
                "figures/annotation/marker_dotplot.pdf",
                "results/annotation/marker_dotplot_mean.csv",
                "results/annotation/marker_dotplot_pct.csv",
            ]
            expected["population_identity"] = [
                "results/population_identity/tables/population_by_annotation.csv"
            ]
        atlas = typer.prompt(
            "Labeled reference atlas (.h5ad), or Enter to omit transfer",
            default="",
            show_default=False,
        )
        if atlas:
            annotation = typer.prompt("Population label column in the atlas", default="cell_type")
            enabled.add("reference_mapping")
            config["reference_mapping"] = {
                "enabled": True,
                "atlas_h5ad": atlas,
                "label_key": annotation,
                "counts_layer": counts,
            }
        if celltype in obs:
            focus = _choose(
                "Population to subcluster, or Enter to omit",
                ["", *sorted(obs[celltype].dropna().astype(str).unique())],
                "",
            )
            if focus:
                enabled.add("subclustering")
                config["subclustering"] = {
                    "enabled": True,
                    "focus": {"label_key": celltype, "labels": [focus]},
                    "counts_layer": counts,
                }
        batch = _choose("Technical batch column for integration, or Enter to omit", ["", *cols], "")
        if batch:
            config["cohort"]["batch_key"] = batch
            if contrast:
                cond = config["design"]["condition_col"]
                arms = [config["design"][k] for k in ("case", "control")]
                table = pd.crosstab(obs[batch], obs[cond]).reindex(columns=arms, fill_value=0)
                if not table.gt(0).all(axis=1).any():
                    raise ValueError("Condition and technical batch are fully confounded.")
            enabled.update(("integration", "integration_benchmark"))
            config["integration"] = {"method": "harmony", "batch_key": batch}
            config["integration_benchmark"] = {
                "enabled": True,
                "batch_key": batch,
                "label_key": celltype,
                "mode": "full" if celltype in obs else "batch_only",
                "embeddings": ["X_pca", "X_pca_harmony"],
            }
            config["clustering"]["use_rep"] = config["embeddings"]["use_rep"] = "X_pca_harmony"
    if mode == "discovery":
        enabled.update(("query_projection", "qc_finalization", "composition"))
        config["query_projection"] = {"label_column": celltype, "use_rep": "X_pca"}
        config.setdefault("composition", {})["cell_type_key"] = celltype
        expected["composition"] = ["results/composition_overall.csv"]
    if contrast and selected & {"differences", "communication"}:
        profile("publication")
        config["r"]["enabled"] = True
        enabled.update(
            ("differential_expression", "composition", "differential_abundance", "de_viz")
        )
        config["differential_expression"].update(layer=counts, cell_type_col=celltype)
        config["de_viz"]["methods"][1]["cell_type_col"] = celltype
        expected["differential_expression"] = ["results/de_pseudobulk_edger_by_subcluster.csv"]
        expected["differential_abundance"] = ["results/da_proportion_ttest.csv"]
    organism = config["project"]["organism"]
    if selected & {"communication", "regulation"}:
        resource_dir = (
            Path(
                typer.prompt(
                    "Reference resource directory (nichenet/ and scenic/)", default="resources"
                )
            )
            .expanduser()
            .resolve()
        )
    if "communication" in selected:
        if not sample:
            if mode == "questions":
                raise ValueError(
                    "Communication needs a library ID. Choose its column during setup."
                )
            actual.discard("communication")
            typer.echo("Communication omitted because per-library sample IDs were not supplied.")
        else:
            profile("communication")
            enabled.update(("cell_cell_communication", "network_analysis", "ccc_viz"))
            config["ccc_network"] = {"methods": [{"method": "topology"}]}
            ccc = config["cell_cell_communication"]
            ccc.update(
                cell_type_col=celltype,
                sample_col=sample,
                layer=layer,
                nichenet_counts_layer=counts,
                tf_optimization="auto",
                max_decomposition_cost=100000000,
            )
            paths = ["cell_cell_communication/liana_ranks.csv"]
            if obs[sample].nunique() >= 3:
                paths.append("cell_cell_communication/tensor_factors_contexts.csv")
            else:
                method_omissions["tensor_c2c"] = (
                    "Tensor-cell2cell needs at least three libraries. "
                    f"This study supplied {obs[sample].nunique()}."
                )
                ccc["methods"] = [m for m in ccc["methods"] if m["method"] != "tensor_c2c"]
            if contrast:
                receiver = None
                if celltype in obs:
                    receiver = _choose(
                        "Receiver population for NicheNet",
                        sorted(obs[celltype].dropna().astype(str).unique()),
                    )
                else:
                    method_omissions["nichenet"] = (
                        "Select a named receiver with differential-expression results."
                    )
                    ccc["methods"] = [m for m in ccc["methods"] if m["method"] != "nichenet"]
                    typer.echo(
                        "NicheNet needs a named receiver and its differential-expression "
                        "table. Choose a receiver after reviewing the clusters."
                    )

                suffix = "_mouse" if organism == "mouse" else ""
                ccc.update(
                    nichenet_sender="all",
                    nichenet_receiver=receiver,
                    nichenet_de_cell_type=receiver,
                    nichenet_ligand_target_matrix=str(
                        resource_dir / "nichenet" / f"ligand_target_matrix_nsga2r_final{suffix}.rds"
                    ),
                    nichenet_lr_network=str(
                        resource_dir / "nichenet" / f"lr_network_{organism}_21122021.rds"
                    ),
                    nichenet_weighted_networks=str(
                        resource_dir / "nichenet" / f"weighted_networks_nsga2r_final{suffix}.rds"
                    ),
                )
                if celltype in obs and config["design"]["paired"]:
                    choices = sorted(obs[celltype].dropna().astype(str).unique())
                    selection = typer.prompt(
                        "Populations required in each paired MultiNicheNet "
                        "library (comma-separated, or Enter for all)",
                        default="",
                        show_default=False,
                    )
                    required = [key.strip() for key in selection.split(",") if key.strip()]
                    if set(required) - set(choices):
                        raise ValueError(f"Choose MultiNicheNet populations from {choices}.")
                    ccc["mnn_cell_types"] = required
                paths.append("mnn_sample_prioritization.csv")
                if receiver:
                    paths.append("nichenet_activities.csv")
            else:
                method_omissions["nichenet"] = (
                    "No replicated contrast for receiver differential expression was declared."
                )
                method_omissions["multinichenet"] = "No replicated condition contrast was declared."
                ccc["methods"] = [
                    m for m in ccc["methods"] if m["method"] in ("liana", "tensor_c2c")
                ]
            viz = {
                "liana": "dotplot_viz",
                "tensor_c2c": "tensor_factors_viz",
                "nichenet": "nichenet_viz",
                "multinichenet": "multinichenet_viz",
            }
            config["ccc_viz"]["methods"] = [{"method": viz[m["method"]]} for m in ccc["methods"]]
            config["ccc_viz"]["methods"].extend(
                {"method": m} for m in ("chord_viz", "sankey_viz", "summary_viz")
            )
            config["ccc_viz"]["figure_formats"] = ["pdf", "svg", "png"]
            expected["ccc_viz"] = ["figures/ccc/chord_liana.pdf"]
            expected["cell_cell_communication"] = [f"results/{p}" for p in paths]
    if "regulation" in selected:
        profile("regulatory")
        species = "hg38" if organism == "human" else "mm10"
        tfs = "allTFs_hg38.txt" if organism == "human" else "allTFs_mm.txt"
        motifs = (
            "motifs-v10nr_clust-nr.hgnc.tbl"
            if organism == "human"
            else "motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl"
        )
        scenic = resource_dir / "scenic"
        config["grn"].update(
            layer=counts,
            group_by=celltype,
            organism=organism,
            tfs_path=str(scenic / tfs),
            motifs_path=str(scenic / motifs),
            rankings_glob=str(
                scenic
                / (
                    f"{species}_500bp_up_100bp_down_full_tx_"
                    "v10_clust.genes_vs_motifs.rankings.feather"
                )
            ),
        )
        config["enrichment"].update(layer=layer, cell_type_col=celltype)
        enabled.update(("enrichment", "grn", "coexpression"))
        config["coexpression"] = {"group_by": celltype, "layer": counts}
        if contrast and config["design"]["paired"] and sample and obs[sample].nunique() >= 4:
            config["r"]["enabled"] = True
            enabled.add("multicellular_programs")
            config["multicellular_programs"] = {
                "cell_type_col": celltype,
                "sample_col": sample,
                "layer": layer,
            }
        expected["grn"] = ["results/grn/scenic_auc_mtx_grn.parquet"]
        expected["enrichment"] = ["results/enrichment_activity_collectri.csv"]
    if "differences" in actual:
        enabled.add("enrichment")
        methods = config.setdefault("enrichment", {}).setdefault("methods", [])
        activity = next((entry for entry in methods if entry["method"] == "activity"), None)
        if activity is None:
            methods.append(
                {"method": "activity", "activity_resources": ["progeny"], "gmt_path": None}
            )
        else:
            activity["activity_resources"].append("progeny")
        gmt = typer.prompt(
            "Gene sets for ranked pathway enrichment (.gmt), or Enter to omit",
            default="",
            show_default=False,
        )
        if gmt:
            methods.extend(
                [
                    {"method": method, "gmt_path": gmt, "gene_set_collections": [Path(gmt).stem]}
                    for method in ("gsea", "gsea_by_cell_type")
                ]
            )
        config["enrichment"].update(layer=layer, cell_type_col=celltype)
        expected.setdefault("enrichment", []).append("results/enrichment_activity_progeny.csv")
        if "populations" in selected:
            enabled.add("differential_state")
            config["differential_state"] = {
                "paired": config["design"]["paired"],
                "obs_columns": [
                    "state_" + name for name in config["state_scoring"]["builtin_programs"]
                ],
            }
    if "enrichment" in enabled:
        enabled.add("enrichment_viz")
        methods = config["enrichment"]["methods"]
        activity = next(entry for entry in methods if entry["method"] == "activity")
        viz_methods = [{"method": "activity_viz", "resources": activity["activity_resources"]}]
        if any(entry["method"] == "gsea" for entry in methods):
            viz_methods.extend({"method": m} for m in ("gsea_viz", "gsea_nes_heatmap"))
        config["enrichment_viz"] = {"methods": viz_methods}
        expected["enrichment_viz"] = [
            f"figures/enrichment/activity_dotplot_{resource}.pdf"
            for resource in activity["activity_resources"]
        ]
        for entry in methods:
            if entry["method"] == "gsea_by_cell_type":
                expected["enrichment_viz"].extend(
                    f"figures/enrichment/gsea_nes_heatmap_{collection}.pdf"
                    for collection in entry["gene_set_collections"]
                )
    if "lineage" in selected:
        key = _choose(
            "Population or stage column defining the lineage root, or Enter to omit lineage",
            ["", *cols],
            "",
        )
        if key:
            profile("trajectory")
            group = _choose(
                "Root population in this lineage", sorted(obs[key].dropna().astype(str).unique())
            )
            config["run"]["lineage_rationale"] = typer.prompt("Biological basis for that root")
            config["trajectory"]["dpt"] = {"root_key": key, "root_group": group}
            enabled.update(("trajectory", "trajectory_viz"))
            config["trajectory_viz"] = {"methods": [{"method": "pseudotime_viz"}]}
            generation = None
            if (
                not has_splice
                and manifest is not None
                and {"bam_path", "gtf_path", "cellranger_path"} <= set(manifest)
            ):
                if (
                    manifest[["bam_path", "gtf_path", "cellranger_path"]].notna().all().all()
                    and manifest.gtf_path.nunique() == 1
                ):
                    if typer.confirm(
                        "Generate missing splice counts from the Cell "
                        "Ranger BAMs using this matching gene annotation?",
                        default=False,
                    ):
                        generation = {
                            "generate_missing": True,
                            "gtf_path": str(manifest.gtf_path.iloc[0]),
                        }
            if has_splice or generation:
                config["trajectory"]["methods"].append({"method": "velocity"})
                config["trajectory"]["velocity"] = {
                    "grouping_col": key,
                    "sample_col": sample or "sample_id",
                }
                if generation:
                    config["trajectory"]["velocity"]["generation"] = generation
        else:
            actual.discard("lineage")
    if "perturbation" in selected:
        targets = [
            x.strip()
            for x in typer.prompt(
                "TFs for simulated perturbations, comma-separated, or Enter to omit",
                default="",
                show_default=False,
            ).split(",")
            if x.strip()
        ]
        if targets:
            profile("perturbation")
            config["perturbation"].update(
                tf_list=targets, organism=organism, cluster_key=celltype, layer=counts
            )
            enabled.add("perturbation")
        else:
            actual.discard("perturbation")
    if contrast and "composition" in enabled:
        config["run"]["sensitivity_analysis"] = typer.confirm(
            "Measure how each donor affects the abundance effect and its rank?", default=False
        )
    if not actual:
        raise ValueError(
            "None of the requested questions has the required "
            "inputs. Choose populations or supply missing metadata."
        )
    reasons = {
        "lineage": "Supply a lineage root and its biological rationale.",
        "perturbation": "Supply one or more transcription factors for a simulated perturbation.",
        "differences": "Supply an eligible replicated contrast.",
        "communication": "Supply library IDs for sample-specific communication estimates.",
    }
    config["run"].update(
        study_questions=sorted(actual),
        required_outputs=expected,
        method_omissions=method_omissions,
        question_omissions={q: reasons[q] for q in sorted(selected - actual)},
    )
