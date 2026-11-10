"""Build manuscript figures from the current run's registered numerical results."""

import ast
import shutil
from dataclasses import dataclass
from functools import partial
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.artifacts import ArtifactManager, _atomic_write
from cellquorum.core.stage import StageExecutionRecord
from cellquorum.visualization.figstyle import apply_theme, atomic_savefig


@dataclass
class Panel:
    key: str
    title: str
    unit: str
    method: str
    legend: str
    sources: list[str]
    values: pd.DataFrame


def _paired_values(
    values: pd.DataFrame, meta: pd.DataFrame, config: CellQuorumConfig, sample: str = "sample"
) -> pd.DataFrame:
    design = config.design
    columns = [config.cohort.sample_key, design.donor_col, design.condition_col]
    if len(set(columns)) != 3 or any(col not in meta for col in columns):
        return values.assign(group=values[sample].astype(str))
    ids = (
        meta[columns]
        .drop_duplicates()
        .rename(columns=dict(zip(columns, ["sample", "donor", "group"], strict=True)))
    )
    return values.merge(ids, left_on=sample, right_on="sample", validate="many_to_one")


def _communication_fallback(files: dict[str, Path], panels: list[Panel]) -> list[Panel]:
    # Each method remains inspectable on studies without a shared receiver-edge contrast.
    for method, filename, group, score in (
        ("LIANA", "liana_ranks.csv", "ligand_complex", "magnitude_rank"),
        ("NicheNet", "nichenet_activities.csv", "test_ligand", "aupr_corrected"),
        ("MultiNicheNet", "mnn_prioritization.csv", "ligand", "prioritization_score"),
    ):
        if filename not in files:
            continue
        rows = pd.read_csv(files[filename])
        values = (
            rows.groupby(group)[score]
            .median()
            .reset_index()
            .sort_values(score, ascending=(method == "LIANA"))
            .head(8)
            .rename(columns={group: "group", score: "value"})
        )
        if not values.empty:
            panels.append(
                Panel(
                    "communication." + method.lower(),
                    f"{method} ligand rankings",
                    score,
                    method,
                    "Median across available scores for each ligand. "
                    "This panel uses this method's rankings independently.",
                    [filename],
                    values,
                )
            )
    return panels


def _communication(
    files: dict[str, Path], obs: pd.DataFrame, config: CellQuorumConfig
) -> list[Panel]:
    panels = []
    name = "tensor_factors_contexts.csv"
    if name in files:
        df = pd.read_csv(files[name], index_col=0)
        vals = df.iloc[:, 0].rename("value").rename_axis("sample").reset_index()
        panels.append(
            Panel(
                "communication.tensor",
                "Communication factor 1",
                "Factor loading",
                "Tensor-cell2cell",
                "First fitted context factor. Factor indices are specific to this fit.",
                [name],
                _paired_values(vals, obs, config),
            )
        )
    names = [
        "mnn_prioritization.csv",
        "liana_ranks.csv",
        "nichenet_activities.csv",
        "nichenet_ligand_receptor_weights.csv",
        "nichenet_ligand_target_weights.csv",
        "mnn_sample_prioritization.csv",
    ]
    if not all(n in files for n in names):
        return _communication_fallback(files, panels)
    priorities, liana, activities, receptor_weights, target_weights, sample_scores = [
        pd.read_csv(files[n]) for n in names
    ]
    settings = config.cell_cell_communication
    priorities = priorities.loc[
        (priorities.receiver == settings.nichenet_receiver)
        & (priorities.group == config.design.case)
    ]
    if settings.nichenet_sender not in ("all", "*"):
        senders = (
            [settings.nichenet_sender]
            if isinstance(settings.nichenet_sender, str)
            else settings.nichenet_sender
        )
        priorities = priorities[priorities.sender.isin(senders or [])]
    # LIANA 2 exports component genes alongside the complex identifiers.
    # The join and plotted rank both describe the complete interaction.
    liana = liana[
        ["sample", "source", "target", "ligand_complex", "receptor_complex", "magnitude_rank"]
    ].rename(
        columns={
            "source": "sender",
            "target": "receiver",
            "ligand_complex": "ligand",
            "receptor_complex": "receptor",
        }
    )
    keys = ["sender", "receiver", "ligand", "receptor"]
    candidates = (
        priorities.merge(liana[keys].drop_duplicates(), on=keys)
        .merge(receptor_weights, on=["ligand", "receptor"])
        .merge(activities, left_on="ligand", right_on="test_ligand")
    )
    if candidates.empty:
        return _communication_fallback(files, panels)
    edge = candidates.sort_values(
        ["rank", "prioritization_score", *keys], ascending=[True, False, True, True, True, True]
    ).iloc[0]
    label = f"{edge.ligand} → {edge.receptor}"
    legend = (
        f"{edge.sender} to {edge.receiver}. Selected among shared edges by NicheNet ligand rank, "
        "then MultiNicheNet case prioritization. These are expression-based predictions."
    )
    for df, key, title, unit, column, method in (
        (liana, "liana", label, "Magnitude rank (lower ranks first)", "magnitude_rank", "LIANA"),
        (
            sample_scores,
            "mnn",
            label,
            "Scaled pseudobulk LR product",
            "scaled_LR_pb_prod",
            "MultiNicheNet",
        ),
    ):
        rows = df.loc[(df[keys] == edge[keys]).all(axis=1), ["sample", *keys, column]].rename(
            columns={column: "value"}
        )
        if not rows.empty:
            panels.append(
                Panel(
                    "communication." + key,
                    title,
                    unit,
                    method,
                    legend,
                    names,
                    _paired_values(rows, obs, config),
                )
            )
    top = (
        target_weights[target_weights.ligand == edge.ligand]
        .nlargest(6, "weight")
        .rename(columns={"target": "group", "weight": "value"})
    )
    if not top.empty:
        panels.append(
            Panel(
                "communication.targets",
                f"{edge.ligand} receiver targets",
                "Prior target weight",
                "NicheNet",
                legend + " Top retained target links, ordered by prior weight.",
                names,
                top,
            )
        )
    return panels


def _regulation(files: dict[str, Path], obs: pd.DataFrame, config: CellQuorumConfig) -> list[Panel]:
    col = config.cell_cell_communication.nichenet_receiver
    condition = config.design.condition_col
    donor = config.design.donor_col
    name = "enrichment_activity_collectri_by_donor.csv"
    if name not in files:
        name = "enrichment_activity_collectri.csv"
        if name not in files:
            return []
        frame = pd.read_csv(files[name], dtype={"cell_type": str})
        if frame.empty:
            return []
        population = (
            col
            if col is not None and col in set(frame.cell_type)
            else sorted(frame.cell_type.unique())[0]
        )
        subset = frame.loc[frame.cell_type.eq(population)].groupby("source").mean_score.mean()
        selected = (
            subset.loc[subset.abs().nlargest(8).index]
            .rename("value")
            .rename_axis("group")
            .reset_index()
        )
        frames = [
            Panel(
                "regulation.collectri",
                f"TF activity in {population}",
                "Mean ULM score",
                "CollecTRI",
                "Descriptive scores for the largest absolute TF activities in "
                "this population. Biological replicate IDs were unavailable.",
                [name],
                selected,
            )
        ]
        auc_name = "scenic_auc_mtx_grn.parquet"
        group_key = config.grn.group_by or config.enrichment.cell_type_col
        if auc_name in files and group_key in obs:
            auc = pd.read_parquet(files[auc_name])
            auc.columns = auc.columns.astype(str).str.removesuffix("(+)")
            shared_scores = subset[subset.index.isin(auc.columns)].dropna()
            if not shared_scores.empty:
                tf = str(shared_scores.abs().idxmax())
                cells = (
                    auc[tf]
                    .rename("activity")
                    .to_frame()
                    .join(obs[group_key].rename("group"), how="inner")
                    .dropna()
                )
                values = (
                    cells.groupby("group", observed=True)
                    .activity.agg(value="mean", n_cells="size")
                    .reset_index()
                )
                if not values.empty:
                    frames.append(
                        Panel(
                            "regulation.scenic",
                            f"{tf} regulon by population",
                            "Mean regulon AUC",
                            "pySCENIC",
                            f"Mean AUCell by population. Among shared TFs, {tf} had the largest "
                            f"absolute CollecTRI score in {population}. Each bar summarizes cells.",
                            [auc_name, name],
                            values.assign(source=tf),
                        )
                    )
                    _append_regulon_targets(frames, files, tf)
        return frames
    ct = pd.read_csv(files[name], dtype={donor: str, "cell_type": str})
    if col not in set(ct.cell_type):
        col = sorted(ct.cell_type.unique())[0]
    ct = ct[ct.cell_type == col]
    if condition not in ct:
        return []
    cohort = ct.groupby("source").mean_score.var().dropna().sort_values(ascending=False)
    auc_name = "scenic_auc_mtx_grn.parquet"
    frames = []
    shared = list(cohort.index)
    if auc_name in files:
        auc = pd.read_parquet(files[auc_name])
        auc.columns = auc.columns.astype(str).str.removesuffix("(+)")
        # Prefer activating regulons for a shared TF, keeping original negative regulons separate.
        shared = [tf for tf in shared if tf in auc.columns]
        if not shared:
            return []
        tf = shared[0]
        group = config.grn.group_by or config.enrichment.cell_type_col
        meta = obs.loc[obs[group].astype(str) == col, [donor, condition]]
        joined = meta.join(auc[tf].rename("activity"), how="inner").dropna()
        vals = (
            joined.groupby([donor, condition], observed=True)
            .activity.agg(value="mean", n_cells="size")
            .reset_index()
        )
        vals = vals.rename(columns={donor: "donor", condition: "group"}).assign(
            source=tf, cell_type=col
        )
        if not vals.empty:
            frames.append(
                Panel(
                    "regulation.scenic",
                    f"{tf} regulon in {col}",
                    "Mean regulon AUC",
                    "pySCENIC",
                    "Donor means across scored cells in the fitted "
                    "regulon. AUCell and ULM use different scales.",
                    [auc_name, name],
                    vals,
                )
            )
    if not shared:
        return []
    tf = shared[0]
    vals = ct[ct.source == tf].rename(
        columns={donor: "donor", condition: "group", "mean_score": "value"}
    )
    frames.insert(
        0,
        Panel(
            "regulation.collectri",
            f"{tf} activity in {col}",
            "Mean ULM score",
            "CollecTRI",
            "TF selected by variability of donor- and condition-specific mean ULM scores among "
            "available TFs shared by the displayed methods. Points are biological replicates.",
            [name],
            vals,
        ),
    )
    _append_regulon_targets(frames, files, tf)
    return frames


def _append_regulon_targets(frames: list[Panel], files: dict[str, Path], tf: str) -> None:
    regulons = "scenic_regulons_grn.csv"
    if regulons in files and any(p.key == "regulation.scenic" for p in frames):
        motifs = pd.read_csv(files[regulons], header=[0, 1], index_col=[0, 1])
        if tf in motifs.index.get_level_values(0):
            entries = motifs[("Enrichment", "TargetGenes")].xs(tf, level=0)
            rows = [
                (tf, motif, gene, float(weight))
                for motif, targets in entries.items()
                for gene, weight in ast.literal_eval(targets)
            ]
            links = pd.DataFrame(rows, columns=["source", "motif", "group", "value"])
            links = (
                links.sort_values(["value", "group"], ascending=[False, True])
                .drop_duplicates("group")
                .head(8)
            )
            if not links.empty:
                frames.append(
                    Panel(
                        "regulation.targets",
                        f"{tf} predicted targets",
                        "Regulon target weight",
                        "pySCENIC",
                        "Highest weighted targets in the motif-pruned regulon. "
                        "For genes shared by motifs, the maximum weight is shown.",
                        [regulons],
                        links,
                    )
                )


def abundance_sensitivity(proportions: pd.DataFrame, config: CellQuorumConfig) -> pd.DataFrame:
    """Refit the abundance effect after dropping each donor."""
    case, control = config.design.case, config.design.control
    if case is None or control is None:
        raise ValueError("Donor sensitivity requires a case and control condition.")
    df = proportions[proportions.condition.isin([case, control])]
    rows = []
    for celltype, group in df.groupby("cell_type", sort=True):
        if group.duplicated(["donor", "condition"]).any():
            if "count" not in df:
                raise ValueError(
                    "Library counts are needed to pool repeated libraries within each donor."
                )
            totals = df.groupby(["donor", "condition"], observed=True)["count"].sum()
            pooled = group.groupby(["donor", "condition"], observed=True)["count"].sum()
            group = (pooled / totals.reindex(pooled.index)).rename("proportion").reset_index()
        pivot = group.pivot(index="donor", columns="condition", values="proportion").reindex(
            columns=[case, control]
        )
        if config.design.paired:
            pivot = pivot.dropna()
        full = pivot.mean()
        for donor in pivot.index:
            remaining = pivot.drop(donor)
            if remaining.count().min() < 2:
                continue
            mean = remaining.mean()
            rows.append(
                (
                    celltype,
                    donor,
                    100 * (full[case] - full[control]),
                    100 * (mean[case] - mean[control]),
                )
            )
    result = pd.DataFrame(
        rows, columns=["cell_type", "omitted_donor", "full_effect_pp", "effect_pp"]
    )
    result["same_direction"] = np.sign(result.effect_pp) == np.sign(result.full_effect_pp)
    result["full_rank"] = result.groupby("omitted_donor").full_effect_pp.rank(
        method="min", ascending=False
    )
    result["rank"] = result.groupby("omitted_donor").effect_pp.rank(method="min", ascending=False)
    return result


def _draw(ax: Axes, panel: Panel, config: CellQuorumConfig) -> None:
    data = panel.values
    groups = data.group.drop_duplicates().tolist()
    if "donor" in data and {config.design.case, config.design.control} == set(groups):
        groups = [config.design.control, config.design.case]
    positions = {g: i for i, g in enumerate(groups)}
    if "donor" in data:
        if config.design.paired:
            for _, frame in data.groupby("donor", observed=True):
                if frame.group.is_unique and len(frame) > 1:
                    x = frame.group.map(positions)
                    ax.plot(x, frame.value, color="#9DA5A8", alpha=0.65, linewidth=0.8, zorder=1)
        for g, frame in data.groupby("group", observed=True):
            ax.scatter(
                [positions[g]] * len(frame),
                frame.value,
                s=18,
                color="#A14D45" if g == config.design.case else "#295478",
                zorder=2,
            )
        labels = [f"{g}\n(n={data.loc[data.group == g, 'donor'].nunique()})" for g in groups]
        ax.set_xticks(range(len(groups)), labels, rotation=30 if len(groups) > 3 else 0)
        ax.set_ylabel(panel.unit)
    else:
        ax.barh(data.group.astype(str), data.value, color="#356C79")
        if {"bootstrap_ci_low_pp", "bootstrap_ci_high_pp"} <= set(data):
            ax.hlines(
                data.group.astype(str),
                data.bootstrap_ci_low_pp,
                data.bootstrap_ci_high_pp,
                color="#232E31",
                lw=1,
            )
        ax.invert_yaxis()
        ax.set_xlabel(panel.unit)
    ax.set_title(panel.title, loc="left", fontsize=10)
    ax.spines[["right", "top"]].set_visible(False)


def _publish_receiver_chord(
    files: dict[str, Path],
    descriptions: dict[str, str],
    manager: ArtifactManager,
    *,
    standalone: bool,
) -> dict | None:
    stem = "liana_receiver_ligand_receptor"
    key = "communication.ligand_receptor"
    formats = ("pdf", "svg", "csv")
    if not all(f"{stem}.{fmt}" in files for fmt in formats):
        return None
    legend = descriptions[f"{stem}.svg"]
    for fmt in formats:
        source = files[f"{stem}.{fmt}"]
        names = [key, "communication"] if standalone and fmt != "csv" else [key]
        for name in names:
            rel = f"reports/publication/{name}.{fmt}"
            _atomic_write(manager.root / rel, partial(shutil.copyfile, source))
            manager.register(name=name, relative_path=rel, kind=fmt, description=legend)
    return {
        "panel": key,
        "question": "communication",
        "method": "LIANA",
        "title": "Ranked ligand–receptor routes",
        "legend": legend,
        "preview": True,
        "source_data": f"reports/publication/{key}.csv",
        "settings": "provenance/resolved_config.json",
        "inputs": [
            str(files[n].relative_to(manager.root))
            for n in ("liana_ranks.csv", f"{stem}.csv")
            if n in files
        ],
    }


def write_publication_bundle(
    config: CellQuorumConfig,
    records: list[StageExecutionRecord],
    manager: ArtifactManager,
    obs: pd.DataFrame | None,
) -> list[dict]:
    """Render only from artifacts recorded by successful stages in this execution."""
    files = {}
    descriptions = {}
    for record in records:
        if record.status != "success" and not record.reused_from_checkpoint:
            continue
        for artifact in record.output_artifacts:
            path = (manager.root / artifact.path).resolve()
            if path.is_relative_to(manager.root) and path.is_file():
                files[path.name] = path
                descriptions[path.name] = artifact.description
    if obs is None:
        obs = pd.DataFrame()
    root = manager.root
    out = root / "reports/publication"
    out.mkdir(exist_ok=True, parents=True)
    panels = []
    wanted = config.run.study_questions
    index: list[dict] = []
    if "communication" in wanted:
        communication = _communication(files, obs, config)
        standalone = all(p.key == "communication.liana" for p in communication)
        native = _publish_receiver_chord(files, descriptions, manager, standalone=standalone)
        if native:
            index.append(native)
        if not (standalone and native):
            panels.extend(communication)
    if "regulation" in wanted:
        panels.extend(_regulation(files, obs, config))
    if "differences" in wanted and "da_proportion_ttest.csv" in files:
        name = "da_proportion_ttest.csv"
        vals = pd.read_csv(files[name]).rename(columns={"cell_type": "group", "effect_pp": "value"})
        vals = vals.sort_values("value", ascending=False).head(12)
        if not vals.empty:
            panels.append(
                Panel(
                    "differences.abundance",
                    "Cell-type abundance",
                    "Difference (percentage points)",
                    "Replicate-level proportion comparison",
                    "Case minus reference. The source table includes "
                    "donor counts, bootstrap intervals and FDR.",
                    [name],
                    vals,
                )
            )
    if "differences" in wanted:
        name = "de_pseudobulk_edger_by_subcluster.csv"
        if name in files:
            de = pd.read_csv(files[name]).dropna(subset=["FDR", "logFC"])
            top = de.sort_values(["FDR", "gene", "cell_type"]).head(8).copy()
            if not top.empty:
                top["group"] = top.gene.astype(str) + " · " + top.cell_type.astype(str)
                panels.append(
                    Panel(
                        "differences.expression",
                        "Differential expression",
                        "Case vs reference log2 fold change",
                        "edgeR",
                        "Top gene and population comparisons ordered by adjusted "
                        "P value. The panel table includes FDR and log fold change.",
                        [name],
                        top.rename(columns={"logFC": "value"}),
                    )
                )
        enrich = sorted(
            n for n in files if n.startswith("enrichment_gsea_") and n.endswith("_by_cell_type.csv")
        )
        if enrich:
            frames = [pd.read_csv(files[n]) for n in enrich]
            top = (
                pd.concat(frames, ignore_index=True)
                .dropna(subset=["score"])
                .sort_values(["padj", "source", "cell_type"])
                .head(6)
                .copy()
            )
            if not top.empty:
                top["group"] = (
                    top.source.astype(str)
                    .str.replace("HALLMARK_", "", regex=False)
                    .str.replace("_", " ", regex=False)
                    .str.title()
                    + " · "
                    + top.cell_type.astype(str)
                )
                panels.append(
                    Panel(
                        "differences.pathways",
                        "Pathway enrichment",
                        "Normalized enrichment score",
                        "GSEA",
                        "Top pathways across tested populations, ordered by adjusted P value.",
                        enrich,
                        top.rename(columns={"score": "value"}),
                    )
                )
    if "perturbation" in wanted and "differential_regulation.csv" in files:
        name = "differential_regulation.csv"
        table = pd.read_csv(files[name])
        top = table.loc[~table.is_knockout_gene].nlargest(8, "distance").copy()
        if not top.empty:
            top.rename(columns={"group": "cell_group", "distance": "value"}, inplace=True)
            top["group"] = top.gene.astype(str) + " · " + top.ko_genes.astype(str)
            panels.append(
                Panel(
                    "perturbation.regulation",
                    "Predicted downstream responses",
                    "Differential regulation distance",
                    "scTenifoldKnk",
                    "Largest unsigned network distances after simulated TF removal. "
                    "Each row identifies the simulated perturbation and gene.",
                    [name],
                    top,
                )
            )
    if "lineage" in wanted:
        for method, key, settings in (
            ("DPT", "dpt_pseudotime", config.trajectory.dpt),
            ("Palantir", "palantir_pseudotime", config.trajectory.palantir),
        ):
            name = key + ".h5ad"
            group = settings.root_key
            if name not in files or group not in obs or key not in obs:
                continue
            values = (
                obs.groupby(group, observed=True)[key]
                .agg(value="median", n_cells="count")
                .reset_index()
            )
            values = values.rename(columns={group: "group"}).sort_values("value")
            panels.append(
                Panel(
                    "lineage." + method.lower(),
                    "Ordering along the selected lineage",
                    "Median inferred pseudotime",
                    method,
                    f"Root group {settings.root_group}. {config.run.lineage_rationale or ''} "
                    "Group medians use cells with finite pseudotime.",
                    [name],
                    values,
                )
            )
    if "populations" in wanted and "population_summary.csv" in files:
        name = "population_summary.csv"
        summary = pd.read_csv(files[name])
        if {"population_id", "n_cells"} <= set(summary):
            values = summary.rename(
                columns={"population_id": "group", "n_cells": "value"}
            ).nlargest(15, "value")
            panels.append(
                Panel(
                    "populations.identity",
                    "Most abundant populations",
                    "Cells",
                    "Population identity",
                    "Cell counts for the largest populations in this analysis.",
                    [name],
                    values,
                )
            )
    if config.run.sensitivity_analysis:
        name = "composition_proportions.csv"
        if name not in files:
            raise ValueError(
                "Donor sensitivity needs the registered composition_proportions.csv output."
            )
        manager.write_dataframe(
            abundance_sensitivity(pd.read_csv(files[name]), config),
            name="abundance_sensitivity",
            relative_path="reports/publication/abundance_sensitivity.csv",
            description="Leave-one-donor-out effects and ranks.",
            index=False,
        )
    for family in dict.fromkeys(p.key.split(".")[0] for p in panels):
        selected = [p for p in panels if p.key.startswith(family + ".")]
        with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            apply_theme("compact")
            plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
            cols = min(2, len(selected))
            rows = (len(selected) + cols - 1) // cols
            fig, axes = plt.subplots(
                rows, cols, figsize=(3.45 * cols, 3.05 * rows), squeeze=False, layout="constrained"
            )
            for unused in axes.flat[len(selected) :]:
                unused.set_visible(False)
            try:
                for letter, (ax, panel) in enumerate(
                    zip(list(axes.flat)[: len(selected)], selected, strict=True)
                ):
                    _draw(ax, panel, config)
                    ax.set_title(
                        chr(65 + letter) + "   " + panel.method + "\n" + panel.title,
                        loc="left",
                        fontsize=10,
                    )
                    single, axis = plt.subplots(figsize=(3.7, 3.2), layout="constrained")
                    try:
                        _draw(axis, panel, config)
                        for fmt in ("svg", "pdf"):
                            rel = f"reports/publication/{panel.key}.{fmt}"
                            atomic_savefig(single, root / rel)
                            manager.register(
                                name=panel.key,
                                relative_path=rel,
                                kind=fmt,
                                description=panel.legend,
                            )
                    finally:
                        plt.close(single)
                    rel = f"reports/publication/{panel.key}.csv"
                    manager.write_dataframe(
                        panel.values,
                        name=panel.key + ".source",
                        relative_path=rel,
                        description=panel.legend,
                        index=False,
                    )
                    index.append(
                        {
                            "panel": panel.key,
                            "question": family,
                            "method": panel.method,
                            "title": panel.title,
                            "legend": panel.legend,
                            "source_data": rel,
                            "settings": "provenance/resolved_config.json",
                            "inputs": [str(files[n].relative_to(root)) for n in panel.sources],
                        }
                    )
                for fmt in ("pdf", "svg"):
                    rel = f"reports/publication/{family}.{fmt}"
                    atomic_savefig(fig, root / rel)
                    manager.register(
                        name=family + "_figure",
                        relative_path=rel,
                        kind=fmt,
                        description=f"Assembled {family} figure.",
                    )
            finally:
                plt.close(fig)
    omissions = dict(config.run.question_omissions)
    for question in wanted:
        if not any(row["question"] == question for row in index):
            omissions[question] = (
                "No manuscript panel could be derived from the recorded outputs. "
                "Inspect the study report for method results and eligibility."
            )
    manager.write_json(
        {"panels": index, "omissions": omissions},
        name="publication_index",
        relative_path="reports/publication/index.json",
        description="Publication panels, source data, legends and settings.",
    )
    references = {
        "LIANA": "Dimitrov et al. (2022). https://doi.org/10.1038/s41467-022-30755-0",
        "Tensor-cell2cell": "Armingol et al. (2022). https://github.com/earmingol/cell2cell",
        "NicheNet": "Browaeys et al. (2020). https://doi.org/10.1038/s41592-019-0667-5",
        "MultiNicheNet": "MultiNicheNet paired analysis. https://github.com/saeyslab/multinichenetr",
        "CollecTRI": "Müller-Dott et al. (2023). https://doi.org/10.1093/nar/gkad841",
        "pySCENIC": "Van de Sande et al. (2020). https://doi.org/10.1038/s41596-020-0336-2",
        "edgeR": "Robinson et al. (2010). https://doi.org/10.1093/bioinformatics/btp616",
        "GSEA": "Subramanian et al. (2005). https://doi.org/10.1073/pnas.0506580102",
        "scTenifoldKnk": "Osorio et al. (2022). https://doi.org/10.1016/j.patter.2022.100434",
        "Palantir": "Setty et al. (2019). https://doi.org/10.1038/s41587-019-0068-4",
        "DPT": "Haghverdi et al. (2016). https://doi.org/10.1038/nmeth.3971",
    }
    citations = [
        references[m] for m in dict.fromkeys(row["method"] for row in index) if m in references
    ]
    manager.write_text(
        "\n\n".join(citations) + "\n",
        name="figure_citations",
        relative_path="reports/publication/citations.md",
        kind="markdown",
        description="References for methods shown in the figure sets.",
    )
    sections = []
    for family in dict.fromkeys(row["question"] for row in index):
        sections.append(
            f'<h2>{escape(family.title())}</h2><a href="{family}.pdf">Assembled PDF</a> · '
            f'<a href="{family}.svg">Editable SVG</a>'
        )
        for row in index:
            if row["question"] == family:
                key = escape(row["panel"], quote=True)
                sections.append(
                    f"<h3>{escape(row['method'])} · {escape(row['title'])}</h3>"
                    f'<img alt="{escape(row["title"], quote=True)}" src="{key}.svg">'
                    f'<p>{escape(row["legend"])}</p><a href="{key}.csv">Exact source data</a> · '
                    f'<a href="{key}.svg">Editable panel</a> · <a href="{key}.pdf">Panel PDF</a>'
                )
    for omitted_question, reason in omissions.items():
        sections.append(f"<h2>{escape(omitted_question.title())}</h2><p>{escape(reason)}</p>")
    if config.run.sensitivity_analysis:
        sections.append('<p><a href="abundance_sensitivity.csv">Donor sensitivity results</a></p>')
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Publication figures</title>'
        "<style>body{max-width:60rem;margin:2rem auto;padding:0 1rem;"
        "font:16px/1.45 system-ui}img{max-width:100%}</style>"
        '<h1>Publication figures</h1><p><a href="../report.html">Study report</a> · '
        '<a href="index.json">Figure index</a> · <a href="citations.md">Citations</a></p>'
        + "".join(sections)
        + "</html>"
    )
    manager.write_text(
        html,
        name="publication_gallery",
        relative_path="reports/publication/index.html",
        kind="html",
        description="Publication figures and their exact source data.",
    )
    return index
