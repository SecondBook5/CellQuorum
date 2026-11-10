"""Check the scientific outputs of an installed CellQuorum reference run."""

import argparse
import ast
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_scc(run: Path, data: Path) -> dict:
    results = run / "results"
    audit = pd.read_csv(results / "mnn_pairing_audit.csv")
    pairs = audit[audit.included_paired_analysis]
    require(
        set(pairs.patient) == {"P2", "P6", "P8", "P9", "P10"} and len(pairs) == 10,
        "Paired MultiNicheNet must use these five complete fibroblast–keratinocyte pairs.",
    )
    samples = pd.read_csv(results / "mnn_sample_prioritization.csv")
    require(set(samples["sample"]) == set(pairs.sample_id), "MultiNicheNet sample support changed.")
    published = {"P6Normal": 12.3, "P10Tumor": 11.8, "P10Normal": 11.7, "P2Tumor": 11.7}
    cd63 = (
        samples.loc[(samples.receiver == "Fibroblast") & (samples.receptor == "CD63")]
        .groupby("sample")
        .pb_receptor.first()
    )
    differences = (cd63.reindex(published) - pd.Series(published)).abs()
    require(
        differences.notna().all() and differences.max() < 0.25,
        "Sample-level CD63 differs from the vignette by more than 0.25 log units.",
    )
    de = pd.read_csv(results / "de_pseudobulk_edger_by_subcluster.csv")
    mmp1 = de.loc[(de.gene == "MMP1") & (de.cell_type == "Fibroblast")].iloc[0]
    require(
        mmp1.logFC > 0 and mmp1.FDR < 0.05,
        "Published direction of the fibroblast MMP1 response was lost.",
    )
    liana = pd.read_csv(results / "cell_cell_communication/liana_ranks.csv")
    tensor = pd.read_csv(
        results / "cell_cell_communication/tensor_factors_contexts.csv", index_col=0
    )
    require(
        set(tensor.index) == set(liana["sample"]) and tensor.shape[1] == 3,
        "Tensor contexts must match scored LIANA libraries.",
    )
    auc = pd.read_parquet(results / "grn/scenic_auc_mtx_grn.parquet")
    require(
        len(auc) == 1000 and auc.shape[1] > 0 and np.isfinite(auc.to_numpy()).all(),
        "SCENIC must score the declared 1,000-cell subsample.",
    )
    tf = pd.read_csv(results / "enrichment_activity_collectri_by_donor.csv")
    require(
        tf.patient.nunique() == 10 and tf.cell_type.nunique() == 4,
        "Unexpected CollecTRI donor or population coverage.",
    )
    mnn_panel = pd.read_csv(run / "reports/publication/communication.mnn.csv")
    for row in mnn_panel.itertuples():
        matched = samples.loc[
            (samples["sample"] == row.sample)
            & (samples.sender == row.sender)
            & (samples.receiver == row.receiver)
            & (samples.ligand == row.ligand)
            & (samples.receptor == row.receptor)
        ]
        require(
            len(matched) == 1 and np.isclose(matched.scaled_LR_pb_prod.iloc[0], row.value),
            "MultiNicheNet figure differs from its source.",
        )
    panel = pd.read_csv(run / "reports/publication/regulation.collectri.csv")
    matched = panel.merge(
        tf,
        left_on=["donor", "group", "source", "cell_type"],
        right_on=["patient", "tum.norm", "source", "cell_type"],
        validate="one_to_one",
    )
    require(
        len(matched) == len(panel) and np.allclose(matched.value, matched.mean_score),
        "TF panel differs from donor means.",
    )
    targets = pd.read_csv(results / "nichenet_ligand_target_weights.csv")
    target_panel = pd.read_csv(run / "reports/publication/communication.targets.csv")
    joined = target_panel.merge(
        targets, left_on=["ligand", "group"], right_on=["ligand", "target"], validate="one_to_one"
    )
    require(
        len(joined) == len(target_panel) and np.allclose(joined.value, joined.weight),
        "NicheNet target panel differs from prior weights.",
    )
    gallery = run / "reports/publication"
    factor = pd.read_csv(gallery / "communication.tensor.csv").set_index("sample")
    require(
        set(factor.index) == set(tensor.index)
        and np.allclose(factor.value, tensor.iloc[:, 0].loc[factor.index]),
        "Tensor panel differs from the estimated factor.",
    )
    ranks = pd.read_csv(gallery / "communication.liana.csv").merge(
        liana,
        left_on=["sample", "sender", "receiver", "ligand", "receptor"],
        right_on=["sample", "source", "target", "ligand_complex", "receptor_complex"],
        validate="one_to_one",
        how="left",
    )
    require(np.allclose(ranks.value, ranks.magnitude_rank), "LIANA panel differs from ranks.")
    scenic = pd.read_csv(gallery / "regulation.scenic.csv")
    cells = ad.read_h5ad(run / "objects/final_annotated.h5ad", backed="r")
    try:
        meta = cells.obs.loc[
            cells.obs.celltype_alt == scenic.cell_type.iat[0], ["patient", "tum.norm"]
        ]
    finally:
        cells.file.close()
    scores = meta.join(auc[scenic.source.iat[0] + "(+)"].rename("auc"), how="inner")
    means = scores.groupby(["patient", "tum.norm"], observed=True).auc.agg(["mean", "count"])
    expected = means.loc[pd.MultiIndex.from_frame(scenic[["donor", "group"]])]
    require(
        np.allclose(expected["mean"], scenic.value)
        and np.array_equal(expected["count"], scenic.n_cells),
        "AUCell panel differs from donor-level means or cell counts.",
    )
    motifs = pd.read_csv(results / "grn/scenic_regulons_grn.csv", index_col=[0, 1], header=[0, 1])
    inferred = pd.read_csv(gallery / "regulation.targets.csv")
    for row in inferred.itertuples():
        contexts = motifs.loc[[(row.source, row.motif)], ("Enrichment", "TargetGenes")]
        weights = [dict(ast.literal_eval(v)).get(row.group, -np.inf) for v in contexts]
        require(np.isclose(max(weights), row.value), "Motif target weight differs.")
    manifest = json.loads((gallery / "index.json").read_text())
    require(len(manifest["panels"]) == 7, "Expected all seven publication panels.")
    for panel in manifest["panels"]:
        paths = [run / f for f in [panel["source_data"], panel["settings"], *panel["inputs"]]]
        paths.extend(gallery / (panel["panel"] + ext) for ext in [".svg", ".pdf"])
        require(
            panel["legend"] and all(f.is_file() and f.stat().st_size for f in paths),
            f"Missing figure evidence for {panel['panel']}",
        )
    return {
        "patients": int(audit.patient.nunique()),
        "mnn_pairs": len(pairs) // 2,
        "liana_libraries": len(tensor),
        "regulons": auc.shape[1],
        "scenic_cells": len(auc),
        "mmp1_log2fc": float(mmp1.logFC),
        "published_cd63_absolute_differences": differences.to_dict(),
    }


def check_lineage(run: Path, data: Path) -> dict:
    obj = ad.read_h5ad(run / "objects/final_annotated.h5ad", backed="r")
    try:
        time = obj.obs.groupby("clusters", observed=True).dpt_pseudotime.median()
        stages = ["Ductal", "Ngn3 high EP", "Pre-endocrine", "Beta"]
        require(
            time.loc[stages].is_monotonic_increasing and time["Alpha"] > time["Pre-endocrine"],
            "Published progenitor-to-endocrine ordering was lost.",
        )
        scores = obj.obs.dpt_pseudotime
        n_missing = int(scores.isna().sum())
        require(
            not np.isinf(scores).any() and np.isfinite(scores).mean() > 0.99,
            "Unreachable cells must be marked missing and at least "
            "99% of this reference must have finite pseudotime.",
        )
        require(
            int(obj.uns["trajectory"]["dpt"]["n_unordered"]) == n_missing,
            "Unordered cells missing from provenance.",
        )
        if "qc_splice_intronic_fraction" in obj.obs:
            totals = [
                np.asarray(obj.layers[name].sum(axis=1)).ravel()
                for name in ("spliced", "unspliced")
            ]
            denominator = totals[0] + totals[1]
            expected_fraction = np.divide(
                totals[1], denominator, out=np.full(len(scores), np.nan), where=denominator > 0
            )
            require(
                np.allclose(obj.obs.qc_splice_intronic_fraction, expected_fraction, equal_nan=True),
                "Splice QC differs from the count layers.",
            )
    finally:
        obj.file.close()
    velocity = ad.read_h5ad(run / "results/trajectory/velocity/whole_object.h5ad", backed="r")
    try:
        require(
            "velocity" in velocity.layers and velocity.n_obs == len(obj.obs),
            "Missing whole-lineage velocity.",
        )
    finally:
        velocity.file.close()
    return {
        "cells": len(obj.obs),
        "unordered_cells": n_missing,
        "median_pseudotime": time.to_dict(),
    }


def check_fates(run: Path, data: Path) -> dict:
    obj = ad.read_h5ad(run / "objects/final_annotated.h5ad")
    medians = obj.obs.groupby("clusters", observed=True).palantir_pseudotime.median()
    require(
        medians["Ductal"] < medians["Ngn3 high EP"] < medians["Beta"],
        "Inspect Palantir root ordering.",
    )
    dimensions = {}
    for method in ("palantir", "cellrank"):
        prob = np.asarray(obj.obsm[method + "_fate_probabilities"])
        require(
            np.isfinite(prob).all()
            and (prob >= -1e-6).all()
            and np.allclose(prob.sum(axis=1), 1, atol=1e-4),
            "Invalid fate probabilities.",
        )
        dimensions[method] = prob.shape[1]
    return {
        "cells": obj.n_obs,
        "destinations": dimensions,
        "median_pseudotime": medians.to_dict(),
        "interpretation": (
            "Palantir recovered one fate. CellRank included ductal and progenitor states."
        ),
    }


def check_atlas(run: Path, data: Path) -> dict:
    truth = pd.read_csv(data / "held_out_labels.csv", index_col=0).cell_type
    results = pd.read_csv(run / "results/transferred_type_assignments.csv", index_col=0)
    pred = results.transferred_type
    require(set(pred.index) == set(truth.index), "Missing query cell predictions.")
    correct = pred == truth.loc[pred.index]
    recall = correct.groupby(truth.loc[pred.index]).mean()
    require(
        correct.mean() > truth.value_counts(normalize=True).max(),
        "Label transfer failed to beat majority-class prediction.",
    )
    diagnostic_file = run / "results/annotation_diagnostics/scdiagnostics_results.csv"
    if diagnostic_file.exists():
        diagnostics = pd.read_csv(diagnostic_file, index_col="barcode")
        require(
            set(diagnostics.index) == set(truth.index)
            and np.isfinite(diagnostics.scdiag_entropy).all()
            and diagnostics.scdiag_entropy.ge(0).all()
            and diagnostics.scdiag_knn_prob_group.dropna().between(0, 1).all()
            and diagnostics.scdiag_entropy.notna().mean() > 0.9,
            "Missing or invalid held-out annotation diagnostics.",
        )
    return {
        "query_cells": len(pred),
        "accuracy": float(correct.mean()),
        "macro_recall": float(recall.mean()),
        "recall_by_type": recall.to_dict(),
    }


def check_knockout(run: Path, data: Path) -> dict:
    table = pd.read_csv(run / "results/sctenifoldknk/differential_regulation.csv")
    table = table[~table.is_knockout_gene].set_index("gene")
    d = ad.read_h5ad(data / "held_out.h5ad")
    mat = d.X.tocsr().astype(float)
    total = np.asarray(mat.sum(axis=1)).ravel()
    mat = mat.multiply(10000 / np.maximum(total, 1)[:, None]).tocsr()
    effects = []
    for _, ids in d.obs.groupby("batch", observed=True).indices.items():
        arm = d.obs.perturbation.iloc[ids].to_numpy()
        if not {"GATA1", "control"} <= set(arm):
            continue
        treated = np.asarray(mat[ids[arm == "GATA1"]].mean(axis=0)).ravel()
        control = np.asarray(mat[ids[arm == "control"]].mean(axis=0)).ravel()
        effects.append(np.log2((treated + 1) / (control + 1)))
    require(
        bool(effects),
        "Held-out evaluation needs measured GATA1 effects and batch-matched controls.",
    )
    measured = pd.Series(np.mean(effects, axis=0), index=d.var_names, name="measured_log2fc")
    joined = table.join(measured, how="inner")
    rho = spearmanr(joined.distance, joined.measured_log2fc.abs()).statistic
    require(np.isfinite(rho) and len(joined) > 20, "Missing comparable gene rankings.")
    joined.to_csv(run / "reports/held_out_perturbation.csv")
    return {
        "evaluation_cells": len(d),
        "genes": len(joined),
        "TF": "GATA1",
        "spearman_abs_effect": float(rho),
        "positive_rank_association": bool(rho > 0),
        "measured_GATA1_log2fc": float(measured["GATA1"]),
    }


def check_nuclei(run: Path, data: Path) -> dict:
    pairs = pd.read_csv(run / "results/qc/qc_donor_comparisons.csv")
    kept = pairs[pairs.metric.eq("pct_removed")].iloc[0]
    require(
        kept.n_group1 == kept.n_group2 == 4 and kept.test == "wilcoxon_signed_rank",
        "Preparation QC must use paired donor comparisons.",
    )
    counts = pd.read_csv(run / "results/qc/qc_report.csv")
    require(
        (counts.cells_before_qc - counts.cells_removed == counts.cells_after_qc).all(),
        "QC retention arithmetic mismatch.",
    )
    return {
        "donor_pairs": 4,
        "preparations": [kept.group1, kept.group2],
        "paired_removal_p": float(kept.p_value),
    }


def check_kang(run: Path, data: Path) -> dict:
    results = run / "results"
    de = pd.read_csv(results / "de_pseudobulk_edger_by_subcluster.csv")
    diff = pd.read_csv(results / "da_proportion_ttest.csv")
    sets = pd.read_csv(results / "enrichment_gsea_hallmark_by_cell_type.csv")
    isg = sets[sets.source.eq("HALLMARK_INTERFERON_ALPHA_RESPONSE")]
    require(
        len(diff) == 8
        and diff.paired.all()
        and diff.n_case.eq(8).all()
        and diff.n_control.eq(8).all(),
        "Expected eight complete donor pairs across eight labeled populations.",
    )
    require(
        set(diff.cell_type) == set(de.cell_type), "Contrasts do not cover the same populations."
    )
    require(
        len(isg) == 6 and (isg.score > 0).all() and (isg.padj < 0.05).all(),
        "Expected paired interferon enrichment in all six eligible populations.",
    )
    return {"paired_donors": 8, "positive_interferon_populations": 6}


def check_integration(run: Path, data: Path) -> dict:
    path = run / "provenance/stages/integration_benchmark/completion.json"
    metrics = json.loads(path.read_text())["metrics"]
    scores = metrics["embeddings"]
    require(
        metrics["mode"] == "full" and metrics["label_used"] == "cell_type",
        "Both identity preservation and batch mixing must be assessed.",
    )
    rows = [
        (rep, domain, key, score)
        for rep, domains in scores.items()
        for domain in ["bio", "batch"]
        for key, score in domains[domain].items()
    ]
    table = pd.DataFrame(rows, columns=["embedding", "category", "metric", "value"])
    require(np.isfinite(table.value).all(), "Integration metric missing.")
    table.to_csv(run / "reports/integration_quality.csv", index=False)
    raw, harmony = scores["X_pca"], scores["X_pca_harmony"]
    require(
        harmony["batch"]["ilisi"] > raw["batch"]["ilisi"]
        and harmony["bio"]["nmi"] >= raw["bio"]["nmi"] - 0.05,
        "Inspect integration mixing and identity preservation on the public atlas.",
    )
    return {
        "baseline_iLISI": raw["batch"]["ilisi"],
        "harmony_iLISI": harmony["batch"]["ilisi"],
        "baseline_NMI": raw["bio"]["nmi"],
        "harmony_NMI": harmony["bio"]["nmi"],
    }


def check_droplets(run: Path, data: Path) -> dict:
    import scanpy as sc

    original = sc.read_10x_h5(data / "pbmc_1k_v3/outs/filtered_feature_bc_matrix.h5")
    cells = ad.read_h5ad(run / "objects/final_annotated.h5ad")
    original.var_names = original.var.gene_ids
    subset = original[cells.obs.barcode.to_numpy(), cells.var.gene_ids.to_numpy()]
    observed = cells.layers["uncorrected_counts"]
    require(
        (subset.X != observed).nnz == 0,
        "Retained raw counts do not match Cell Ranger by barcode and gene ID.",
    )
    corrected = cells.layers["counts"]
    require(
        corrected.min() >= 0 and corrected.sum() < observed.sum(), "Invalid ambient correction."
    )
    rho = pd.read_csv(run / "objects/soupx/soupx_contamination_fractions.csv")
    return {
        "retained_cells": cells.n_obs,
        "retained_genes": cells.n_vars,
        "original_umis": int(observed.sum()),
        "corrected_umis": int(corrected.sum()),
        "estimated_fraction": float(rho.iloc[0, -1]),
    }


def check_review(run: Path, data: Path) -> dict:
    cells = ad.read_h5ad(run / "objects/final_annotated.h5ad", backed="r")
    try:
        counts = cells.obs.qc_state_final.value_counts()
        total = cells.n_obs
        clusters = set(cells.obs.leiden.astype(str))
    finally:
        cells.file.close()
    consensus = pd.read_csv(run / "results/annotation_consensus_agreement.csv")
    decisions = pd.read_csv(run / "results/adjudication_results.csv", dtype={"cluster_id": str})
    fit = pd.read_csv(run / "results/module_remodeling/module_effect_sizes_lmm.csv")
    provenance = run / "provenance/stages"
    metrics = [
        json.loads((provenance / name / "completion.json").read_text())["metrics"]
        for name in ("query_projection", "qc_finalization", "subclustering")
    ]
    require(
        metrics[0]["n_borderline"] == counts["rescued"] + counts["unresolved_borderline"]
        and metrics[1]["n_rescued"] == counts["rescued"],
        "Borderline projections do not reconcile.",
    )
    require(
        consensus.n_cells.sum() == total and set(decisions.cluster_id) == clusters,
        "Missing cluster adjudications or consensus labels.",
    )
    require(
        fit.group.nunique() == 8
        and fit.n_pairs.eq(np.where(fit.group == "Megakaryocytes", 3, 8)).all(),
        "Unexpected donor support for the fitted population.",
    )
    return {
        "cells": total,
        "borderline": metrics[0]["n_borderline"],
        "rescued": int(counts["rescued"]),
        "subclustered_b_cells": metrics[2]["n_cells_after_focus"],
        "adjudicated_clusters": len(decisions),
        "pairs_per_population": fit.groupby("group").n_pairs.first().to_dict(),
        "unconverged_fits": int((~fit.converged).sum()),
    }


def check_scc_programs(run: Path, data: Path) -> dict:
    results = run / "results"
    obj = ad.read_h5ad(run / "objects/final_annotated.h5ad", backed="r")
    try:
        usage = pd.DataFrame(
            obj.obsm["X_cnmf"], columns=obj.uns["cnmf"]["programs"], index=obj.obs_names
        )
        expected = usage.groupby(obj.obs.celltype_alt, observed=True).mean()
        table = pd.read_csv(results / "discovery_nmf_usage_by_celltype.csv")
        observed = table.pivot(index="cell_type", columns="program", values="mean_usage")
        require(
            np.allclose(expected.loc[observed.index, observed.columns], observed),
            "NMF means disagree with cell usage.",
        )
        meta = pd.read_csv(results / "metacells/metacell_assignments.csv").set_index("cell_id")
        require(
            meta.index.is_unique and set(meta.index) == set(obj.obs_names),
            "Metacell assignment coverage changed.",
        )
        require(
            np.array_equal(meta.loc[obj.obs_names, "metacell_id"], obj.obs.metacell_id.astype(str)),
            "Metacell labels differ from final object.",
        )
        mods = pd.read_csv(results / "coexpression/modules.csv")
        eig = pd.read_csv(results / "coexpression/eigengenes.csv", index_col=0)
        assigned = sorted(set(mods.module) - {"grey"})
        require(
            set(eig.index) == set(obj.obs_names)
            and assigned
            and np.isfinite(eig[assigned]).all().all(),
            "Missing module eigengenes.",
        )
        scores = pd.read_csv(results / "multicellular_programs/mcp_scores.csv")
        support = pd.read_csv(results / "multicellular_programs/program_donor_support.csv")
        scored = scores.join(obj.obs.patient, on="cell_id", validate="many_to_one")
        require(
            scored.patient.notna().all() and np.isfinite(scored.score).all(),
            "Invalid DIALOGUE cell scores.",
        )
        counts = scored.groupby("program").patient.nunique().reindex(support.program)
        require(
            np.array_equal(counts, support.n_donors),
            "DIALOGUE donor support differs from contributing cells.",
        )
        stability = pd.read_csv(results / "multicellular_programs/program_stability.csv")
        require(
            set(stability.program) == set(support.program) and (stability.n_resamples == 5).all(),
            "Missing resample evidence.",
        )
        return {
            "cells": len(meta),
            "metacells": meta.loc[meta.metacell_id != "Outliers", "metacell_id"].nunique(),
            "outlier_cells": int((meta.metacell_id == "Outliers").sum()),
            "modules": assigned,
            "nmf_programs": len(observed.columns),
            "dialogue_resampling": stability.to_dict("records"),
            "dialogue_donors": support.to_dict("records"),
        }
    finally:
        obj.file.close()


def check_scc_network(run: Path, data: Path) -> dict:
    base = run / "results/ccc_network"
    topology = pd.read_csv(base / "topology_cci.csv").set_index("node")
    curvature = pd.read_csv(base / "differential_curvature_cci.csv")
    require(
        set(topology.index) == {"CD8T", "CLEC9A", "KC_other", "Fibroblast"}
        and np.isclose(topology.Pagerank.sum(), 1),
        "Invalid population topology.",
    )
    require(
        len(curvature) > 0
        and np.allclose(curvature.curv_case - curvature.curv_control, curvature.delta_curvature),
        "Incorrect curvature difference.",
    )
    extra = {}
    settings = json.loads((run / "provenance/resolved_config.json").read_text())
    if settings["ccc_network"]["build_gci"]:
        genes = pd.read_csv(base / "topology_gci.csv")
        edges = pd.read_csv(base / "curvature_gci_edges.csv")
        difference = pd.read_csv(base / "differential_curvature_gci.csv")
        require(
            genes.node.is_unique
            and np.isclose(genes.Pagerank.sum(), 1)
            and len(edges) > 0
            and np.isfinite(edges.ricci_curvature).all()
            and np.allclose(
                difference.curv_case - difference.curv_control, difference.delta_curvature
            ),
            "Gene-channel graph checks failed.",
        )
        extra = {"gene_channel_nodes": len(genes), "gene_channel_edges": len(edges)}
    return {
        **extra,
        "highest_outgoing": topology.Influencer.idxmax(),
        "highest_incoming": topology.Listener.idxmax(),
        "edges_compared": len(curvature),
    }


def check_latent(run: Path, data: Path) -> dict:
    from sklearn.neighbors import NearestNeighbors

    obj = ad.read_h5ad(run / "objects/final_annotated.h5ad", backed="r")
    try:
        labels = obj.obs.cell_type.astype(str).to_numpy()
        studies = obj.obs.study.astype(str).to_numpy()
        require(len(set(studies)) == 2, "Expected the two pancreas studies.")
        agreement = {}
        for key in ["X_pca", "X_scvi", "X_scanvi"]:
            latent = np.asarray(obj.obsm[key])
            require(
                len(latent) == len(studies) and np.isfinite(latent).all(),
                "Missing latent coordinates.",
            )
            matches = []
            for study in sorted(set(studies)):
                train = studies != study
                idx = (
                    NearestNeighbors(n_neighbors=1)
                    .fit(latent[train])
                    .kneighbors(latent[~train], return_distance=False)[:, 0]
                )
                matches.extend(labels[train][idx] == labels[~train])
            agreement[key] = float(np.mean(matches))
        return {"cells": len(studies), "cross_study_nearest_neighbor_label_agreement": agreement}
    finally:
        obj.file.close()


def check_oracle(run: Path, data: Path) -> dict:
    base = run / "results/perturbation"
    ranking = pd.read_csv(base / "perturbation_ranking.csv")
    require(set(ranking.tf) == {"Neurog3", "Pdx1"}, "Missing planned transcription factors.")
    for row in ranking.itertuples():
        vectors = pd.read_parquet(base / f"shift_vectors_{row.tf}.parquet")
        require(
            len(vectors) == row.n_cells == 2000
            and vectors.index.is_unique
            and np.isclose(np.linalg.norm(vectors, axis=1).mean(), row.score),
            "KO ranking differs from cell shifts.",
        )
    return {
        "mean_predicted_shift": dict(zip(ranking.tf, ranking.score, strict=True)),
        "cells_per_simulation": 2000,
    }


def check_extended_lineage(run: Path, data: Path) -> dict:
    obj = ad.read_h5ad(run / "objects/final_annotated.h5ad", backed="r")
    try:
        scores = obj.obs.cytotrace2_score
        medians = obj.obs.groupby("clusters", observed=True).cytotrace2_score.median()
        require(
            len(scores) == 3696 and scores.between(0, 1).all(), "Incomplete potency predictions."
        )
        require(
            medians["Ductal"] > medians["Ngn3 high EP"] > medians["Beta"],
            "Review potency along the reference lineage.",
        )
        fates = np.asarray(obj.obsm["cellrank_fate_probabilities"])
        require(
            fates.shape[1] > 1 and np.allclose(fates.sum(axis=1), 1, atol=1e-4),
            "Fate probabilities do not sum to one.",
        )
        result = {
            "cells": len(scores),
            "median_potency": medians.to_dict(),
            "fates": fates.shape[1],
        }
        publication = run / "reports/publication"
        for stem in ("lineage.palantir", "lineage"):
            for extension in ("pdf", "svg"):
                page = publication / f"{stem}.{extension}"
                require(page.is_file() and page.stat().st_size > 0, f"Missing figure: {page.name}")
        panel = pd.read_csv(publication / "lineage.palantir.csv").set_index("group")
        expected = obj.obs.groupby("clusters", observed=True).palantir_pseudotime.agg(
            ["count", "median"]
        )
        require(
            panel.index.is_unique and set(expected.index) == set(panel.index),
            "Palantir figure populations differ from the analysis object.",
        )
        expected = expected.loc[panel.index]
        require(
            np.array_equal(panel.n_cells, expected["count"])
            and np.allclose(panel.value, expected["median"], rtol=0, atol=1e-12),
            "Palantir figure cell counts or medians differ from the analysis object.",
        )
        settings = json.loads((run / "provenance/resolved_config.json").read_text())[
            "trajectory_viz"
        ]
        if any(m.get("gam_distribution") == "gaussian" for m in settings["methods"]):
            curves = pd.read_csv(run / "results/trajectory/gene_trends.csv")
            grouped = curves.groupby(["gene", "lineage"])
            require(
                set(curves.gene) == set(settings["genes"])
                and len(grouped) == len(settings["genes"]) * fates.shape[1]
                and np.isfinite(curves[["pseudotime", "fitted_expression"]]).all().all()
                and grouped.pseudotime.apply(lambda t: t.is_monotonic_increasing).all(),
                "Missing or invalid fitted gene-lineage curves.",
            )
            result["fitted_gene_lineage_curves"] = len(grouped)
        return result
    finally:
        obj.file.close()


def check_kang_expression(run: Path, data: Path) -> dict:
    obj = ad.read_h5ad(run / "objects/final_annotated.h5ad", backed="r")
    try:
        phate = np.asarray(obj.obsm["X_phate"])
        require(
            phate.shape == (obj.n_obs, 2)
            and np.isfinite(phate).all()
            and np.all(np.ptp(phate, axis=0) > 0),
            "Invalid PHATE coordinates.",
        )
        require(obj.var.highly_variable.sum() == 1500, "Unexpected Seurat feature count.")
        design = obj.obs[["replicate", "label", "specimen"]].drop_duplicates()
        require(
            len(design) == design.specimen.nunique() == 16
            and design.groupby("replicate", observed=True).label.nunique().eq(2).all(),
            "Density panel requires two distinct specimens per donor.",
        )
        ora = pd.read_csv(run / "results/enrichment_ora_hallmark.csv")
        alpha = ora[(ora.direction == "up") & (ora.source == "HALLMARK_INTERFERON_ALPHA_RESPONSE")]
        require(
            len(alpha) == 1 and alpha.padj.iat[0] < 0.05 and alpha["count"].iat[0] > 0,
            "Stimulated PBMCs lost the interferon alpha enrichment positive control.",
        )
        return {
            "cells": obj.n_obs,
            "specimens": len(design),
            "interferon_alpha_overlap": int(alpha["count"].iat[0]),
            "interferon_alpha_adjusted_p": float(alpha.padj.iat[0]),
        }
    finally:
        obj.file.close()


def check_kang_sensitivity(run: Path, data: Path) -> dict:
    base = run / "results"
    dream = pd.read_csv(
        base / "differential_expression/pseudobulk_dream/de_pseudobulk_dream_all.csv"
    )
    isg = dream.loc[dream.gene == "ISG15"].set_index("cell_type")
    eligibility = pd.read_csv(
        base / "differential_expression/pseudobulk_dream/cell_type_summary.csv"
    )
    fitted = eligibility.loc[eligibility.status == "fitted", "cell_type"]
    require(
        set(isg.index) == set(fitted)
        and "CD14+ Monocytes" in isg.index
        and (isg.logFC > 0).all()
        and (isg.FDR < 0.05).all()
        and isg.loc["CD14+ Monocytes", "n_subjects"] == 8,
        "ISG15 positive control or population eligibility changed.",
    )
    gsva = pd.read_csv(base / "enrichment_gsva_scores_hallmark.csv", index_col=0)
    contrast = pd.read_csv(base / "enrichment_gsva_contrast_hallmark.csv").set_index("source")
    for group, col in [("stim", "case_mean"), ("ctrl", "control_mean")]:
        group_scores = gsva[gsva.index.str.endswith("__" + group)]
        require(
            len(group_scores) == 8
            and np.allclose(group_scores[contrast.index].mean(), contrast[col]),
            "GSVA group means differ from specimen scores.",
        )
    progeny = pd.read_csv(base / "enrichment_activity_progeny_by_donor.csv")
    require(
        progeny.replicate.nunique() == 8
        and set(progeny.label) == {"ctrl", "stim"}
        and progeny.mean_score.notna().all(),
        "PROGENy sample coverage differs.",
    )
    da = pd.read_csv(base / "da_propeller.csv")
    require(
        np.allclose(100 * (da.case_mean_prop - da.control_mean_prop), da.effect_pp)
        and (da.n_donors_blocked == 8).all(),
        "Propeller effects differ from paired proportions.",
    )
    return {
        "paired_donors": 8,
        "isg15_log2fc": isg.logFC.to_dict(),
        "contributing_subjects": isg.n_subjects.to_dict(),
        "gsva_gene_sets": len(contrast),
        "progeny_networks": progeny.source.nunique(),
    }


CHECKS = {
    "kang_expression_figures": check_kang_expression,
    "scc_network": check_scc_network,
    "scc_programs": check_scc_programs,
    "pancreas_latent_sensitivity": check_latent,
    "pancreas_celloracle": check_oracle,
    "pancreas_lineage_programs": check_extended_lineage,
    "kang_sensitivity": check_kang_sensitivity,
    "pbmc_droplets": check_droplets,
    "kang_population_review": check_review,
    "pancreas_fates": check_fates,
    "pancreas_integration": check_integration,
    "kang_paired_analysis": check_kang,
    "gse140819_cells_vs_nuclei": check_nuclei,
    "scc_paired_signaling": check_scc,
    "pancreas_lineage": check_lineage,
    "pancreas_transfer": check_atlas,
    "k562_gata1": check_knockout,
}


# Each entry points to a quantity read by the corresponding independent check.
ARTIFACTS = {
    "kang_expression_figures": {
        "embeddings.phate": "objects/final_annotated.h5ad",
        "feature_selection.seurat": "objects/final_annotated.h5ad",
        "enrichment.ora": "results/enrichment_ora_hallmark.csv",
    },
    "scc_network": {
        "ccc_network.topology": "results/ccc_network/topology_cci.csv",
        "ccc_network.ricci": "results/ccc_network/differential_curvature_cci.csv",
    },
    "scc_programs": {
        "discovery.nmf": "results/discovery_nmf_usage_by_celltype.csv",
        "metacells.metacells": "results/metacells/metacell_assignments.csv",
        "coexpression.hdwgcna": "results/coexpression/eigengenes.csv",
        "multicellular_programs.dialogue": (
            "results/multicellular_programs/program_donor_support.csv"
        ),
    },
    "pancreas_latent_sensitivity": {
        "integration.scvi": "objects/final_annotated.h5ad",
        "integration.scanvi": "objects/final_annotated.h5ad",
    },
    "pancreas_celloracle": {
        "perturbation.celloracle": "results/perturbation/perturbation_ranking.csv"
    },
    "pancreas_lineage_programs": {
        "trajectory.cytotrace": "objects/final_annotated.h5ad",
        "trajectory.cellrank": "objects/final_annotated.h5ad",
    },
    "kang_sensitivity": {
        "differential_expression.pseudobulk_dream": (
            "results/differential_expression/pseudobulk_dream/de_pseudobulk_dream_all.csv"
        ),
        "enrichment.gsva": "results/enrichment_gsva_scores_hallmark.csv",
        "enrichment.activity": "results/enrichment_activity_progeny_by_donor.csv",
        "differential_abundance.propeller": "results/da_propeller.csv",
    },
    "pbmc_droplets": {"ambient_correction": "objects/final_annotated.h5ad"},
    "kang_population_review": {
        "annotation_consensus": "results/annotation_consensus_agreement.csv",
        "query_projection": "objects/final_annotated.h5ad",
        "qc_finalization": "objects/final_annotated.h5ad",
        "adjudication": "results/adjudication_results.csv",
        "module_remodeling.module_remodeling": (
            "results/module_remodeling/module_effect_sizes_lmm.csv"
        ),
    },
    "scc_paired_signaling": {
        "cell_cell_communication.liana": "results/cell_cell_communication/liana_ranks.csv",
        "cell_cell_communication.tensor_c2c": (
            "results/cell_cell_communication/tensor_factors_contexts.csv"
        ),
        "cell_cell_communication.nichenet": "results/nichenet_ligand_target_weights.csv",
        "cell_cell_communication.multinichenet": "results/mnn_sample_prioritization.csv",
        "grn.pyscenic": "results/grn/scenic_auc_mtx_grn.parquet",
        "enrichment.activity": "results/enrichment_activity_collectri_by_donor.csv",
        "differential_expression.pseudobulk_edger": "results/de_pseudobulk_edger_by_subcluster.csv",
    },
    "pancreas_lineage": {
        "trajectory.dpt": "objects/final_annotated.h5ad",
        "trajectory.velocity": "results/trajectory/velocity/whole_object.h5ad",
    },
    "pancreas_fates": {
        "trajectory.palantir": "results/trajectory/palantir/palantir_pseudotime.h5ad",
        "trajectory.cellrank": "results/trajectory/cellrank/fate_mapping.h5ad",
    },
    "pancreas_transfer": {"reference_mapping.scarches": "results/transferred_type_assignments.csv"},
    "pancreas_integration": {
        "integration.harmony": "reports/integration_quality.csv",
        "integration_benchmark.scib_benchmark": "reports/integration_quality.csv",
    },
    "kang_paired_analysis": {
        "enrichment.gsea_by_cell_type": "results/enrichment_gsea_hallmark_by_cell_type.csv",
        "differential_abundance.proportion_ttest": "results/da_proportion_ttest.csv",
    },
    "gse140819_cells_vs_nuclei": {"qc_evidence": "results/qc/qc_donor_comparisons.csv"},
    "k562_gata1": {"perturbation.sctenifoldknk": "reports/held_out_perturbation.csv"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "provenance/resolved_config.json").read_text())
    name = config["project"]["name"]
    records = json.loads((run / "provenance/stage_execution_records.json").read_text())
    plan = json.loads((run / "provenance/pipeline_plan.json").read_text())
    enabled = {s["name"] for s in plan["stages"] if s["enabled"]}
    complete = {r["stage_name"] for r in records if r["status"] == "success"}
    require(
        enabled <= complete,
        f"Enabled stages without successful execution: {sorted(enabled - complete)}",
    )
    result = CHECKS[name](run, args.data)
    checked = dict(ARTIFACTS[name])
    if "fitted_gene_lineage_curves" in result:
        checked["trajectory_viz.gene_trend_viz"] = "results/trajectory/gene_trends.csv"
        checked["trajectory_viz"] = checked["trajectory_viz.gene_trend_viz"]
    if name == "pancreas_transfer" and "annotation_diagnostics" in enabled:
        checked["annotation_diagnostics"] = (
            "results/annotation_diagnostics/scdiagnostics_results.csv"
        )
        checked["annotation_diagnostics.scdiagnostics"] = checked["annotation_diagnostics"]
    if name == "pancreas_lineage" and "qc_splice_metrics" in enabled:
        checked["qc_splice_metrics"] = "objects/final_annotated.h5ad"
    for key, value in ARTIFACTS[name].items():
        checked.setdefault(key.split(".")[0], value)
    payload = {
        "study": name,
        "checked_artifacts": checked,
        "results": result,
        "successful_stages": sorted(enabled),
        "config_seed": config["run"]["random_seed"],
    }
    (run / "reports/reference_validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
