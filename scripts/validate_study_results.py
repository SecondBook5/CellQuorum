"""Audit real public-study results at the scientific resolution of each method.

Reads existing runs. Requires the release reference-study layout, Poppler's pdftotext,
and the CellQuorum science environment. Writes a review independently of run receipts.
"""

import argparse
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from matplotlib.figure import Figure
from scipy import sparse, stats
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

FINAL = "objects/final_annotated.h5ad"
KANG = "kang/runs/kang_paired_analysis"
KANG_SENSITIVITY = "kang/runs/kang_sensitivity_docker"
KANG_EXPRESSION = "kang/runs/kang_expression_verified"
KANG_SUBCLUSTERS = "kang/runs/kang_population_review"
PANCREAS_ATLAS = "pancreas/runs/pancreas_latent_sensitivity"
PANCREAS_LINEAGE = "pancreas/runs/pancreas_lineage_report"
PANCREAS_FATES = "pancreas/runs/pancreas_fates"
PANCREAS_DPT = "pancreas/runs/pancreas_splice"
SCC = "scc/runs/scc_release"
SCC_NETWORK = "scc/runs/scc_network_gene_channels"
PBMC_DROPLETS = "pbmc_droplets/runs/pbmc_droplets_repaired"
IFN = "HALLMARK_INTERFERON_ALPHA_RESPONSE"
CELL_TYPE = "cell_type"
SUPPORTED = "supported"
LIMITED = "limited"
FAILED = "contradicted"


class Review:
    def __init__(self, data: Path, out: Path) -> None:
        self.base = data
        self.out = out
        self.rows = []

    def csv(self, run: str, path: str, **kwargs: object) -> pd.DataFrame:
        return pd.read_csv(self.base / run / path, **kwargs)

    def elem(self, run: str, path: str, file: str = FINAL) -> object:
        with h5py.File(self.base / run / file) as handle:
            return read_elem(handle[path])

    def pdf(self, run: str, path: str) -> str:
        result = subprocess.check_output(["pdftotext", "-layout", str(self.base / run / path), "-"])
        return " ".join(result.decode().split())

    def record(
        self,
        key: str,
        run: str,
        verdict: str,
        criterion: str,
        result: str,
        paths: list[str],
    ) -> None:
        stage, _, method = key.partition(".")
        for path in paths:
            if not (self.base / run / path).is_file():
                raise FileNotFoundError(self.base / run / path)
        self.rows.append(
            dict(
                stage=stage,
                method=method or "(stage)",
                run=run,
                verdict=verdict,
                criterion=criterion,
                result=result,
                evidence=" | ".join(paths),
            )
        )

    def test(
        self,
        key: str,
        run: str,
        passed: bool,
        criterion: str,
        result: str,
        paths: list[str],
    ) -> None:
        self.record(key, run, SUPPORTED if passed else FAILED, criterion, result, paths)


def paired(values: pd.Series | np.ndarray, obs: pd.DataFrame, statistic: str = "mean") -> pd.Series:
    groups = pd.Series(np.asarray(values), index=obs.index).groupby(
        [obs.replicate, obs.label], observed=True
    )
    means = groups.agg(statistic).unstack()
    return (means.stim - means.ctrl).dropna()


def marker_frame(
    matrix: sparse.spmatrix | np.ndarray,
    var: pd.Index,
    genes: Sequence[str],
    obs: pd.DataFrame,
) -> pd.DataFrame:
    positions = var.get_indexer(genes)
    if (positions < 0).any():
        raise ValueError(f"Missing genes: {np.asarray(genes)[positions < 0].tolist()}")
    values = matrix[:, positions]
    if sparse.issparse(values):
        values = values.toarray()
    return pd.DataFrame(values, columns=genes, index=obs.index)


def check_pbmc(
    review: Review,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, sparse.csr_matrix]:
    obs = review.elem(KANG, "obs")
    var = review.elem(KANG, "var")
    raw = ad.read_h5ad(review.base / "kang/data/kang_2018.h5ad", backed="r")
    try:
        published = raw.obs.loc[obs.index]
        same = (
            (
                published[[CELL_TYPE, "replicate", "label"]].astype(str)
                == obs[[CELL_TYPE, "replicate", "label"]].astype(str)
            )
            .all()
            .all()
        )
        genes = raw.var_names.get_indexer(var.index)
        rows = raw.obs_names.get_indexer(obs.index)
        # Slicing on one axis at a time also supports the legacy sparse H5AD.
        source = raw.X[:][rows][:, genes].tocsr()
    finally:
        raw.file.close()
    assert np.all(genes >= 0) and same
    counts = review.elem(KANG, "layers/counts").tocsr()
    norm = review.elem(KANG, "layers/cellquorum_normalized").tocsr()
    expected = counts.multiply((1e4 / np.asarray(counts.sum(axis=1)).ravel())[:, None]).tocsr()
    expected.data = np.log1p(expected.data)
    err = abs(expected - norm).max()
    mismatches = (source != counts).nnz
    review.test(
        "preprocessing",
        KANG,
        mismatches == 0 and err < 2e-5,
        "Retained source UMIs are identical by barcode and gene. Recompute log1p(CP10K).",
        f"{len(obs):,} cells; {mismatches} changed count entries; maximum normalized error"
        f" {err:.3g}.",
        [FINAL],
    )
    del expected, source
    # Positive controls chosen by lineage and stimulus, independently of HVG rank.
    for run, method, controls in [
        (KANG, "seurat_v3", ["MS4A1", "CD79A", "GNLY", "NKG7", "S100A8", "ISG15", "MX1"]),
        (
            PANCREAS_ATLAS,
            "pearson_residuals",
            ["INS", "GCG", "SST", "PPY", "PRSS1", "KRT19"],
        ),
    ]:
        table = review.elem(run, "var")
        hits = table.index[table.highly_variable]
        reference = [g for g in controls if g in table.index]
        found = [g for g in reference if g in hits]
        p = stats.hypergeom.sf(len(found) - 1, len(table), len(hits), len(reference))
        marker_result = (
            f"{len(found)}/{len(reference)} assayed markers in"
            f" {len(hits)}/{len(table)} HVGs; enrichment P={p:.3g}; recovered"
            f" {', '.join(found)}."
        )
        if method == "seurat_v3":
            review.test(
                "feature_selection." + method,
                run,
                len(found) > 0 and p < 0.05,
                "Prespecified PBMC lineage/stimulus markers are enriched in the variable"
                " genes (one-sided hypergeometric P < 0.05).",
                marker_result,
                [FINAL],
            )
        else:
            x = review.elem(run, "layers/counts").tocsc()
            totals = np.asarray(x.sum(axis=1)).ravel().astype(float)
            frequency = np.asarray(x.sum(axis=0)).ravel() / totals.sum()
            clip = np.sqrt(x.shape[0])
            variance = np.zeros(x.shape[1])
            for j in range(0, x.shape[1], 64):
                window = slice(j, j + 64)
                mu = totals[:, None] * frequency[None, window]
                residual = np.divide(
                    x[:, window].toarray() - mu,
                    np.sqrt(mu + mu**2 / 100),
                    out=np.zeros_like(mu),
                    where=mu > 0,
                )
                variance[window] = np.clip(residual, -clip, clip).var(axis=0)
            overlap = len(set(table.index[np.argsort(variance)[-len(hits) :]]) & set(hits))
            verdict = FAILED if overlap != len(hits) else SUPPORTED if p < 0.05 else LIMITED
            review.record(
                "feature_selection." + method,
                run,
                verdict,
                "Independently reconstruct clipped negative-binomial Pearson residual"
                " variances and top genes. Evaluate marker enrichment within the supplied"
                " gene universe.",
                f"{overlap}/{len(hits)} selected genes reproduced. "
                + marker_result
                + " The atlas input already restricted the gene universe.",
                [FINAL],
            )
    # Leave every donor out of its own retrieval reference.
    retrieval = {}
    for rep, key in [("X_pca", "dimensionality.pca"), ("X_umap", "embeddings.umap")]:
        xy = review.elem(KANG, "obsm/" + rep)
        truth = obs[CELL_TYPE].astype(str).to_numpy()
        pred = np.empty(len(obs), object)
        for donor in obs.replicate.unique():
            hold = obs.replicate.eq(donor).to_numpy()
            idx = (
                NearestNeighbors(n_neighbors=1)
                .fit(xy[~hold])
                .kneighbors(xy[hold], return_distance=False)
                .ravel()
            )
            pred[hold] = truth[~hold][idx]
        recall = pd.Series(pred == truth).groupby(truth).mean()
        acc = np.mean(pred == truth)
        donor_acc = (
            pd.Series(pred == truth, index=obs.index).groupby(obs.replicate, observed=True).mean()
        )
        retrieval[rep] = (
            f"{len(xy):,} cells, {xy.shape[1]} dimensions. Held-out-donor 1-NN accuracy"
            f" {acc:.1%}, macro recall {recall.mean():.1%}, donor range"
            f" {donor_acc.min():.1%}–{donor_acc.max():.1%}. Lowest recall"
            f" {recall.idxmin()} {recall.min():.1%}."
        )
        for k in [key, "dimensionality"] if rep == "X_pca" else [key]:
            review.test(
                k,
                KANG,
                acc > obs[CELL_TYPE].value_counts(normalize=True).max() and recall.mean() > 0.5,
                "Retrieve a published cell label using only other donors. Exceed majority"
                " accuracy and 50% macro recall. Embedding fit included all donors.",
                retrieval[rep],
                [FINAL],
            )
    cross = pd.crosstab(obs.leiden, obs[CELL_TYPE])
    purity = cross.max(axis=1).sum() / len(obs)
    ari = adjusted_rand_score(obs[CELL_TYPE], obs.leiden)
    hom = pd.crosstab(obs.leiden, obs.label, normalize="index").max(axis=1)
    for key in ["clustering", "clustering.leiden"]:
        review.record(
            key,
            KANG,
            LIMITED if (hom > 0.9).any() else SUPPORTED,
            "Compare unsupervised clusters with supplied cell identities and inspect"
            " condition-restricted clusters.",
            f"{len(cross)} clusters, majority-label purity {purity:.1%}, ARI {ari:.3f};"
            f" {(hom > 0.9).sum()} clusters >90% one condition. A cluster can represent"
            " stimulation as well as identity.",
            [FINAL],
        )
    ss = review.elem(KANG_SENSITIVITY, "obs")
    sref = published.reindex(ss.index)[CELL_TYPE].astype(str)
    pred = ss.cq_celltypist.astype(str)
    mapping = {
        "Tcm/Naive helper T cells": "CD4 T cells",
        "Tem/Effector helper T cells": "CD4 T cells",
        "Regulatory T cells": "CD4 T cells",
        "Tem/Trm cytotoxic T cells": "CD8 T cells",
        "Tcm/Naive cytotoxic T cells": "CD8 T cells",
        "CD16+ NK cells": "NK cells",
        "NK cells": "NK cells",
        "B cells": "B cells",
        "Plasma cells": "B cells",
        "Classical monocytes": "CD14+ Monocytes",
        "Non-classical monocytes": "FCGR3A+ Monocytes",
        "DC2": "Dendritic cells",
        "pDC": "Dendritic cells",
        "Megakaryocytes/platelets": "Megakaryocytes",
    }
    # An intermediate macrophage prediction does not uniquely map to either blood monocyte label.
    for name, labels, truth, run in [
        (
            "marker_vote",
            obs.marker_vote_label.astype(str),
            published[CELL_TYPE].astype(str),
            KANG,
        ),
        ("celltypist", pred.map(mapping).fillna("unmapped"), sref, KANG_SENSITIVITY),
    ]:
        right = labels.eq(truth)
        recall = right.groupby(truth).mean()
        pd.crosstab(truth, labels).to_csv(review.out / f"{name}-confusion.csv")
        result = (
            f"Accuracy {right.mean():.1%}; macro recall {recall.mean():.1%}; B-cell recall"
            f" {recall['B cells']:.1%}, CD8 T recall {recall['CD8 T cells']:.1%}; worst"
            f" {recall.idxmin()} {recall.min():.1%}. Unmapped predictions"
            f" {labels.eq('unmapped').sum()}."
        )
        review.record(
            "annotation." + name,
            run,
            FAILED if recall.min() == 0 else LIMITED,
            "All eight published immune labels should be recovered. Report exact/coarsened"
            " agreement and count unmatched predictions as errors.",
            result,
            [FINAL],
        )
        if name == "marker_vote":
            review.record(
                "annotation",
                KANG,
                FAILED if recall.min() == 0 else LIMITED,
                "Routine annotation should recover each supplied broad lineage.",
                result,
                [FINAL],
            )
    same = ss.cq_published.astype(str).eq(sref)
    review.test(
        "annotation.passthrough",
        KANG_SENSITIVITY,
        same.all(),
        "Retain every original published label by barcode.",
        f"{same.sum():,}/{len(same):,} exact. This tests label transfer by identifier.",
        [FINAL],
    )
    pops = review.csv(KANG, "results/population_identity/tables/population_summary.csv").set_index(
        "population_id"
    )
    ns = obs.groupby(CELL_TYPE, observed=True).size().reindex(pops.index)
    nd = obs.groupby(CELL_TYPE, observed=True).replicate.nunique().reindex(pops.index)
    review.test(
        "population_identity",
        KANG,
        np.array_equal(ns, pops.n_cells) and np.array_equal(nd, pops.n_donors),
        "Reconcile per-population cell and donor counts with public metadata.",
        f"{len(ns)} populations, {ns.sum():,} cells; smallest {ns.idxmin()} has"
        f" {ns.min()} cells across {nd[ns.idxmin()]} donors.",
        ["results/population_identity/tables/population_summary.csv", FINAL],
    )
    sub = review.elem(KANG_SUBCLUSTERS, "obs")
    focus = sub[sub.subcluster.notna()]
    by = pd.crosstab(focus.subcluster, focus.label)
    donor = pd.crosstab(focus.subcluster, focus.replicate)
    review.record(
        "subclustering",
        KANG_SUBCLUSTERS,
        LIMITED,
        "Check B-cell identity, donor representation and condition purity in each proposed"
        " subcluster.",
        f"{len(focus)} cells; {focus[CELL_TYPE].eq('B cells').mean():.1%} supplied B cells."
        f" {len(by)} subclusters, minimum {(donor > 0).sum(axis=1).min()} donors; strongest"
        f" single-condition fraction {by.div(by.sum(axis=1), axis=0).max().max():.1%}."
        " Subtype ground truth unavailable.",
        [FINAL],
    )
    # Reconstruct actual marker-table values across all displayed groups and genes.
    for run, name, prefix in [
        (KANG, "marker_dotplot", "marker_dotplot"),
        (KANG_EXPRESSION, "lineage_dotplot", "lineage_marker_dotplot"),
    ]:
        files = [f"results/annotation/{prefix}_{s}.csv" for s in ["mean", "pct"]]
        table, pct = [review.csv(run, f, index_col=0) for f in files]
        if run == KANG_EXPRESSION:
            table, pct = table.T, pct.T
        ids = [str(g).split("|")[-1] for g in table.index]
        ob = review.elem(run, "obs")
        x = marker_frame(norm, var.index, ids, obs).loc[ob.index]
        avg = x.groupby(ob[CELL_TYPE], observed=True).mean().T[table.columns]
        detect = x.gt(0).groupby(ob[CELL_TYPE], observed=True).mean().T[table.columns] * 100
        err = np.max(abs(avg.to_numpy() - table.to_numpy()))
        perr = np.max(abs(detect.to_numpy() - pct.to_numpy()))
        fig = f"figures/annotation/{prefix}.pdf"
        text = review.pdf(run, fig)
        present = sum(g in text for g in set(ids))
        result = (
            f"{table.size} means and {pct.size} detection fractions; largest errors"
            f" {err:.3g} expression units and {perr:.3g} percentage points;"
            f" {present}/{len(set(ids))} gene labels recoverable in PDF text."
        )
        for key in (
            ["annotation_viz", f"annotation_viz.{name}"]
            if run == KANG
            else [f"annotation_viz.{name}"]
        ):
            review.test(
                key,
                run,
                err < 1e-4 and perr < 1e-5 and present > 0,
                "Recompute every displayed marker summary from normalized single-cell"
                " measurements; inspect PDF gene labels.",
                result,
                [*files, fig, FINAL],
            )
    return obs, ss, var, norm


def check_pbmc_contrasts(
    review: Review,
    obs: pd.DataFrame,
    ss: pd.DataFrame,
    var: pd.DataFrame,
    norm: sparse.csr_matrix,
) -> None:
    signals = {}
    for method, run, o, store, col in [
        ("aucell", KANG, obs, "aucell", "mean_auc"),
        ("ucell", KANG_SENSITIVITY, ss, "ucell", "mean_ucell"),
        ("score_genes", KANG, obs, None, "mean_score"),
    ]:
        if store:
            names = list(review.elem(run, "uns/state_" + store + "/programs"))
            frame = pd.DataFrame(
                review.elem(run, "obsm/X_state_" + store), columns=names, index=o.index
            )
        else:
            frame = o.filter(regex="^state_").rename(columns=lambda c: c.removeprefix("state_"))
        by = frame.groupby(o[CELL_TYPE], observed=True).mean()
        path = f"results/state_scoring_{method}_by_celltype.csv"
        table = review.csv(run, path).pivot(index=CELL_TYPE, columns="program", values=col)
        err = np.nanmax(abs(table - by.loc[table.index, table.columns]))
        diff = paired(frame.interferon_isg, o)
        signals[method] = diff
        result = (
            f"ISG score increased in {(diff > 0).sum()}/{len(diff)} donor pairs, median"
            f" paired shift {diff.median():.4g}; maximum population-table difference"
            f" {err:.3g}."
        )
        for key in (
            [f"state_scoring.{method}", "state_scoring"]
            if method == "score_genes"
            else [f"state_scoring.{method}"]
        ):
            review.test(
                key,
                run,
                err < 1e-4 and len(diff) == 8 and (diff > 0).all(),
                "Recompute population means and test the prespecified IFN-stimulation"
                " direction within donor.",
                result,
                [path, FINAL],
            )
    pd.DataFrame(signals).to_csv(review.out / "interferon-donor-differences.csv")
    st = review.csv(KANG, "results/differential_state_results.csv")
    errors = []
    p_errors = []
    for row in st.itertuples():
        diff = paired(obs[row.program], obs, "median")
        errors.append(abs(diff.median() - row.effect))
        p_errors.append(abs(stats.wilcoxon(diff).pvalue - row.p_value))
    review.test(
        "differential_state",
        KANG,
        max(errors) < 1e-7 and max(p_errors) < 1e-7,
        "Reproduce each paired donor-median effect and exact two-sided signed-rank P from"
        " cell scores.",
        f"{len(st)} programs, largest effect error {max(errors):.2g}, P error"
        f" {max(p_errors):.2g}; IFN effect"
        f" {st.set_index('program').loc['state_interferon_isg', 'effect']:.3f}.",
        ["results/differential_state_results.csv", FINAL],
    )
    auc = review.elem(KANG, "obsm/X_state_aucell")[
        :, list(review.elem(KANG, "uns/state_aucell/programs")).index("interferon_isg")
    ]
    groups = []
    for cell in obs[CELL_TYPE].unique():
        keep = obs[CELL_TYPE].eq(cell)
        diff = paired(auc[keep], obs[keep], "median")
        groups.append((cell, len(diff), int((diff > 0).sum()), stats.wilcoxon(diff).pvalue))
    violin = "figures/state_scoring/state_score_violin_aucell_interferon_isg.pdf"
    labels = review.pdf(KANG, violin)
    n = int((stats.false_discovery_control([g[3] for g in groups]) < 0.05).sum())
    drawn = len(re.findall(r"(?<!\*)\*+(?!\*)", labels))
    for key in ["state_scoring_viz", "state_scoring_viz.state_score_violin"]:
        review.record(
            key,
            KANG,
            (SUPPORTED if n == drawn and n > 0 and "donor-level paired test" in labels else FAILED),
            "Read the IFN panel replicate legend and independently test each population's"
            " within-donor median direction.",
            f"{n}/{len(groups)} populations had paired signed-rank BH-adjusted P<0.05;"
            f" {drawn} significance marks on the PDF. Rare-population estimates have fewer"
            " contributing cells.",
            [violin, FINAL],
        )
    cat = "figures/embeddings/categorical_umap_cell_type.pdf"
    t = review.pdf(KANG, cat)
    counted = sum(
        f"{cell} ({count:,})" in t for cell, count in obs[CELL_TYPE].value_counts().items()
    )
    review.test(
        "embeddings.categorical_embedding",
        KANG,
        counted == obs[CELL_TYPE].nunique(),
        "Reconcile all PDF legend counts with public labels retained by QC.",
        f"{counted}/{obs[CELL_TYPE].nunique()} category totals match.",
        [cat, FINAL],
    )
    gene = "ISG15"
    x = marker_frame(norm, var.index, [gene], obs)[gene]
    d = paired(x, obs)
    overlay = "figures/embeddings/overlay_umap_ISG15.pdf"
    label = review.pdf(KANG_EXPRESSION, overlay)
    review.test(
        "embeddings.continuous_overlay",
        KANG_EXPRESSION,
        ("ISG15" in label and "log-normalised" in label and (d > 0).all()),
        "Check gene/layer legend and independent within-donor expression direction for the"
        " illustrated ISG.",
        f"ISG15 increased in {int((d > 0).sum())}/8 donors, median change"
        f" {d.median():.3f} log-normalized units. Checks the plotted biological contrast.",
        [overlay, FINAL],
    )
    statefig = "figures/embeddings/state_score_aucell_interferon_isg.pdf"
    state_text = review.pdf(KANG_EXPRESSION, statefig)
    counts = obs.label.value_counts()
    eq = all(f"{n:,} cells" in state_text for n in [len(obs), *counts])
    review.test(
        "embeddings.state_score_umap",
        KANG_EXPRESSION,
        eq and (signals["aucell"] > 0).all(),
        "Match the three panel denominators and check the score's response in matched donors.",
        f"Panel denominators {len(obs):,}, {counts.ctrl:,} control,"
        f" {counts.stim:,} stimulated; positive score response in"
        f" {(signals['aucell'] > 0).sum()}/8 donors.",
        [statefig, FINAL],
    )
    xy = review.elem(KANG_EXPRESSION, "obsm/X_umap")
    eo = review.elem(KANG_EXPRESSION, "obs")
    # Independent fixed-grid sensitivity to leaving a patient out, equal specimen weights.
    limits = np.percentile(xy, [0.3, 99.7], axis=0).T
    maps = {}
    for donor in eo.replicate.unique():
        arms = {}
        for arm in ["stim", "ctrl"]:
            sel = eo.replicate.eq(donor) & eo.label.eq(arm)
            h, _, _ = np.histogram2d(*xy[sel].T, bins=32, range=limits)
            arms[arm] = h / h.sum()
        maps[donor] = arms["stim"] - arms["ctrl"]
    allmap = np.mean(list(maps.values()), axis=0)
    leave = [
        np.corrcoef(
            allmap.ravel(),
            np.mean([m for d, m in maps.items() if d != donor], axis=0).ravel(),
        )[0, 1]
        for donor in maps
    ]
    density = "figures/embeddings/density_difference_umap.pdf"
    t = review.pdf(KANG_EXPRESSION, density)
    review.test(
        "embeddings.density_difference_umap",
        KANG_EXPRESSION,
        min(leave) > 0.8 and "8 specimens" in t,
        "Check specimen denominators and leave-one-donor-out stability on an independent"
        " 32×32 grid.",
        "Eight paired specimens per arm; full versus leave-one-pair-out density-map"
        f" correlations {min(leave):.3f}–{max(leave):.3f}. Describes location on a fitted"
        " embedding.",
        [density, FINAL],
    )
    paga = review.elem(KANG, "uns/paga")
    grouping = obs[paga["groups"]]
    edges = paga["connectivities"].tocoo()
    labels = grouping.cat.categories
    edge = max(zip(edges.data, edges.row, edges.col, strict=True))
    missed = set(obs[CELL_TYPE].unique()) - set(labels)
    review.record(
        "embeddings.paga",
        KANG,
        LIMITED,
        "Check graph endpoints against the original annotations. A missing immune lineage"
        " prevents interpretation of its edges.",
        f"{len(labels)} graph labels; strongest edge"
        f" {labels[edge[1]]}–{labels[edge[2]]} ({edge[0]:.3f}); absent published groups:"
        f" {', '.join(sorted(missed)) or 'none'}. Network uses predicted {paga['groups']}.",
        [FINAL],
    )
    prop = review.csv(KANG, "results/composition_proportions.csv")
    groups = [
        obs.replicate.astype(str).rename("donor"),
        obs.label.astype(str).rename("condition"),
        obs[CELL_TYPE].astype(str).rename(CELL_TYPE),
    ]
    independent = obs.groupby(groups, observed=True).size()
    use = prop.set_index(["donor", "condition", CELL_TYPE])
    rec = independent.reindex(use.index, fill_value=0)
    error = abs(use.proportion - rec / rec.groupby(level=[0, 1]).transform("sum")).max()
    review.test(
        "composition",
        KANG,
        np.array_equal(rec, use["count"]) and error < 1e-9,
        "Recompute cell counts and fractions in every donor × condition × population.",
        f"{len(prop)} strata in 16 specimens; max fraction difference {error:.2g}; counts"
        f" sum to {rec.sum():,}.",
        ["results/composition_proportions.csv", FINAL],
    )
    sc = review.csv(KANG_SENSITIVITY, "results/da_sccoda.csv")
    primary = sc[sc.is_primary]
    diag = json.loads(
        (review.base / KANG_SENSITIVITY / "scratch/da_sccoda_diagnostics.json").read_text()
    )
    delta = prop.pivot(index=["donor", CELL_TYPE], columns="condition", values="proportion").eval(
        "100*(stim-ctrl)"
    )
    ranked = primary.set_index(CELL_TYPE)
    err = (delta.groupby(level=1).mean() - ranked.mean_delta_pp).abs().max()
    calls = {
        str(ref): set(g.loc[g.credible_effect, CELL_TYPE]) for ref, g in sc.groupby("reference")
    }
    review.record(
        "differential_abundance.sccoda",
        KANG_SENSITIVITY,
        LIMITED if len(set(map(frozenset, calls.values()))) > 1 else SUPPORTED,
        "Check observed paired fractions and whether credible population calls survive a"
        " change in model reference.",
        f"Maximum observed donor-shift error {err:.3g} pp. Credible effects per reference"
        f" { {k: len(val) for k, val in calls.items()} };"
        f" {int((primary.sign_test_fdr < 0.05).sum())} donor sign tests at FDR<0.05."
        f" {len(diag['fits'])} reference fits.",
        ["results/da_sccoda.csv", "scratch/da_sccoda_diagnostics.json", FINAL],
    )
    milo = review.csv(KANG_SENSITIVITY, "results/da_milo.csv")
    sig = milo.SpatialFDR.lt(0.05)
    broad = review.csv(KANG_SENSITIVITY, "results/da_propeller.csv")
    # Re-estimate local condition segregation independently in the fitted PCA space.
    xyz = review.elem(KANG_SENSITIVITY, "obsm/X_pca")
    idx = (
        NearestNeighbors(n_neighbors=31)
        .fit(xyz)
        .kneighbors(xyz[::10], return_distance=False)[:, 1:]
    )
    arm = ss.label.astype(str).to_numpy()
    purity = (arm[idx] == arm[::10, None]).mean()
    review.record(
        "differential_abundance.milo",
        KANG_SENSITIVITY,
        LIMITED,
        "Distinguish neighborhood redistribution under IFN from changes in broad population"
        " proportions. Independently measure local arm segregation.",
        f"{sig.sum()}/{len(milo)} neighborhoods at spatial FDR<0.05"
        f" ({(sig & milo.logFC.gt(0)).sum()} positive); same-arm fraction among 30 PCA"
        f" neighbors {purity:.1%}; {broad.FDR.lt(0.05).sum()}/8 broad Propeller calls."
        " Per-neighborhood donor membership is unavailable for replication.",
        ["results/da_milo.csv", "results/da_propeller.csv", FINAL],
    )


def pathways(review: Review, obs: pd.DataFrame, var: pd.DataFrame, norm: sparse.csr_matrix) -> None:
    top = "results/enrichment_gsea_hallmark.csv"
    running = "results/enrichment_gsea_runningES_hallmark.csv"
    sets = "results/enrichment_gsea_hallmark_by_cell_type.csv"
    gsea = review.csv(KANG, top).set_index("source")
    traces = review.csv(KANG, running)
    errors = []
    for _, tab in traces.groupby("source"):
        tab = tab.sort_values("rank")
        weights = tab.metric.abs() * tab.hit
        steps = weights / weights.sum() - (1 - tab.hit) / (1 - tab.hit).sum()
        errors.append(abs(tab.running_es - steps.cumsum()).max())
    ct = review.csv(KANG, sets)
    positive = ct[ct.source.eq(IFN)]
    good = (positive.score.gt(0) & positive.padj.lt(0.05)).sum()
    review.test(
        "enrichment.gsea",
        KANG,
        max(errors) < 1e-7 and good == len(positive) and good > 0,
        "Rebuild weighted running enrichment from ranked gene metrics and hits. Check"
        " IFN-alpha direction and adjusted significance per population.",
        f"{len(errors)} running curves, maximum trace error {max(errors):.3g};"
        f" {good}/{len(positive)} tested populations positive at FDR<0.05.",
        [top, running, sets],
    )
    div = "figures/enrichment/gsea_diverging_hallmark.pdf"
    txt = review.pdf(KANG, div)
    rank = gsea.score.rank(ascending=False).loc[IFN]
    review.test(
        "enrichment_viz.gsea_viz",
        KANG,
        rank <= 3 and "HALLMARK_INTERFERON_ALPHA" in txt,
        "Check whether a top positive interferon pathway from the contrast table is displayed.",
        f"IFN alpha ranked {int(rank)}/{len(gsea)} by NES ({gsea.loc[IFN, 'score']:.3f},"
        f" adjusted P={gsea.loc[IFN, 'padj']:.3g}).",
        [div, top],
    )
    nesmat = "scratch/nes_hallmark.csv"
    nmat = review.csv(KANG, nesmat, index_col=0)
    truth = ct.pivot(index="source", columns=CELL_TYPE, values="score").reindex(
        index=nmat.index, columns=nmat.columns
    )
    err = np.nanmax(abs(nmat - truth))
    mask_same = np.array_equal(nmat.isna(), truth.isna())
    fig = "figures/enrichment/gsea_nes_heatmap_hallmark.pdf"
    text = review.pdf(KANG, fig)
    rownums = re.search(r"HALLMARK_INTERFERON_ALPHA_RESPONSE (.*?)(?:HALLMARK_|$)", text)
    nums = sorted(re.findall(r"([\d]+\.\d)\*?", rownums[1])[: len(positive)]) if rownums else []
    expected = sorted(f"{x:.1f}" for x in positive.score)
    for key in ["enrichment_viz", "enrichment_viz.gsea_nes_heatmap"]:
        review.test(
            key,
            KANG,
            err < 1e-7 and mask_same and nums == expected,
            "Match the full heatmap input, missing-test masks and the six rounded IFN-alpha"
            " values printed on the PDF.",
            f"{truth.notna().sum().sum()} finite NES entries, max error {err:.2g}; PDF IFN"
            f" row {', '.join(nums)}.",
            [fig, nesmat, sets],
        )
    gs = "results/enrichment_gsva_scores_hallmark.csv"
    con = "results/enrichment_gsva_contrast_hallmark.csv"
    scores = review.csv(KANG_SENSITIVITY, gs, index_col=0)
    table = review.csv(KANG_SENSITIVITY, con).set_index("source")
    arms = scores.index.str.rsplit("__", n=1).str[-1]
    err = max(
        (scores[arms == "stim"].mean() - table.case_mean).abs().max(),
        (scores[arms == "ctrl"].mean() - table.control_mean).abs().max(),
    )
    series = scores[IFN].copy()
    series.index = pd.MultiIndex.from_tuples([x.rsplit("__", 1) for x in series.index])
    means = series.unstack()
    fig = "figures/enrichment/gsva_heatmap_hallmark.pdf"
    review.test(
        "enrichment_viz.gsva_viz",
        KANG_SENSITIVITY,
        err < 1e-7
        and (means.stim > means.ctrl).all()
        and "HALLMARK_INTERFERON_ALPHA" in review.pdf(KANG_SENSITIVITY, fig),
        "Rebuild pathway group means from specimen scores and check IFN direction within donor.",
        f"All {len(scores.columns)} pathways matched (max error {err:.2g}); IFN alpha"
        f" increased in {(means.stim > means.ctrl).sum()}/8 pairs.",
        [fig, gs, con],
    )
    gmt = (review.base / "kang/data/h.all.v2025.1.Hs.symbols.gmt").read_text().splitlines()
    genes = set(next(s.split("\t")[2:] for s in gmt if s.startswith(IFN + "\t")))
    de = review.csv(KANG_EXPRESSION, "results/de_pseudobulk_edger.csv")
    hits = de[de.FDR.lt(0.05) & de.logFC.gt(0)]
    count = len(set(hits.gene) & genes)
    ora = "results/enrichment_ora_hallmark.csv"
    table = review.csv(KANG_EXPRESSION, ora)
    observed = table[(table.source == IFN) & (table.direction == "up")].iloc[0]
    fig = "figures/enrichment/ora_barplot_hallmark.pdf"
    review.test(
        "enrichment_viz.ora_viz",
        KANG_EXPRESSION,
        count == observed["count"]
        and "HALLMARK_INTERFERON_ALPHA" in review.pdf(KANG_EXPRESSION, fig),
        "Intersect the tested upregulated genes with the release GMT independently and"
        " check displayed pathway/arm.",
        f"{count} directly measured overlapping upregulated genes; ORA exported"
        f" {int(observed['count'])}, adjusted P={observed.padj:.3g}.",
        [fig, ora, "results/de_pseudobulk_edger.csv"],
    )
    activity = "results/enrichment_activity_progeny.csv"
    bydonor = "results/enrichment_activity_progeny_by_donor.csv"
    act = (
        review.csv(KANG_SENSITIVITY, activity).set_index([CELL_TYPE, "label", "source"]).mean_score
    )
    donor = review.csv(KANG_SENSITIVITY, bydonor)
    grp = [CELL_TYPE, "label", "source"]
    recomputed = (
        donor.assign(total=donor.mean_score * donor.n_cells).groupby(grp).total.sum()
        / donor.groupby(grp).n_cells.sum()
    )
    err = (recomputed - act).abs().max()
    j = (
        donor[donor.source == "JAK-STAT"]
        .pivot(index=[CELL_TYPE, "replicate"], columns="label", values="mean_score")
        .dropna()
    )
    fig = "figures/enrichment/activity_dotplot_progeny.pdf"
    review.test(
        "enrichment_viz.activity_viz",
        KANG_SENSITIVITY,
        err < 1e-5 and "JAK-STAT" in review.pdf(KANG_SENSITIVITY, fig),
        "Re-aggregate donor-specific pathway activities using contributing cell counts;"
        " examine the IFN-linked JAK–STAT direction.",
        f"{len(act)} group scores, max weighted-mean error {err:.3g}; JAK–STAT higher in"
        f" {(j.stim > j.ctrl).sum()}/{len(j)} matched donor–population comparisons."
        " Activities are inferred.",
        [fig, bydonor, activity],
    )
    glob = "results/de_pseudobulk_edger.csv"
    sub = "results/de_pseudobulk_edger_by_subcluster.csv"
    globalde = review.csv(KANG, glob)
    by = review.csv(KANG, sub)

    def volcano(df: pd.DataFrame, fig: str) -> tuple[bool, int, int]:
        up = int((df.logFC.gt(1) & df.FDR.lt(0.05)).sum())
        down = int((df.logFC.lt(-1) & df.FDR.lt(0.05)).sum())
        return f"{up} up, {down} down" in review.pdf(KANG, fig), up, down

    fig = "figures/differential_expression/volcano.pdf"
    matched, up, down = volcano(globalde, fig)
    for key in ["de_viz", "de_viz.volcano_viz"]:
        review.test(
            key,
            KANG,
            matched,
            "Recompute both PDF discovery counts from the full donor-aware edgeR contrast"
            " using its displayed thresholds.",
            f"{up} up and {down} down at FDR<0.05 and |log2FC|>1; exact PDF label match {matched}.",
            [fig, glob],
        )
    matches = []
    files = []
    for group, df in by.groupby(CELL_TYPE):
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", group).strip("_")
        fig = f"figures/differential_expression/volcano_subcluster_{slug}.pdf"
        matches.append(volcano(df, fig)[0])
        files.append(fig)
    review.test(
        "de_viz.volcano_by_subcluster",
        KANG,
        all(matches),
        "Recompute both threshold counts separately for each exported population volcano.",
        f"{sum(matches)}/{len(matches)} population PDFs exactly reproduce both counts.",
        [sub, *files],
    )
    path = "scratch/de_heatmap_matrix.csv"
    mat = review.csv(KANG, path, index_col=0)
    data = marker_frame(norm, var.index, mat.index.tolist(), obs)
    grouped = data.groupby(obs[CELL_TYPE].astype(str) + "_" + obs.label.astype(str)).mean().T
    err = (mat - grouped.loc[mat.index, mat.columns]).abs().max().max()
    fig = "figures/differential_expression/de_heatmap.pdf"
    printed = review.pdf(KANG, fig).split()
    visible = sum(g in printed for g in mat.index)
    review.test(
        "de_viz.de_heatmap",
        KANG,
        err < 1e-5 and visible > 0,
        "Recompute displayed gene-by-population-by-condition expression means from cells"
        " and inspect the heatmap labels.",
        f"{mat.size} source means, maximum error {err:.3g}; {visible}/{len(mat)} gene"
        " labels extracted from PDF.",
        [fig, path, FINAL],
    )


def droplets(review: Review) -> None:
    unc = review.elem(PBMC_DROPLETS, "layers/uncorrected_counts")
    cor = review.elem(PBMC_DROPLETS, "layers/counts")
    var = review.elem(PBMC_DROPLETS, "var")
    raw = review.base / "pbmc_droplets/pbmc_1k_v3/outs"
    with h5py.File(raw / "filtered_feature_bc_matrix.h5") as f:
        called = set(f["matrix/barcodes"].asstr()[:])
    with h5py.File(raw / "raw_feature_bc_matrix.h5") as f:
        g = f["matrix"]
        mat = sparse.csc_matrix(
            (g["data"][:], g["indices"][:], g["indptr"][:]), shape=g["shape"][:]
        )
        sums = np.asarray(mat.sum(axis=0)).ravel()
        # Exclude called barcodes and use a low-UMI window as an independent ambient proxy.
        empty = (sums >= 10) & (sums <= 100) & ~np.isin(g["barcodes"].asstr()[:], list(called))
        genes = pd.Index(g["features/id"].asstr()[:])
        background = np.asarray(mat[:, empty].sum(axis=1)).ravel()[genes.get_indexer(var.gene_ids)]
    removed = np.asarray((unc - cor).sum(axis=0)).ravel()
    keep = background > 0
    rho = stats.spearmanr(background[keep], removed[keep]).statistic
    take = np.argsort(background)[-100:]
    share = removed[take].sum() / removed.sum()
    gene_totals = np.asarray(unc.sum(axis=0)).ravel()
    rawshare = gene_totals[take].sum() / unc.sum()
    tested = (gene_totals >= 20) & (background > 0)
    specificity = stats.spearmanr(
        background[tested] / gene_totals[tested], removed[tested] / gene_totals[tested]
    ).statistic
    review.record(
        "ambient_correction.soupx",
        PBMC_DROPLETS,
        LIMITED,
        "Compare removed counts with RNA measured in 10–100 UMI uncalled droplets. No"
        " per-molecule contamination truth is available.",
        f"{empty.sum():,} background barcodes, {int(removed.sum()):,} UMIs removed."
        f" Per-gene Spearman rho {rho:.3f}; top 100 ambient genes account for"
        f" {share:.1%} of removals versus {rawshare:.1%} of original cellular UMIs. Across"
        f" {tested.sum()} genes with at least 20 cellular UMIs and ambient reads, removal"
        f" fraction versus ambient/cellular enrichment rho {specificity:.3f}.",
        [FINAL, "objects/soupx/soupx_contamination_fractions.csv"],
    )


def trajectory(review: Review) -> None:
    source = ad.read_h5ad(review.base / "pancreas/endocrinogenesis.h5ad", backed="r")
    try:
        labels = source.obs.clusters.astype(str)
    finally:
        source.file.close()
    do = review.elem(PANCREAS_DPT, "obs")
    pt = do.dpt_pseudotime
    lab = labels.loc[do.index]
    med = pt.where(np.isfinite(pt)).groupby(lab).median()
    order = ["Ductal", "Ngn3 high EP", "Pre-endocrine", "Beta"]
    fig = "figures/trajectory/pseudotime_dpt_pseudotime.pdf"
    review.test(
        "trajectory_viz.pseudotime_viz",
        PANCREAS_DPT,
        med[order].is_monotonic_increasing
        and "pseudotime" in review.pdf(PANCREAS_DPT, fig).lower(),
        "Ductal, Ngn3-high, pre-endocrine and beta medians increase along rooted DPT;"
        " disclose unreachable cells.",
        "Group medians "
        + ", ".join(f"{s} {med[s]:.3f}" for s in order)
        + f"; {sum(~np.isfinite(pt))} unreachable cells. The ductal root was supplied.",
        [fig, FINAL],
    )
    po = review.elem(PANCREAS_LINEAGE, "obs")
    lab = labels.loc[po.index]
    var = review.elem(PANCREAS_LINEAGE, "var")
    norm = review.elem(PANCREAS_LINEAGE, "layers/cellquorum_normalized").tocsr()
    genes = ["Neurog3", "Ins1", "Gcg", "Sst"]
    expr = marker_frame(norm, var.index, genes, po)
    averages = expr.groupby(lab).mean()
    peak = averages.idxmax()
    expected = ["Ngn3 high EP", "Beta", "Alpha", "Delta"]
    fig = "figures/trajectory/pseudotime_heatmap.pdf"
    txt = review.pdf(PANCREAS_LINEAGE, fig)
    qt = np.quantile(po.palantir_pseudotime, [0, 0.25, 0.5, 0.75, 1])
    qt = (qt - qt[0]) / (qt[-1] - qt[0])
    correct_axis = "rank within condition" in txt
    review.test(
        "trajectory_viz.pseudotime_heatmap",
        PANCREAS_LINEAGE,
        correct_axis
        and all(peak[g] == x and g in txt for g, x in zip(genes, expected, strict=True)),
        "Check displayed endocrine genes against public labels and verify that the axis"
        " identifies equal-cell-count rank bins.",
        "Highest means: "
        + ", ".join(f"{g} in {peak[g]}" for g in genes)
        + ". The 25th/50th/75th cell percentiles fall at"
        f" {qt[1]:.1%}/{qt[2]:.1%}/{qt[3]:.1%} of the numeric time range. Original PDF"
        f" rank axis correct: {correct_axis}.",
        [fig, FINAL],
    )
    file = "results/trajectory/velocity/whole_object.h5ad"
    vel = review.elem(PANCREAS_LINEAGE, "obs", file)
    graph = review.elem(PANCREAS_LINEAGE, "uns/velocity_graph", file).tocoo()
    ordinal = {
        "Ductal": 0,
        "Ngn3 low EP": 1,
        "Ngn3 high EP": 2,
        "Pre-endocrine": 3,
        "Beta": 4,
        "Alpha": 4,
        "Delta": 4,
        "Epsilon": 4,
    }
    step = labels.loc[vel.index].map(ordinal).to_numpy()
    assert np.isfinite(step).all()
    delta = step[graph.col] - step[graph.row]
    # Normalize per source cell before comparing cross-state forward and reverse weights.
    rowsums = np.bincount(graph.row, weights=graph.data, minlength=len(step))
    weights = graph.data / rowsums[graph.row]
    forward = weights[delta > 0].sum() / weights[delta != 0].sum()
    med = vel.velocity_pseudotime.groupby(labels.loc[vel.index]).median()
    projected = review.elem(PANCREAS_LINEAGE, "obsm/velocity_umap", file)
    fig = "figures/trajectory/velocity_stream.pdf"
    review.record(
        "trajectory_viz.velocity_viz",
        PANCREAS_LINEAGE,
        LIMITED if forward > 0.5 else FAILED,
        "Evaluate transition direction on independently annotated developmental stages."
        " Terminal endocrine types share a rank; within-stage edges excluded.",
        f"Forward share of normalized cross-stage weight {forward:.1%}; finite projected"
        f" cell vectors {np.isfinite(projected).all(axis=1).sum()}/{len(vel)}. Ductal/Beta"
        f" velocity-time medians {med['Ductal']:.3f}/{med['Beta']:.3f}. One embryonic"
        " snapshot; direction lacks time-traced validation.",
        [fig, file],
    )
    fo = review.elem(PANCREAS_FATES, "obs")
    lineage = list(review.elem(PANCREAS_FATES, "uns/trajectory/cellrank/fate_names"))
    truth = labels.loc[fo.index]
    probs = pd.DataFrame(
        review.elem(PANCREAS_FATES, "obsm/cellrank_fate_probabilities"),
        columns=lineage,
        index=fo.index,
    )
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    pred = probs.idxmax(axis=1)
    endocrine = ["Alpha", "Beta", "Delta", "Epsilon"]
    recall = pred.eq(truth).groupby(truth).mean().reindex(endocrine)
    recall.to_csv(
        review.out / "endocrine-fate-recall.csv", header=["same_label_maximum_probability"]
    )
    fig = "figures/trajectory/fate_Delta.pdf"
    review.record(
        "trajectory_viz.fate_viz",
        PANCREAS_FATES,
        LIMITED if set(endocrine) - set(lineage) or (recall < 0.5).any() else SUPPORTED,
        "Compare highest inferred fate with observed terminal identities in already"
        " differentiated cells.",
        "Fraction with matching highest fate: "
        + ", ".join(f"{s} {recall[s]:.1%}" for s in endocrine)
        + ". Progenitor future fate remains unobserved.",
        [fig, FINAL],
    )
    macro = fo.cellrank_macrostates.dropna().astype(str)
    fig = "figures/trajectory/macrostates.pdf"
    review.record(
        "trajectory_viz.macrostate_viz",
        PANCREAS_FATES,
        LIMITED,
        "Compare assigned macrostate representatives and the terminal-state inventory with"
        " the four endocrine lineages.",
        f"{len(macro)}/{len(fo)} representative cells; terminal states"
        f" {', '.join(lineage)}. Missing endocrine fates:"
        f" {', '.join(sorted(set(endocrine) - set(lineage))) or 'none'}. Ductal and"
        " progenitor states were called terminal.",
        [fig, FINAL],
    )
    genes = review.elem(PANCREAS_FATES, "var").index
    driver = pd.DataFrame(
        review.elem(PANCREAS_FATES, "varm/cellrank_lineage_drivers"),
        index=genes,
        columns=review.elem(PANCREAS_FATES, "uns/trajectory/cellrank/driver_columns"),
    )
    fig = "figures/trajectory/drivers_heatmap.pdf"
    printed = set(review.pdf(PANCREAS_FATES, fig).split())
    displayed = [g for g in genes if g in printed and g not in lineage]
    hormone = {"Beta": "Ins1", "Delta": "Sst", "Alpha": "Gcg"}
    ranks = {
        g: int(driver[fate + "_corr"].rank(ascending=False)[g])
        for fate, g in hormone.items()
        if fate in lineage and g in genes
    }
    review.record(
        "trajectory_viz.driver_viz",
        PANCREAS_FATES,
        LIMITED,
        "Compare candidate drivers with canonical hormone markers and check finite lineage"
        " scores for displayed genes.",
        f"{len(displayed)} recognizable gene labels,"
        f" {driver.loc[displayed, [fate + '_corr' for fate in lineage]].notna().sum().sum()} finite"
        f" correlations. Hormone ranks among assayed genes per fate {ranks}. Associations"
        " lack perturbational evidence.",
        [fig, FINAL],
    )
    # With an intercept and binary GMT membership ULM t scores are monotone in the
    # per-cell correlation of expression with membership. Only ranks enter Spearman.
    gmts = {
        s.split("\t")[0]: set(s.split("\t")[2:])
        for s in (review.base / "pancreas/mh.all.v2025.1.Mm.symbols.gmt").read_text().splitlines()
    }
    keep = np.asarray(norm.getnnz(axis=0) > 0).ravel()
    x = norm[:, keep].astype(float)
    mean = np.asarray(x.mean(axis=1)).ravel()
    sd = np.sqrt(np.maximum(np.asarray(x.power(2).mean(axis=1)).ravel() - mean**2, 0))
    correlations = {}
    for name, members in gmts.items():
        ix = var.index[keep].isin(members)
        if ix.sum() < 5:
            continue
        mean_set = np.asarray(x[:, ix].mean(axis=1)).ravel()
        standardized = np.divide(mean_set - mean, sd, out=np.zeros_like(sd), where=sd > 0)
        correlations[name] = stats.spearmanr(standardized, po.palantir_pseudotime).statistic
    fig = "figures/trajectory/activity_cascade_hallmark.pdf"
    lines = (
        subprocess.check_output(
            ["pdftotext", "-layout", str(review.base / PANCREAS_LINEAGE / fig), "-"]
        )
        .decode()
        .splitlines()
    )
    measured = []
    for line in lines:
        m = re.match(r"^\s*([\w -]+?)\s+([↑↓])\s*(\d+\.\d+)", line)
        if m:
            name = "HALLMARK_" + m[1].strip().upper().replace(" ", "_").replace("-", "_")
            if name in correlations:
                measured.append(
                    (name, correlations[name], float(m[3]) * (1 if m[2] == "↑" else -1))
                )
    e = max((abs(a - b) for _, a, b in measured), default=np.inf)
    pd.DataFrame(measured, columns=["pathway", "independent_rho", "pdf_rho"]).to_csv(
        review.out / "cascade-correlations.csv", index=False
    )
    review.test(
        "trajectory_viz.activity_cascade",
        PANCREAS_LINEAGE,
        len(measured) == len(re.findall("[↑↓]", review.pdf(PANCREAS_LINEAGE, fig)))
        and len(measured) > 0
        and e < 0.011,
        "Recompute signed gene-set correlations from normalized cells, mouse GMT membership"
        " and Palantir time. Compare the coefficients printed on the PDF.",
        f"{len(measured)} labeled pathways compared; maximum absolute difference from"
        f" printed rho {e:.3g}. Activity follows inferred time in this dataset.",
        [fig, FINAL],
    )


def image_agreement(fig: Figure, path: Path, dpi: int) -> float:
    """Compare colored/inked pixels of two renders, excluding shared white margins."""
    from io import BytesIO

    import matplotlib.pyplot as plt
    from PIL import Image

    result = BytesIO()
    try:
        fig.savefig(result, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    result.seek(0)

    def pixels(src: Path | BinaryIO) -> np.ndarray:
        with Image.open(src) as img:
            return (
                np.asarray(
                    img.convert("RGB").resize((360, 360), Image.Resampling.LANCZOS),
                    dtype=float,
                )
                / 255
            )

    orig, rebuilt = pixels(path), pixels(result)
    ink = (orig.min(axis=2) < 0.95) | (rebuilt.min(axis=2) < 0.95)
    return float(np.abs(orig[ink] - rebuilt[ink]).mean())


def communication(review: Review) -> None:
    from cellquorum.stages.cell_cell_communication.viz import _plots
    from cellquorum.visualization.figstyle import apply_theme

    ranks = review.csv(SCC, "results/cell_cell_communication/liana_ranks.csv")
    carried = review.elem(SCC_NETWORK, "uns/liana_res")
    ids = ["sample", "source", "target", "ligand_complex", "receptor_complex"]
    compare = ranks.merge(carried, on=ids, validate="one_to_one")
    assert len(compare) == len(ranks) == len(carried)
    assert np.allclose(compare.magnitude_rank_x, compare.magnitude_rank_y)
    canon = (
        ranks[ids]
        .rename(columns={"ligand_complex": "ligand", "receptor_complex": "receptor"})
        .copy()
    )
    canon["weight"] = 1 - ranks.magnitude_rank
    pair = canon.groupby(["source", "target"]).weight.sum().sort_values(ascending=False)
    raw_o = review.elem("scc", "obs", "scc.h5ad")
    raw_var = review.elem("scc", "var", "scc.h5ad")
    raw_x = review.elem("scc", "X", "scc.h5ad")
    palette = _plots.celltype_palette(sorted(canon.source.unique()))
    edges_path = "results/ccc_network/curvature_cci_edges.csv"
    nodes_path = "results/ccc_network/curvature_cci_nodes.csv"
    edges = review.csv(SCC_NETWORK, edges_path)
    nodes = review.csv(SCC_NETWORK, nodes_path)
    # Rebuild five panels using the separate *primary* LIANA export, bypassing
    # the network run's canonicalization and file routing.
    top = 15
    dpi = json.loads((review.base / SCC_NETWORK / "provenance/resolved_config.json").read_text())[
        "ccc_viz"
    ]["dpi"]
    panels = [
        (
            "dotplot_viz",
            "dotplot_liana",
            lambda: _plots.interaction_dotplot(canon, top_k=top),
        ),
        (
            "chord_viz",
            "chord_liana",
            lambda: _plots.chord_diagram(canon, palette=palette, top_k=top),
        ),
        (
            "sankey_viz",
            "sankey_liana",
            lambda: _plots.sankey_flow(canon, palette=palette, top_k=top),
        ),
        (
            "summary_viz",
            "heatmap_liana",
            lambda: _plots.cci_heatmap(canon, diverging=False),
        ),
        (
            "network_viz",
            "network_cci",
            lambda: _plots.curvature_network(edges, nodes, top_k=top),
        ),
    ]
    best = pair.index[0]
    strongest = (
        canon.assign(lr=canon.ligand + "->" + canon.receptor).groupby("lr").weight.sum().idxmax()
    )
    bylib = (
        canon.groupby(["sample", "source", "target"])
        .weight.sum()
        .unstack(["source", "target"], fill_value=0)
    )
    lib_wins = int((bylib[best] == bylib.max(axis=1)).sum())
    unique = canon.sort_values("weight", ascending=False, kind="stable").drop_duplicates(
        ["source", "target"]
    )
    union = {x for row in unique.itertuples() for x in [row.ligand, row.receptor]}
    edge_error = max(
        abs(
            row.weight
            - max(pair.get((row.source, row.target), 0), pair.get((row.target, row.source), 0))
        )
        for row in edges.itertuples()
    )
    for key, stem, render in panels:
        apply_theme("notebook")
        image = f"figures/ccc/{stem}.png"
        pdf = f"figures/ccc/{stem}.pdf"
        err = image_agreement(render(), review.base / SCC_NETWORK / image, dpi)
        txt = review.pdf(SCC_NETWORK, pdf)
        paths = [pdf, image, FINAL]
        value = True
        if key == "dotplot_viz":
            names = (
                canon.assign(lr=canon.ligand + "->" + canon.receptor)
                .groupby("lr")
                .weight.sum()
                .nlargest(top)
                .index
            )
            hit = sum(g in txt for g in names)
            result = (
                f"{hit}/{top} independently ranked interaction names appear in PDF; highest"
                f" aggregate {strongest}."
            )
            value = hit == top
        elif key == "network_viz":
            result = (
                f"{len(edges)} plotted interpopulation edges; maximum weight error against"
                f" the stronger directed source edge {edge_error:.2g}. Curvature panel is"
                " undirected."
            )
            value = edge_error < 1e-6
            paths.extend([edges_path, nodes_path])
        elif key == "sankey_viz":
            hit = sum(x in txt.split() for x in union)
            result = (
                f"{hit}/{len(union)} unique ligand/receptor labels of the strongest flow"
                f" per population pair visible (panel is capped at {top} routes)."
            )
        else:
            result = (
                f"Across {len(bylib)} libraries strongest pooled route {best[0]} to"
                f" {best[1]}, sum {pair.iloc[0]:.2f}; strongest within"
                f" {lib_wins}/{len(bylib)} libraries."
            )
        # Pixel disagreement can reflect a different renderer. Never call it a biological failure.
        verdict = FAILED if not value else SUPPORTED if err <= 0.04 else LIMITED
        review.record(
            "ccc_viz." + key,
            SCC_NETWORK,
            verdict,
            "Compare the source-derived interaction summary with the saved panel and"
            " reconstruct the representative PNG (mean ink-pixel error ≤0.04 on a 0–1"
            " scale).",
            result + f" Re-render error {err:.4f}. Weights summarize inferred RNA interactions.",
            paths,
        )
    audit = review.csv(SCC, "results/mnn_pairing_audit.csv")
    paired_audit = audit[audit.included_paired_analysis]
    # Compare ligand and receptor expression in the original paired specimens.
    lr = marker_frame(raw_x, raw_var.index, ["TGFB1", "ENG"], raw_o)
    sent = (
        lr.TGFB1[raw_o.celltype_alt == "KC_other"]
        .gt(0)
        .groupby(raw_o.sample_id, observed=True)
        .mean()
    )
    recv = (
        lr.ENG[raw_o.celltype_alt == "Fibroblast"]
        .gt(0)
        .groupby(raw_o.sample_id, observed=True)
        .mean()
    )
    sample_ids = paired_audit.sample_id
    both = (sent.reindex(sample_ids).gt(0) & recv.reindex(sample_ids).gt(0)).sum()
    meta = review.csv(SCC, "results/mnn_sample_prioritization.csv")
    cd = (
        meta[(meta.receiver == "Fibroblast") & (meta.receptor == "CD63")]
        .groupby("sample")
        .pb_receptor.first()
    )
    published = pd.Series({"P6Normal": 12.3, "P10Tumor": 11.8, "P10Normal": 11.7, "P2Tumor": 11.7})
    err = (cd.reindex(published.index) - published).abs().max()
    figures = []
    mnn = review.csv(SCC, "results/mnn_prioritization.csv")
    hits = []
    for group, tab in mnn.groupby("group"):
        fig = f"figures/ccc/multinichenet_top_lr_{group}.pdf"
        figures.append(fig)
        top_pairs = tab.nlargest(30, "prioritization_score")
        text = review.pdf(SCC, fig)
        hits.extend(f"{r.ligand} - {r.receptor}" in text for r in top_pairs.itertuples())
    result = (
        f"{sum(hits)}/{len(hits)} top scored route rows have their LR pair label in the"
        " correct group PDF; raw sender TGFB1 and receiver ENG both detected in"
        f" {both}/{len(sample_ids)} specimens ({paired_audit.patient.nunique()} pairs); max"
        f" difference from four published CD63 values {err:.3f} log units."
    )
    for key in ["ccc_viz", "ccc_viz.multinichenet_viz"]:
        review.test(
            key,
            SCC,
            all(hits) and both == len(sample_ids) and err < 0.25,
            "Rebuild the two top-30 lists, check paired expression in original SCC counts"
            " and compare four receiver values with the vignette.",
            result,
            [
                *figures,
                "results/mnn_prioritization.csv",
                "results/mnn_pairing_audit.csv",
                "results/mnn_sample_prioritization.csv",
            ],
        )
    activity = review.csv(SCC, "results/nichenet_activities.csv").sort_values(
        "aupr_corrected", ascending=False
    )
    lig = activity.iloc[0].test_ligand
    expr = review.csv(SCC, "results/nichenet_sender_expression.csv")
    orig = raw_o.celltype_alt.eq("KC_other")
    frac = (
        float(lr.loc[orig, lig].gt(0).mean())
        if lig in lr
        else float(marker_frame(raw_x, raw_var.index, [lig], raw_o).loc[orig, lig].gt(0).mean())
    )
    row = expr[(expr.sender == "KC_other") & (expr.ligand == lig)].iloc[0]
    fig = "figures/ccc/nichenet_ligand_activity.pdf"
    printed = review.pdf(SCC, fig).split()
    top20 = activity.head(20).test_ligand.tolist()
    hits = sum(g in printed for g in top20)
    review.test(
        "ccc_viz.nichenet_viz",
        SCC,
        hits == len(top20) and abs(frac - row.fraction_expressing) < 1e-6,
        "Check the ligand ranking in the PDF and independently measure detection of the top"
        " ligand in its source population.",
        f"{hits}/20 top ligands labeled; {lig} ranks first (corrected AUPR"
        f" {activity.iloc[0].aupr_corrected:.3f}); original keratinocyte detection"
        f" {frac:.1%}, {orig.sum():,} cells. Protein signaling is unmeasured.",
        [fig, "results/nichenet_activities.csv", "results/nichenet_sender_expression.csv"],
    )
    path = "results/cell_cell_communication/tensor_factors_contexts.csv"
    fac = review.csv(SCC, path, index_col=0)
    linked = review.csv(SCC, "reports/publication/communication.tensor.csv").set_index("sample")
    err = (linked.value - fac.iloc[:, 0].reindex(linked.index)).abs().max()
    fig = "figures/ccc/tensor_factors.pdf"
    text = review.pdf(SCC, fig)
    meta = raw_o[["sample_id", "patient", "tum.norm"]].drop_duplicates().set_index("sample_id")
    merged = fac.join(meta)
    paired_fac = merged.pivot(index="patient", columns="tum.norm", values=fac.columns[0]).dropna()
    delta = paired_fac.Tumor - paired_fac.Normal
    review.test(
        "ccc_viz.tensor_factors_viz",
        SCC,
        err < 1e-7 and set(fac.index) == set(bylib.index) and all(s in text for s in fac.index),
        "Verify every tensor context against LIANA and source specimen IDs, all PDF sample"
        " labels, and the publication panel's first factor.",
        f"{len(fac)} contexts, {fac.shape[1]} factors, max first-factor error {err:.3g}."
        f" {sum(delta > 0)}/{len(delta)} complete patient pairs had higher tumor factor 1"
        " (descriptive).",
        [fig, path, "reports/publication/communication.tensor.csv"],
    )


def corrected_heatmap(review: Review) -> None:
    from types import SimpleNamespace

    from cellquorum.stages.trajectory.viz._pseudotime_plots import (
        PseudotimeHeatmapVizMethod,
    )

    config = json.loads(
        (review.base / PANCREAS_LINEAGE / "provenance/resolved_config.json").read_text()
    )["trajectory_viz"]
    obj = ad.read_h5ad(review.base / PANCREAS_LINEAGE / FINAL, backed="r")
    try:
        ctx = SimpleNamespace(paths=SimpleNamespace(figures=review.out / "figures"))
        result = PseudotimeHeatmapVizMethod().run(obj, config, ctx)
        if not getattr(result, "artifacts", []):
            raise ValueError(f"Corrected heatmap did not render: {result}")
    finally:
        obj.file.close()
    fig = review.out / "figures/trajectory/pseudotime_heatmap.pdf"
    if "rank within condition (percentile)" not in " ".join(
        subprocess.check_output(["pdftotext", "-layout", str(fig), "-"]).decode().split()
    ):
        raise ValueError("Corrected PDF is missing the percentile axis")


def comparison_figure(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.1), width_ratios=[1.65, 1, 1], layout="constrained"
    )
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    a, b, c = axes
    colors = ["#216986", "#BB572C"]
    for method, color, delta, name in [
        ("marker_vote", colors[0], -0.14, "Marker voting"),
        ("celltypist", colors[1], 0.14, "CellTypist"),
    ]:
        tab = pd.read_csv(out / f"{method}-confusion.csv", index_col=0)
        recall = (
            np.array([tab.loc[g, g] / tab.loc[g].sum() if g in tab else 0 for g in tab.index]) * 100
        )
        a.scatter(recall, np.arange(len(tab)) + delta, color=color, label=name, s=28, zorder=3)
    a.set(
        yticks=np.arange(len(tab)),
        yticklabels=tab.index,
        xlim=(-4, 104),
        xlabel="Recall against supplied labels (%)",
    )
    a.invert_yaxis()
    a.legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=2, frameon=False)
    d = pd.read_csv(out / "interferon-donor-differences.csv", index_col=0).aucell.sort_values()
    labels = [str(i).removeprefix("patient_") for i in d.index]
    b.hlines(np.arange(len(d)), 0, d, color="#9ABBC5")
    b.scatter(d, np.arange(len(d)), color=colors[0])
    b.set(
        yticks=np.arange(len(d)),
        yticklabels=labels,
        xlabel="AUCell score change (stim − ctrl)",
        ylabel="Donor",
        xlim=(0, max(d) * 1.12),
    )
    f = pd.read_csv(out / "endocrine-fate-recall.csv", index_col=0).iloc[:, 0] * 100
    bars = c.barh(f.index, f, color=colors[0])
    c.bar_label(bars, labels=[f"{u:.0f}%" for u in f], padding=4, fontsize=9)
    c.set(xlim=(0, 118), xlabel="Matching highest fate probability (%)")
    c.invert_yaxis()
    for ax, title in zip(
        axes,
        ["A  PBMC annotation", "B  Interferon response", "C  Endocrine fate"],
        strict=True,
    ):
        ax.set_title(title, loc="left", weight="bold", fontsize=13, pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.16)
        ax.set_axisbelow(True)
    for fmt in ["svg", "png"]:
        fig.savefig(out / f"scientific-validation.{fmt}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def publish(review: Review, catalog: Path) -> None:
    import os
    from html import escape
    from urllib.parse import quote

    df = pd.DataFrame(review.rows).sort_values(["stage", "method"])
    key = ["stage", "method"]
    if df.duplicated(key).any():
        raise ValueError("Duplicate scientific verdict")
    coverage = pd.read_csv(catalog)
    # Supports both the pre-review and the reviewed release matrix.
    target = coverage.status.eq("executed")
    if "review_verdict" in coverage:
        target |= coverage.review_verdict.notna()
    if set(map(tuple, df[key].to_numpy())) != set(map(tuple, coverage.loc[target, key].to_numpy())):
        raise ValueError("Review does not match the catalog's execution-only cohort")
    from study_coverage import evidence_digest, file_digest

    inputs_by_run = {
        KANG: ["kang/data/kang_2018.h5ad"],
        KANG_SENSITIVITY: ["kang/data/kang_2018.h5ad"],
        KANG_EXPRESSION: [
            "kang/data/kang_2018.h5ad",
            "kang/data/h.all.v2025.1.Hs.symbols.gmt",
        ],
        KANG_SUBCLUSTERS: ["kang/data/kang_2018.h5ad"],
        PANCREAS_ATLAS: ["pancreas/integration.h5ad"],
        PANCREAS_DPT: ["pancreas/endocrinogenesis.h5ad"],
        PANCREAS_LINEAGE: [
            "pancreas/endocrinogenesis.h5ad",
            "pancreas/mh.all.v2025.1.Mm.symbols.gmt",
        ],
        PANCREAS_FATES: ["pancreas/endocrinogenesis.h5ad"],
        SCC: ["scc/scc.h5ad"],
        SCC_NETWORK: [
            "scc/scc.h5ad",
            SCC + "/results/cell_cell_communication/liana_ranks.csv",
        ],
        PBMC_DROPLETS: [
            "pbmc_droplets/pbmc_1k_v3/outs/raw_feature_bc_matrix.h5",
            "pbmc_droplets/pbmc_1k_v3/outs/filtered_feature_bc_matrix.h5",
        ],
    }
    df["study_inputs"] = [" | ".join(inputs_by_run[r]) for r in df.run]
    input_digests = {}
    for name in sorted({n for names in inputs_by_run.values() for n in names}):
        path = review.base / name
        stat = path.stat()
        input_digests[name] = file_digest(path, stat.st_size, stat.st_mtime_ns)
    hashes = [f"{value}  {name}" for name, value in input_digests.items()]
    df["study_input_sha256"] = [
        json.dumps({p: input_digests[p] for p in inputs_by_run[run]}, sort_keys=True)
        for run in df.run
    ]
    (review.out / "study-inputs.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    df["evidence_sha256"] = [
        evidence_digest(review.base / r.run, r.stage, r.evidence.split(" | "))
        for r in df.itertuples()
    ]
    df.to_csv(review.out / "scientific-validation.csv", index=False)
    comparison_figure(review.out)
    paths = [
        "scientific-validation.csv",
        "study-inputs.sha256",
        "interferon-donor-differences.csv",
        "marker_vote-confusion.csv",
        "celltypist-confusion.csv",
        "endocrine-fate-recall.csv",
        "cascade-correlations.csv",
        "scientific-validation.svg",
        "figures/trajectory/pseudotime_heatmap.pdf",
    ]
    links = " · ".join(f'<a href="{escape(p)}">{escape(Path(p).name)}</a>' for p in paths)

    def link(path: Path) -> str:
        href = quote(Path(os.path.relpath(path, review.out)).as_posix(), safe="/.")
        return f'<a href="{href}">{escape(path.name)}</a>'

    parts = [
        (
            '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport"'
            ' content="width=device-width"><title>CellQuorum scientific'
            " validation</title><style>body{margin:2rem auto;padding:0"
            " 1rem;max-width:1050px;font:16px/1.5"
            " system-ui;color:#183445}img{width:100%}article{padding:1rem 0;border-top:1px"
            " solid #c7d1d5}p{max-width:90ch}a{color:#145d7e}nav"
            " a{overflow-wrap:anywhere}h3{margin-bottom:.4rem}</style><main><h1>Scientific"
            " validation on public studies</h1>"
        ),
        (
            "<p>Each verdict applies to the stated criterion on the named study. Supported"
            " means the measured comparison agreed. Limited means the data support only"
            " part of the intended inference. Contradicted means the actual result"
            " disagreed with the tested claim. Figure checks apply to the illustrated"
            " quantities and representative panels identified below.</p>"
        ),
        (
            '<img alt="PBMC label recall, paired donor interferon responses, and endocrine'
            ' fate agreement" src="scientific-validation.png">'
        ),
        f"<p>{links}</p>",
    ]
    for r in df.itertuples():
        paths = [review.base / r.run / p for p in r.evidence.split(" | ")] + [
            review.base / p for p in r.study_inputs.split(" | ")
        ]
        name = r.stage if r.method == "(stage)" else r.stage + "." + r.method
        parts.append(
            f"<article><h3>{escape(name)} ·"
            f" {r.verdict}</h3><p>{escape(r.criterion)}</p><p>{escape(r.result)}</p>"
            f"<p>{escape(r.run)}</p><nav>" + " · ".join(map(link, paths)) + "</nav></article>"
        )
    parts.append("</main></html>")
    (review.out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(
        df.groupby([df.method.eq("(stage)").map({True: "stages", False: "methods"}), "verdict"])
        .size()
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    review = Review(args.data.resolve(), args.output.resolve())
    obs, ss, var, norm = check_pbmc(review)
    check_pbmc_contrasts(review, obs, ss, var, norm)
    pathways(review, obs, var, norm)
    del norm
    droplets(review)
    trajectory(review)
    communication(review)
    corrected_heatmap(review)
    publish(review, args.coverage)


if __name__ == "__main__":
    main()
