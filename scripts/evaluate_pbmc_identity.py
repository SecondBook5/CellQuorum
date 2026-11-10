"""Evaluate frozen RNA calls in Kang and Hao against published labels and Hao surface proteins."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.mixture import GaussianMixture

CD8, NK = "CD8 T cells", "NK cells"
CLASSES = [CD8, NK]
METHODS = {"cluster_vote": "Cluster vote", "call": "Refined"}


def read_calls(study_dir: Path, study: str) -> tuple[pd.DataFrame, str]:
    run = study_dir / "runs" / f"{study}_pbmc_identity"
    records = json.loads((run / "provenance/stage_execution_records.json").read_text())
    assert all(r["status"] in {"success", "skipped"} for r in records)
    prediction = run / "results/annotation/marker_vote_label_evidence.csv"
    summary = next(
        r["metrics"]["pbmc_refinement"] for r in records if r["stage_name"] == "annotation"
    )
    cells = pd.read_csv(prediction, index_col="barcode")
    assert cells.index.is_unique and len(cells) == summary["n_cells"]
    assert (cells.call == "Unassigned").sum() == summary["uncertain"]
    with h5py.File(run / "objects/final_annotated.h5ad") as handle:
        obs = ad.io.read_elem(handle["obs"])
    for column, key in {
        "call": "marker_vote_label",
        "cluster_vote": "marker_vote_label_cluster_vote",
        "reason": "marker_vote_label_reason",
    }.items():
        if not cells.index.equals(obs.index) or not np.array_equal(cells[column], obs[key]):
            raise ValueError(f"{study}: exported {column} differs from the final object")
    receipt = hashlib.sha256(prediction.read_bytes()).hexdigest()
    return cells, receipt


def source_metadata(study_dir: Path, study: str) -> pd.DataFrame:
    if study == "hao":
        meta = pd.read_csv(study_dir / "source/GSE164378_sc.meta.data_3P.csv.gz", index_col=0)
        meta["reference"] = meta["celltype.l1"].replace({"CD8 T": CD8, "NK": NK})
        return meta
    meta = ad.read_h5ad(study_dir / "data/kang_2018.h5ad").obs
    return meta.rename(columns={"cell_type": "reference", "replicate": "donor"})


def evaluate(cells: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows = []
    for donor, group in [("All", cells), *cells.groupby("donor", observed=True)]:
        for name, method in METHODS.items():
            for label in CLASSES:
                actual = group[reference] == label
                called = group[name] == label
                tp = int((actual & called).sum())
                rows.append(
                    dict(
                        donor=donor,
                        method=method,
                        label=label,
                        true_positive=tp,
                        reference=int(actual.sum()),
                        called=int(called.sum()),
                        recall=tp / actual.sum() if actual.any() else np.nan,
                        precision=tp / called.sum() if called.any() else np.nan,
                        abstentions=int((actual & (group[name] == "Unassigned")).sum()),
                    )
                )
    return pd.DataFrame(rows)


def protein_gates(root: Path, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = root / "source"
    prefix = "GSM5008738_ADT_3P-"
    genes = pd.Index(pd.read_csv(source / (prefix + "features.tsv.gz"), sep="\t", header=None)[1])
    barcodes = pd.Index(pd.read_csv(source / (prefix + "barcodes.tsv.gz"), header=None)[0])
    assert genes.is_unique and barcodes.is_unique and set(barcodes) == set(meta.index)
    markers = ["CD3-1", "CD8a", "CD56-1", "CD14", "CD19"]
    idx = genes.get_indexer(markers)
    assert (idx >= 0).all()
    with gzip.open(source / (prefix + "matrix.mtx.gz"), "rb") as handle:
        mat = mmread(handle).T.tocsr()
    total = np.asarray(mat.sum(axis=1)).ravel()
    x = np.log1p(mat[:, idx].toarray() * 10000.0 / np.maximum(total, 1)[:, None])
    post = pd.DataFrame(np.nan, columns=markers, index=barcodes)
    fit = []
    for lane, ids in meta.groupby("lane").groups.items():
        positions = barcodes.get_indexer(ids)
        for j, marker in enumerate(markers):
            v = x[positions, j, None]
            model = GaussianMixture(2, random_state=42, n_init=3).fit(v)
            if not model.converged_:
                raise ValueError(f"Protein mixture failed for lane {lane}, {marker}")
            high = int(model.means_.argmax())
            post.loc[ids, marker] = model.predict_proba(v)[:, high]
            fit.append(
                dict(
                    lane=lane,
                    marker=marker,
                    low_mean=model.means_.min(),
                    high_mean=model.means_.max(),
                    high_weight=model.weights_[high],
                )
            )
    gate = pd.Series("Outside either gate", index=barcodes, name="protein_gate")
    gate[(post["CD3-1"] >= 0.9) & (post["CD8a"] >= 0.9)] = CD8
    gate[
        (post["CD3-1"] <= 0.1)
        & (post["CD56-1"] >= 0.9)
        & (post["CD14"] <= 0.1)
        & (post["CD19"] <= 0.1)
    ] = NK
    gate[total == 0] = "Outside either gate"
    return post.join(gate), pd.DataFrame(fit)


def draw(metrics: pd.DataFrame, cells: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"Cluster vote": "#80929d", "Refined": "#126469"}
    fig, axes = plt.subplots(
        2, 2, figsize=(8.5, 6.1), sharex=True, sharey=True, layout="constrained"
    )
    for row, metric in enumerate(["recall", "precision"]):
        for col, study in enumerate(["Kang", "Hao"]):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(
                    study + ("  ·  development" if col == 0 else "  ·  held out"), loc="left"
                )
            for i, label in enumerate(CLASSES):
                subset = metrics[(metrics.study == study) & (metrics.label == label)]
                for j, method in enumerate(METHODS.values()):
                    block = subset[subset.method == method]
                    donors = block[block.donor != "All"][metric].dropna().to_numpy() * 100
                    pooled = block[block.donor == "All"][metric].iloc[0] * 100
                    pos = i + (j - 0.5) * 0.37
                    if np.isfinite(pooled):
                        ax.scatter(
                            pos + np.linspace(-0.055, 0.055, len(donors)),
                            donors,
                            color=colors[method],
                            s=15,
                            alpha=0.55,
                        )
                        ax.plot([pos - 0.1, pos + 0.1], [pooled] * 2, lw=3, color=colors[method])
                    else:
                        ax.text(pos, 8, "no calls", rotation=90, ha="center", fontsize=8)
            ax.set(
                xlim=(-0.5, 1.5),
                ylim=(-4, 104),
                xticks=[0, 1],
                xticklabels=["CD8 T", "NK"],
                yticks=[0, 25, 50, 75, 100],
            )
            ax.grid(axis="y", color="#e5ebeb", lw=0.7)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(metric.capitalize() + " (%)")
    for method, color in colors.items():
        axes[0, 0].plot([], [], lw=3, color=color, label=method)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out / f"cell_identity.{ext}", dpi=175)
    plt.close(fig)

    hao = cells[cells.study == "Hao"]
    groups = [
        ("All retained", hao),
        ("Published CD8 T", hao[hao.reference == CD8]),
        ("Published NK", hao[hao.reference == NK]),
    ]
    rows = []
    for label, group in groups:
        for donor, g in [("All", group), *group.groupby("donor", observed=True)]:
            rows.append(
                dict(
                    group=label,
                    donor=donor,
                    uncertain=int(g.call.eq("Unassigned").sum()),
                    total=len(g),
                    fraction=g.call.eq("Unassigned").mean(),
                )
            )
    sources = pd.DataFrame(rows)
    sources.to_csv(out / "hao_uncertainty.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.6, 3.25), layout="constrained")
    for i, (label, _) in enumerate(groups):
        v = sources[sources.group == label]
        donors = v[v.donor != "All"].fraction * 100
        pooled = v[v.donor == "All"].fraction.iat[0] * 100
        ax.scatter(donors, i + np.linspace(-0.08, 0.08, len(donors)), color="#197f80", s=26)
        ax.scatter([pooled], [i], marker="|", color="#102f36", s=290, linewidths=2.5)
    ax.set(
        yticks=range(3),
        yticklabels=[k for k, _ in groups],
        xlabel="Cells left Unassigned (%)",
        title="Hao   |   Uncertainty by published population",
    )
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e5ebeb")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"hao_uncertainty.{ext}", dpi=175)
    plt.close(fig)


def plot_cd8_subtypes(cells: pd.DataFrame, out: Path) -> None:
    cd8 = cells[(cells.study == "Hao") & (cells.reference == CD8)]
    table = pd.crosstab(cd8["celltype.l2"], cd8.call)
    order = ["CD8 Naive", "CD8 TCM", "CD8 TEM", "CD8 Proliferating"]
    table = table.reindex(order)
    labels = [CD8, "CD4 T cells", NK, "Unassigned"]
    parts = table.reindex(columns=labels, fill_value=0).copy()
    parts["Other"] = table.sum(axis=1) - parts.sum(axis=1)
    parts.to_csv(out / "hao_cd8_fates.csv", index_label="published_subtype")
    totals = parts.sum(axis=1)
    fig, ax = plt.subplots(figsize=(8.3, 3.6), layout="constrained")
    colors = ["#11716C", "#A17745", "#655794", "#D4DBE0", "#3B4850"]
    left = np.zeros(len(parts))
    for (name, values), color in zip(parts.items(), colors, strict=True):
        pct = values / totals * 100
        ax.barh(range(len(parts)), pct, left=left, color=color, height=0.58, label=name)
        left += pct.to_numpy()
    ax.set(
        yticks=range(len(parts)),
        yticklabels=[f"{k}  ·  {v:,}" for k, v in totals.items()],
        xlabel="Refined RNA calls within each published subtype (%)",
        xlim=(0, 100),
        title="Hao   |   Where published CD8 T cells went",
    )
    ax.invert_yaxis()
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=5, frameon=False)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out / f"hao_cd8_fates.{ext}", dpi=175)
    plt.close(fig)


def write_review(cells: pd.DataFrame, metrics: pd.DataFrame, root: Path, out: Path) -> None:
    """Link the post-unblinding comparison and paired transitions to the original runs."""
    from html import escape
    from os.path import relpath

    changes = []
    for (study, donor, label), block in cells[cells.reference.isin(CLASSES)].groupby(
        ["study", "donor", "reference"], observed=True
    ):
        before = block.cluster_vote.eq(label)
        after = block.call.eq(label)
        changes.append(
            dict(
                study=study,
                donor=donor,
                reference=label,
                cells=len(block),
                gained=int((~before & after).sum()),
                lost=int((before & ~after).sum()),
                correct_both=int((before & after).sum()),
                neither=int((~before & ~after).sum()),
            )
        )
    pd.DataFrame(changes).to_csv(out / "paired_transitions.csv", index=False)
    body = [
        "<h1>Can RNA distinguish CD8 T cells from NK cells?</h1>",
        "<p>Kang was used to develop the human PBMC refinement. Hao was held out. "
        "Both calls use the same cells, RNA neighbors and cluster votes within each study. "
        "Every Unassigned cell counts in the published population's recall denominator. "
        "Precision includes false positives from every other population.</p>",
        '<figure><img src="cell_identity.svg" alt="CD8 T and NK recall and precision by study">'
        "<figcaption>Dots show donors. Horizontal bars show pooled cell-level performance. "
        '<a href="cell_identity.pdf">PDF</a> · <a href="cell_identity.svg">Editable SVG</a> · '
        '<a href="identity_metrics.csv">Counts and donor results</a> · '
        '<a href="paired_transitions.csv">Paired gains and losses</a></figcaption></figure>',
    ]
    pooled = metrics[metrics.donor == "All"].copy()
    held = pooled[pooled.study == "Hao"].pivot(index="label", columns="method", values="recall")
    delta = 100 * (held["Refined"] - held["Cluster vote"])
    cd8_verdict = "passed" if delta[CD8] > 0 else "failed"
    nk_verdict = "passed" if delta[NK] >= -2 else "failed"
    body.insert(
        1,
        f"<p><strong>Held-out CD8 recovery {cd8_verdict}.</strong> CD8 recall changed "
        f"by {delta[CD8]:+.2f} percentage points. The two-point NK recall tolerance "
        f"{nk_verdict} with a change of {delta[NK]:+.2f} points.</p>",
    )
    for col in ["precision", "recall"]:
        pooled[col] = pooled[col].map(lambda v: f"{v:.1%}" if np.isfinite(v) else "No calls")
    body.append(
        pooled[
            [
                "study",
                "label",
                "method",
                "reference",
                "called",
                "true_positive",
                "recall",
                "precision",
                "abstentions",
            ]
        ].to_html(index=False)
    )
    body.append(
        '<h2>Uncertain calls</h2><figure><img src="hao_uncertainty.svg" '
        'alt="Unassigned fractions by donor and published population"><figcaption>Each '
        "point is one donor. Vertical ticks show the pooled fraction. "
        '<a href="hao_uncertainty.csv">Source counts</a> · '
        '<a href="hao_uncertainty.svg">Editable SVG</a></figcaption></figure>'
    )
    for study in ["kang", "hao"]:
        path = root / study / "runs" / f"{study}_pbmc_identity" / "reports/report.html"
        body.append(
            f'<p><a href="{escape(relpath(path, out), quote=True)}">'
            f"{study.title()} RNA-only run report and per-cell uncertainty reasons</a></p>"
        )
    body.append(
        '<figure><img src="hao_cd8_fates.svg" alt="Refined RNA calls by published CD8 subtype">'
        "<figcaption>Labels at left give reference cell counts. TCM and TEM denote central "
        'and effector memory. <a href="hao_cd8_fates.csv">Exact source counts</a> · '
        '<a href="hao_cd8_fates.svg">Editable SVG</a></figcaption></figure>'
    )
    body.append(
        "<h2>Measured surface proteins in Hao</h2><p>Each lane has two intensity components "
        "per antibody. CD3-high CD8a-high cells define the CD8 comparison. CD3-low "
        "CD56-high CD14-low CD19-low cells define the NK comparison. All component "
        "assignments require posterior probability ≥0.90. This check uses a subset of the "
        "cells used for the published multimodal labels.</p>"
    )
    gate_counts = pd.read_csv(out / "hao_protein_gate_by_call.csv", index_col=0)
    body.append(gate_counts.to_html())
    body.append(
        '<p><a href="hao_protein_gated_metrics.csv">Donor results within the protein gates</a> · '
        '<a href="hao_protein_gate_by_reference.csv">Gates versus published identities</a> · '
        '<a href="hao_protein_mixtures.csv">Per-lane fits</a></p>'
        '<p><a href="qc_retention.csv">QC retention by donor and identity</a> · '
        '<a href="hao_subtypes_by_call.csv">Published subtypes by RNA call</a> · '
        '<a href="evaluated_cells.csv.gz">Barcode-level evaluation</a></p>'
    )
    (out / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>PBMC identity</title><style>body{max-width:66rem;margin:2rem auto;"
        "padding:0 1rem;font:16px/1.5 system-ui;color:#183942}figure{margin:1rem 0}img{width:100%;"
        "max-width:48rem}table{border-collapse:collapse;font-size:.86rem}td,th{padding:.35rem;"
        "border:1px solid #d8e5e3}a{color:#126469}figcaption{max-width:48rem}</style><main>"
        + "".join(body)
        + "</main></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    # Hash completed predictions before opening any held-out reference labels.
    runs = {s: read_calls(args.reference_root / s, s) for s in ["kang", "hao"]}
    (out / "predictions.sha256").write_text("".join(f"{h}  {s}\n" for s, (_, h) in runs.items()))
    metrics, joined, attrition = [], [], []
    for study, (calls, _) in runs.items():
        directory = args.reference_root / study
        meta = source_metadata(directory, study)
        if not meta.index.is_unique or not calls.index.isin(meta.index).all():
            raise ValueError(f"{study}: barcode mismatch in truth join")
        fields = ["donor", "reference"] + (
            ["celltype.l2", "time", "lane"] if study == "hao" else ["label"]
        )
        cells = calls.join(meta[fields], validate="one_to_one")
        assert cells[["donor", "reference"]].notna().all().all()
        if study == "hao":
            post, fits = protein_gates(directory, meta)
            fits.to_csv(out / "hao_protein_mixtures.csv", index=False)
            cells = cells.join(post, validate="one_to_one")
            gated = cells[cells.protein_gate.isin(CLASSES)]
            evaluate(gated, "protein_gate").to_csv(
                out / "hao_protein_gated_metrics.csv", index=False
            )
            pd.crosstab(cells.protein_gate, cells.call).to_csv(out / "hao_protein_gate_by_call.csv")
            pd.crosstab(cells.protein_gate, cells.reference).to_csv(
                out / "hao_protein_gate_by_reference.csv"
            )
            pd.crosstab(cells["celltype.l2"], cells.call).to_csv(out / "hao_subtypes_by_call.csv")
        excluded = (
            meta.assign(retained=meta.index.isin(calls.index))
            .groupby(["donor", "reference"], observed=True)["retained"]
            .agg(["size", "sum"])
            .reset_index()
            .rename(columns={"size": "input", "sum": "retained"})
        )
        attrition.append(excluded.assign(study=study.title()))
        cells["study"] = study.title()
        for column in METHODS:
            pd.crosstab(cells.reference, cells[column]).to_csv(
                out / f"{study}_{column}_confusion.csv"
            )
        metrics.append(evaluate(cells, "reference").assign(study=study.title()))
        joined.append(cells)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_metrics.to_csv(out / "identity_metrics.csv", index=False)
    pd.concat(attrition).to_csv(out / "qc_retention.csv", index=False)
    all_cells = pd.concat(joined, sort=False)
    all_cells.to_csv(out / "evaluated_cells.csv.gz", index_label="barcode")
    draw(all_metrics, all_cells, out)
    plot_cd8_subtypes(all_cells, out)
    write_review(all_cells, all_metrics, args.reference_root, out)


if __name__ == "__main__":
    main()
