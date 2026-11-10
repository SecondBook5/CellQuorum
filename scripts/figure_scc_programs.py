"""Assemble the public SCC program figure and export the observations behind each panel."""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellquorum.visualization.figstyle import atomic_savefig

GROUPS = ["CD8T", "CLEC9A", "Fibroblast", "KC_other"]
LABELS = ["CD8 T", "CLEC9A DC", "Fibroblast", "Keratinocyte"]
BLUE = "#245477"
ORANGE = "#BE5C35"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    out = args.run / "reports/publication/scc_programs"
    out.mkdir(parents=True, exist_ok=True)
    base = args.run / "results"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), layout="constrained")
    a, b, c, d = axes.flat
    for axis, title in zip(
        axes.flat,
        [
            "A   Consensus expression programs",
            "B   Metacell sizes",
            "C   Coexpression modules",
            "D   Multicellular program in keratinocytes",
        ],
        strict=True,
    ):
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    usage = pd.read_csv(base / "discovery_nmf_usage_by_celltype.csv")
    usage.to_csv(out / "A_nmf.csv", index=False)
    mat = usage.pivot(index="cell_type", columns="program", values="mean_usage").loc[GROUPS]
    im = a.imshow(mat, cmap="Blues", aspect="auto", vmin=0)
    a.set_xticks(range(len(mat.columns)), [col.replace("program_", "P") for col in mat.columns])
    a.set_yticks(range(len(LABELS)), LABELS)
    for (i, j), v in np.ndenumerate(mat.values):
        a.text(
            j,
            i,
            f"{v:.1f}",
            ha="center",
            va="center",
            color="white" if v > mat.values.max() * 0.55 else "#1B3445",
            fontsize=8,
        )
    fig.colorbar(im, ax=a, shrink=0.8, label="Mean usage")
    assignments = pd.read_csv(base / "metacells/metacell_assignments.csv")
    counts = (
        assignments.loc[assignments.metacell_id != "Outliers"]
        .groupby("metacell_id")
        .size()
        .rename("cells")
    )
    counts.to_csv(out / "B_metacell_sizes.csv")
    b.hist(
        counts,
        bins=np.arange((counts.min() // 5) * 5, counts.max() + 6, 5),
        color=BLUE,
        edgecolor="white",
    )
    b.set(xlabel="Cells in a metacell", ylabel="Metacells")
    b.text(
        0.03,
        0.96,
        f"{len(counts)} metacells\n{(assignments.metacell_id=='Outliers').sum()} outlier cells",
        transform=b.transAxes,
        va="top",
    )
    modules = pd.read_csv(base / "coexpression/modules.csv")
    mods = (
        modules.loc[modules.module != "grey"].groupby("module").size().rename("genes").sort_index()
    )
    mods.to_csv(out / "C_module_sizes.csv")
    bars = c.bar(mods.index, mods, color=BLUE)
    c.bar_label(bars, padding=3, fontsize=8)
    c.set(ylabel="Assigned genes", xlabel="hdWGCNA module", ylim=(0, mods.max() * 1.18))
    scores = pd.read_csv(base / "multicellular_programs/mcp_scores.csv")
    stability = pd.read_csv(base / "multicellular_programs/program_stability.csv")
    if len(stability) != 1:
        raise ValueError("Select a program explicitly if this reference yields multiple programs.")
    program = stability.program.iat[0]
    obj = ad.read_h5ad(args.run / "objects/final_annotated.h5ad", backed="r")
    try:
        scores = scores[(scores.cell_type == "KC_other") & (scores.program == program)].join(
            obj.obs[["patient", "tum.norm"]], on="cell_id", validate="many_to_one"
        )
    finally:
        obj.file.close()
    means = (
        scores.groupby(["patient", "tum.norm"], observed=True)
        .score.agg(mean_score="mean", cells="count")
        .reset_index()
    )
    means["program"] = program
    means.to_csv(out / "D_donor_scores.csv", index=False)
    paired = means.pivot(index="patient", columns="tum.norm", values="mean_score")[
        ["Normal", "Tumor"]
    ]
    for _, row in paired.iterrows():
        d.plot([0, 1], row, color="#8494A0", linewidth=1, zorder=1)
    for x, group, color in [(0, "Normal", BLUE), (1, "Tumor", ORANGE)]:
        d.scatter(
            np.full(len(paired), x),
            paired[group],
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
    d.set(
        xticks=[0, 1],
        xticklabels=["Normal", "Tumor"],
        xlim=(-0.28, 1.28),
        ylabel=f"{program} mean cell score",
    )
    d.set_xlabel(
        f"{paired.dropna().shape[0]} pairs  ·  resampling stability "
        f"{stability.mean_stability.iat[0]:.2f}"
    )
    fig.suptitle(
        "Expression programs in paired cutaneous squamous-cell carcinoma",
        fontsize=14,
        fontweight="bold",
    )
    for ext in ["svg", "pdf", "png"]:
        atomic_savefig(fig, out / f"scc_programs.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (out / "legend.txt").write_text(
        f"A. Mean consensus NMF usage across cells of each annotated population. "
        "B. Distribution of cell counts per assigned metacell, excluding outliers. "
        f"C. Number of genes in each non-grey hdWGCNA module. "
        f"D. Each point is the mean DIALOGUE {program} score in keratinocytes from one specimen. "
        f"Lines connect paired specimens. Across {int(stability.n_resamples.iat[0])} resamples, "
        f"mean stability was {stability.mean_stability.iat[0]:.3f}. The panel is descriptive.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
