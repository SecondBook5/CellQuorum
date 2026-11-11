"""Recompute paired SCC rank evidence from the matched LIANA library exports."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, false_discovery_control

ROOT = Path(__file__).resolve().parents[1] / "docs/benchmarks/pandas-anndata-review"
KEY = ["source", "target", "ligand_complex", "receptor_complex"]


def main() -> None:
    ranks = pd.read_csv(ROOT / "scc-magnitude-interactions.csv.gz")
    libraries = (
        pd.read_csv(ROOT / "scc-panel-measurements.csv")[["sample", "donor", "group"]]
        .dropna()
        .drop_duplicates()
    )
    ranks = ranks.merge(libraries, on="sample", how="left", validate="many_to_one")
    if ranks["donor"].isna().any():
        raise ValueError("Missing donor metadata for scored interaction")
    values = ["base_percentile", "next_percentile"]
    paired = ranks.pivot(index=[*KEY, "donor"], columns="group", values=values).dropna()
    differences = paired.xs("Tumor", axis=1, level="group") - paired.xs(
        "Normal", axis=1, level="group"
    )
    groups = differences.groupby(KEY)
    summary = pd.DataFrame({"pairs": groups.size()})
    for name, column in zip(("baseline", "candidate"), values, strict=True):
        summary[name + "_median_delta_pp"] = groups[column].median()
        stronger = differences[column].lt(0).groupby(KEY).sum()
        weaker = differences[column].gt(0).groupby(KEY).sum()
        summary[name + "_tumor_stronger"] = stronger
        summary[name + "_tumor_weaker"] = weaker
        summary[name + "_top100_libraries"] = ranks.groupby(KEY)[
            ("base" if name == "baseline" else "next") + "_top100"
        ].sum()
        evaluable = summary["pairs"].ge(5)
        p = [
            binomtest(s, s + w).pvalue if s + w else 1.0
            for s, w in zip(stronger[evaluable], weaker[evaluable], strict=True)
        ]
        summary.loc[evaluable, name + "_sign_test_bh"] = false_discovery_control(p)
    summary.to_csv(
        ROOT / "scc-paired-rank-summary.csv.gz", compression={"method": "gzip", "mtime": 0}
    )
    differences.to_csv(
        ROOT / "scc-paired-rank-donors.csv.gz", compression={"method": "gzip", "mtime": 0}
    )
    subset = summary[summary.pairs >= 5]
    flip = subset.baseline_median_delta_pp * subset.candidate_median_delta_pp < 0
    print("All measured combinations", ranks.groupby(KEY).ngroups, "paired", len(summary))
    print("At least five pairs", len(subset), "median changes sign", flip.sum())
    print(
        "Median zero in either version",
        ((subset.baseline_median_delta_pp == 0) | (subset.candidate_median_delta_pp == 0)).sum(),
    )
    for name in ("baseline", "candidate"):
        print(name, "minimum adjusted p", subset[name + "_sign_test_bh"].min())
    print(
        subset[flip]
        .assign(
            shift=np.minimum(
                subset.baseline_median_delta_pp.abs(), subset.candidate_median_delta_pp.abs()
            )
        )
        .nlargest(5, "shift")
        .to_string()
    )
    plot_donor_changes(differences)


def plot_donor_changes(differences: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = pd.read_csv(ROOT / "scc-publication-donor-differences.csv").set_index("donor")
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.3), layout="constrained")
    rank_columns = ["baseline_tumor_minus_normal", "candidate_tumor_minus_normal"]
    examples = [
        (
            "A   TGFB1 → ENG",
            "Keratinocyte to fibroblast",
            panels.loc[panels.panel == "communication.liana", rank_columns],
            "Magnitude score",
        ),
        (
            "B   Tensor factor 1",
            "Matched sample loadings",
            panels.loc[panels.panel == "communication.tensor", rank_columns],
            "Factor loading",
        ),
        (
            "C   COL18A1 → ITGB5",
            "Fibroblast to fibroblast",
            differences.loc[("Fibroblast", "Fibroblast", "COL18A1", "ITGB5")],
            "Rank percentile",
        ),
        (
            "D   NAMPT → ITGA5–ITGB1",
            "Keratinocyte to fibroblast",
            differences.loc[("KC_other", "Fibroblast", "NAMPT", "ITGA5_ITGB1")],
            "Rank percentile",
        ),
    ]
    colors = ("#376783", "#AD4B32")
    for ax, (title, subtitle, frame, units) in zip(axes.flat, examples, strict=True):
        data = frame.loc[sorted(frame.index, key=lambda p: int(p[1:]))]
        positions = np.arange(len(data))
        ax.axvline(0, color="#959595", lw=0.8, ls="--")
        ax.hlines(positions, data.iloc[:, 0], data.iloc[:, 1], color="#8a8a8a", lw=1.1)
        ax.scatter(
            data.iloc[:, 0], positions, facecolor="white", edgecolor=colors[0], s=48, zorder=3
        )
        ax.scatter(data.iloc[:, 1], positions, color=colors[1], s=15, zorder=4)
        ax.set_yticks(positions, data.index)
        ax.set_ylim(len(data) - 0.45, -0.65)
        ax.set_title(title + "\n" + subtitle, loc="left", fontsize=10)
        ax.set_xlabel(units + " difference (tumor − normal)", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(labelsize=8)
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=colors[0],
            label="LIANA 1.10.0",
        ),
        Line2D([], [], ls="", marker="o", color=colors[1], label="LIANA 2.1.1"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    for suffix in ("png", "svg"):
        fig.savefig(ROOT / ("scc-paired-rank-changes." + suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
