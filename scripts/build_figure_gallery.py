"""Build deterministic publication examples from explicitly synthetic results."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import anndata as ad  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cellquorum.stages.comparative.differential_expression.viz.plots import volcano  # noqa: E402
from cellquorum.stats.donor_comparison import two_group_test_on_donor_medians  # noqa: E402
from cellquorum.visualization.enrichment import gsea_dotplot, gsea_running_es  # noqa: E402
from cellquorum.visualization.figstyle import (  # noqa: E402
    LE_RED,
    NORMAL_BLUE,
    categorical_embedding,
    render_figure,
    save_figure,
    set_publication_style,
)
from cellquorum.visualization.qc.panels import (  # noqa: E402
    assemble_qc_frame,
    plot_paired_dumbbell,
    write_qc_overview_figure,
)


def _themed(build: Callable[[], Figure], theme: Callable[[], None]) -> Callable[[], Figure]:
    def themed() -> Figure:
        theme()
        return build()

    return themed


# Every ``gallery_<stage>.py`` next to this script exposes ``FIGURES``, a
# {name -> zero-argument builder returning a Figure} registry, and ``THEME``, the
# stage's own theme call (applied first, as the real pipeline does). These figures are
# baselined as PNG only; the six hand-built examples below also ship PDF and SVG.
STAGE_FIGURES = {
    name: _themed(build, module.THEME)
    for module in (
        importlib.import_module(path.stem)
        for path in sorted(Path(__file__).resolve().parent.glob("gallery_*.py"))
    )
    for name, build in module.FIGURES.items()
}


def build_gallery(output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "source": [
                "Matrix organization",
                "Interferon response",
                "Cell adhesion",
                "Oxidative metabolism",
                "Ribosome assembly",
                "Vesicle transport",
            ],
            "score": [2.3, 1.8, 1.2, -2.1, -1.5, -0.8],
            "raw_es": [0.65, 0.51, 0.34, -0.60, -0.42, -0.23],
            "padj": [0.001, 0.012, 0.08, 0.004, 0.03, 0.2],
            "leading_edge_size": [35, 24, 18, 30, 20, 12],
        }
    )
    table.to_csv(output / "enrichment_source.csv", index=False)
    figures: list[Path] = []
    warnings: list[str] = []

    def enrichment() -> list[Path]:
        fig = gsea_dotplot(
            table, case_label="Case", control_label="Control", title="Synthetic enrichment example"
        )
        return save_figure(fig, output, "enrichment", formats=("png", "pdf", "svg"))

    render_figure("enrichment", enrichment, figures=figures, warnings=warnings)
    rng = np.random.default_rng(17)
    coordinates = np.vstack(
        [
            rng.normal(center, 0.35, size=(count, 2))
            for center, count in [((-2, 0), 180), ((1, 1.5), 140), ((1, -1.5), 30)]
        ]
    )
    embedding_table = pd.DataFrame(coordinates, columns=["UMAP1", "UMAP2"])
    embedding_table["population"] = (
        ["Population A"] * 180 + ["Population B"] * 140 + ["Rare population"] * 30
    )
    embedding_table.to_csv(output / "embedding_source.csv", index=False)

    def embedding() -> list[Path]:
        set_publication_style()
        data = ad.AnnData(
            X=np.zeros((len(embedding_table), 1)),
            obs=embedding_table[["population"]].set_axis(
                [f"cell_{i}" for i in range(len(embedding_table))]
            ),
        )
        data.obsm["X_umap"] = coordinates
        fig = categorical_embedding(
            data,
            "population",
            legend=True,
            label_on_plot=False,
            point_size=9,
            title="Synthetic embedding example",
            figsize=(6.2, 4.0),
        )
        return save_figure(fig, output, "embedding", formats=("png", "pdf", "svg"))

    render_figure("embedding", embedding, figures=figures, warnings=warnings)
    paired_table = pd.DataFrame(
        {
            "donor": [f"D{i}" for i in range(1, 7)] * 2,
            "condition": ["Control"] * 6 + ["Case"] * 6,
            "value": [4.0, 5.2, 6.0, 3.5, 7.1, 5.8, 5.2, 6.8, 5.5, 4.3, 9.1, 6.1],
        }
    )
    paired_table.to_csv(output / "paired_source.csv", index=False)
    comparison = two_group_test_on_donor_medians(
        paired_table,
        value_col="value",
        group_col="condition",
        donor_col="donor",
        group1="Control",
        group2="Case",
        paired=True,
    )
    if comparison is None:
        raise RuntimeError("Synthetic paired fixture has insufficient donors")
    (output / "paired_comparison.json").write_text(
        json.dumps(comparison.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    def paired() -> list[Path]:
        set_publication_style()
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        plot_paired_dumbbell(
            ax,
            paired_table,
            "value",
            case_label="Case",
            xlabel="Synthetic measurement",
            title="Synthetic paired donors",
            comparison=comparison,
        )
        return save_figure(fig, output, "paired", formats=("png", "pdf", "svg"))

    render_figure("paired", paired, figures=figures, warnings=warnings)
    rank = np.arange(1, 101)
    metric = np.linspace(3, -3, len(rank))
    hit = np.isin(rank, [2, 4, 7, 10, 14, 18, 22, 28, 36, 50, 67, 85])
    weights = np.abs(metric) * hit
    increments = weights / weights.sum() - (~hit) / (~hit).sum()
    walk = pd.DataFrame(
        {
            "source": "Synthetic pathway",
            "rank": rank,
            "metric": metric,
            "hit": hit.astype(int),
            "running_es": np.cumsum(increments),
        }
    )
    if not np.isclose(walk.running_es.iloc[-1], 0):
        raise RuntimeError("Synthetic running score must return to zero")
    walk.to_csv(output / "running_score_source.csv", index=False)

    def running_score() -> list[Path]:
        fig = gsea_running_es(
            walk,
            "Synthetic pathway",
            case_label="Case",
            control_label="Control",
            title="Synthetic enrichment walk",
        )
        return save_figure(fig, output, "running_score", formats=("png", "pdf", "svg"))

    render_figure("running_score", running_score, figures=figures, warnings=warnings)
    volcano_rng = np.random.default_rng(31)
    effects = volcano_rng.uniform(-2.8, 2.8, 120)
    volcano_table = pd.DataFrame(
        {
            "gene": [f"Gene{i:03d}" for i in range(120)],
            "logFC": effects,
            "FDR": np.power(10.0, -volcano_rng.uniform(0.1, 5.0, 120)),
            "likely_ambient": np.arange(120) % 17 == 0,
        }
    )
    volcano_table.to_csv(output / "volcano_source.csv", index=False)

    def volcano_panel() -> list[Path]:
        fig = volcano(
            volcano_table,
            fc_cut=1,
            fdr_cut=0.05,
            case_color=LE_RED,
            control_color=NORMAL_BLUE,
            x_label="Log2 fold change (Case / Control)",
            top_n_labels=0,
        )
        fig.axes[0].set_title("Synthetic differential expression", fontweight="bold")
        return save_figure(fig, output, "volcano", formats=("png", "pdf", "svg"))

    render_figure("volcano", volcano_panel, figures=figures, warnings=warnings)
    qc_rng = np.random.default_rng(43)
    qc_obs = pd.DataFrame(
        {
            "sample": np.repeat(["Control1", "Case1", "Control2", "Case2"], 100),
            "donor": np.repeat(["D1", "D1", "D2", "D2"], 100),
            "condition": np.repeat(["Control", "Case", "Control", "Case"], 100),
            "total_counts": qc_rng.lognormal(8, 0.6, 400),
            "n_genes_by_counts": qc_rng.integers(300, 2400, 400),
            "pct_counts_mito": qc_rng.uniform(1, 20, 400),
        },
        index=[f"cell_{i}" for i in range(400)],
    )
    decisions = pd.DataFrame({"keep": qc_obs.pct_counts_mito < 16}, index=qc_obs.index)
    decisions["high_mito"] = ~decisions["keep"]
    qc_frame = assemble_qc_frame(
        obs=qc_obs,
        cell_decisions=decisions,
        sample_key="sample",
        donor_key="donor",
        condition_key="condition",
    )
    qc_frame.to_csv(output / "qc_summary_source.csv", index_label="cell")

    def qc_summary() -> list[Path]:
        return write_qc_overview_figure(
            qc_frame,
            output,
            case_label="Case",
            stem="qc_summary",
            formats=("png", "pdf", "svg"),
            title="Synthetic QC overview",
        )

    render_figure("qc_summary", qc_summary, figures=figures, warnings=warnings)
    for name, build in STAGE_FIGURES.items():
        render_figure(
            name,
            lambda name=name, build=build: save_figure(build(), output, name, formats=("png",)),
            figures=figures,
            warnings=warnings,
        )
    expected = {
        output / f"{name}.{suffix}"
        for name in ("enrichment", "embedding", "paired", "running_score", "volcano", "qc_summary")
        for suffix in ("png", "pdf", "svg")
    } | {output / f"{name}.png" for name in STAGE_FIGURES}
    if warnings or set(figures) != expected:
        raise RuntimeError(f"Incomplete gallery output: {warnings}; written={figures}")
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/gallery/assets"))
    args = parser.parse_args()
    for path in build_gallery(args.output):
        print(path)


if __name__ == "__main__":
    main()
