"""End-to-end dispatch tests for EnrichmentVizStage."""

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.comparative.enrichment.viz.stage import EnrichmentVizStage
from cellquorum.core.context import PipelineContext, PipelinePaths


def _ctx(tmp_path, config=None):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    return PipelineContext(
        config=config, paths=paths, manifest=None, adata=ad.AnnData(np.ones((3, 2)))
    )


def _write_all_csvs(paths):
    pd.DataFrame(
        {
            "source": ["A", "B", "C", "D"],
            "score": [2.0, -1.0, 1.0, -2.0],
            "pvalue": [0.01, 0.2, 0.03, 1e-4],
            "padj": [0.03, 0.3, 0.05, 1e-3],
            "significant": [True, False, False, True],
            "collection": "hallmark",
        }
    ).to_csv(paths.results / "enrichment_gsea_hallmark.csv", index=False)
    pd.DataFrame(
        {
            "source": ["A", "B", "C", "D"],
            "direction": ["up", "up", "down", "down"],
            "score": [1.0, 0.5, 1.2, 0.3],
            "pvalue": [0.01, 0.2, 1e-4, 0.4],
            "padj": [0.03, 0.3, 1e-3, 0.5],
            "significant": [True, False, True, False],
            "collection": "hallmark",
            "count": [10, 4, 12, 2],
            "gene_ratio": [0.2, 0.08, 0.24, 0.04],
        }
    ).to_csv(paths.results / "enrichment_ora_hallmark.csv", index=False)
    rng = np.random.default_rng(0)
    pd.DataFrame(
        rng.normal(size=(4, 4)), index=["A", "B", "C", "D"], columns=[f"s{j}" for j in range(4)]
    ).to_csv(paths.results / "enrichment_gsva_scores_hallmark.csv")
    pd.DataFrame(
        {
            "source": ["A", "B", "C", "D"],
            "case_mean": [0.5, -0.3, 0.1, 0.8],
            "control_mean": [0.0, 0.2, 0.1, -0.2],
            "statistic": [2.1, -1.0, 0.1, 3.0],
            "pvalue": [0.01, 0.3, 0.9, 1e-4],
            "padj": [0.03, 0.4, 0.9, 1e-3],
            "significant": [True, False, False, True],
            "collection": "hallmark",
        }
    ).to_csv(paths.results / "enrichment_gsva_contrast_hallmark.csv", index=False)
    rows = [
        {"cell_type": ct, "source": f"TF_{i}", "mean_score": float(i) - len(ct) * 0.1}
        for ct in ["T", "B"]
        for i in range(4)
    ]
    pd.DataFrame(rows).to_csv(paths.results / "enrichment_activity_collectri.csv", index=False)


def test_stage_runs_four_methods_and_produces_figures(tmp_path):
    ctx = _ctx(tmp_path)
    _write_all_csvs(ctx.paths)
    result = EnrichmentVizStage().run(ctx)
    assert result.metrics["n_methods"] == 4
    assert len(result.metrics["per_method"]) == 4
    figs = list((ctx.paths.figures / "enrichment").glob("*"))
    assert len(figs) > 0


def test_stage_always_on_skips_cleanly_when_no_inputs(tmp_path):
    ctx = _ctx(tmp_path)  # no CSVs written
    result = EnrichmentVizStage().run(ctx)
    # Every method records a skip; stage completes without crashing.
    assert result.metrics["n_methods"] == 4
    assert all(m.get("skipped") for m in result.metrics["per_method"])


def test_methods_registered():
    import cellquorum.stages.comparative.enrichment.viz  # noqa: F401  (triggers registration)
    from cellquorum.methods.registry import METHOD_REGISTRY

    for name in ["gsea_viz", "ora_viz", "gsva_viz", "activity_viz"]:
        assert METHOD_REGISTRY.has("enrichment_viz", name)
