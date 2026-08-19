"""Tests for GseaVizMethod: figures produced, skip-not-crash, running-ES optional."""

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.enrichment.viz.viz_methods import GseaVizMethod
from cellquorum.methods.base import MethodSkip


def _context(tmp_path):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    adata = ad.AnnData(np.ones((3, 2)))
    return PipelineContext(config=None, paths=paths, manifest=None, adata=adata)


def _write_gsea_csv(paths, collection):
    df = pd.DataFrame(
        {
            "source": [f"PATHWAY_{i}" for i in range(6)],
            "score": [2.1, -1.5, 0.8, -3.0, 1.2, -0.4],
            "pvalue": [1e-4, 0.2, 0.03, 1e-5, 0.01, 0.5],
            "padj": [1e-3, 0.3, 0.05, 1e-4, 0.02, 0.6],
            "significant": [True, False, False, True, True, False],
            "collection": collection,
        }
    )
    df.to_csv(paths.results / f"enrichment_gsea_{collection}.csv", index=False)


def test_gsea_viz_produces_figures(tmp_path):
    ctx = _context(tmp_path)
    _write_gsea_csv(ctx.paths, "hallmark")
    result = GseaVizMethod()._run(ctx.adata, {"figure_formats": ["png"]}, ctx)
    figs = list((ctx.paths.figures / "enrichment").glob("*.png"))
    names = {p.name for p in figs}
    assert "gsea_diverging_hallmark.png" in names
    assert "gsea_activity_hallmark.png" in names
    assert result.metrics["n_figures"] >= 2
    assert all(a.kind == "figure" for a in result.artifacts)


def test_gsea_viz_running_es_when_csv_present(tmp_path):
    ctx = _context(tmp_path)
    _write_gsea_csv(ctx.paths, "hallmark")
    n = 40
    running = pd.DataFrame(
        {
            "source": ["PATHWAY_0"] * n,
            "rank": np.arange(1, n + 1),
            "running_es": np.concatenate([np.linspace(0, 0.5, 20), np.linspace(0.5, 0, 20)]),
            "hit": ([1, 0, 0, 0] * 10),
            "metric": np.linspace(3, -3, n),
        }
    )
    running.to_csv(ctx.paths.results / "enrichment_gsea_runningES_hallmark.csv", index=False)
    GseaVizMethod()._run(ctx.adata, {"figure_formats": ["png"]}, ctx)
    figs = {p.name for p in (ctx.paths.figures / "enrichment").glob("*.png")}
    assert any(n.startswith("gsea_runningES_hallmark_") for n in figs)


def test_gsea_viz_skips_when_no_csv(tmp_path):
    ctx = _context(tmp_path)  # no gsea csv written
    out = GseaVizMethod()._run(ctx.adata, {}, ctx)
    assert isinstance(out, MethodSkip)


def test_gsea_viz_malformed_csv_skips_instead_of_crash(tmp_path):
    ctx = _context(tmp_path)
    # Write a 0-byte CSV that will trigger EmptyDataError on read.
    (ctx.paths.results / "enrichment_gsea_hallmark.csv").write_text("")
    out = GseaVizMethod()._run(ctx.adata, {}, ctx)
    assert isinstance(out, MethodSkip)
    assert any("failed to read" in str(w) for w in out.details.get("warnings", []))
