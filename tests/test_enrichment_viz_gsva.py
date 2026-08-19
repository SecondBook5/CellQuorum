"""Tests for GsvaVizMethod: heatmap (with/without strip) + contrast bar."""

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.enrichment.viz.gsva_viz import GsvaVizMethod
from cellquorum.methods.base import MethodSkip


def _context(tmp_path):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    return PipelineContext(
        config=None, paths=paths, manifest=None, adata=ad.AnnData(np.ones((2, 2)))
    )


def _write_gsva(paths, collection):
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(
        rng.normal(size=(6, 4)),
        index=[f"PATH_{i}" for i in range(6)],
        columns=[f"sampleA_{j}" for j in range(2)] + [f"sampleB_{j}" for j in range(2)],
    )
    scores.to_csv(paths.results / f"enrichment_gsva_scores_{collection}.csv")
    pd.DataFrame(
        {
            "source": [f"PATH_{i}" for i in range(6)],
            "case_mean": [0.5, -0.3, 0.1, 0.8, -0.6, 0.0],
            "control_mean": [0.0, 0.2, 0.1, -0.2, 0.1, 0.0],
            "statistic": [2.1, -1.0, 0.1, 3.0, -2.0, 0.0],
            "pvalue": [0.01, 0.3, 0.9, 1e-4, 0.02, 1.0],
            "padj": [0.03, 0.4, 0.9, 1e-3, 0.05, 1.0],
            "significant": [True, False, False, True, False, False],
            "collection": collection,
        }
    ).to_csv(paths.results / f"enrichment_gsva_contrast_{collection}.csv", index=False)


def test_gsva_viz_produces_heatmap_and_contrast(tmp_path):
    ctx = _context(tmp_path)
    _write_gsva(ctx.paths, "hallmark")
    result = GsvaVizMethod()._run(ctx.adata, {"figure_formats": ["png"]}, ctx)
    names = {p.name for p in (ctx.paths.figures / "enrichment").glob("*.png")}
    assert "gsva_heatmap_hallmark.png" in names
    assert "gsva_contrast_hallmark.png" in names
    assert result.metrics["n_figures"] >= 2


def test_gsva_viz_heatmap_with_condition_strip(tmp_path):
    ctx = _context(tmp_path)
    _write_gsva(ctx.paths, "hallmark")
    mapping = {"sampleA_0": "case", "sampleA_1": "case", "sampleB_0": "ctrl", "sampleB_1": "ctrl"}
    result = GsvaVizMethod()._run(
        ctx.adata, {"figure_formats": ["png"], "sample_condition": mapping}, ctx
    )
    assert result.metrics["n_figures"] >= 2  # strip path must not crash


def test_gsva_viz_skips_when_no_csv(tmp_path):
    ctx = _context(tmp_path)
    assert isinstance(GsvaVizMethod()._run(ctx.adata, {}, ctx), MethodSkip)
