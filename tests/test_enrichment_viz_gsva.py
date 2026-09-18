"""Tests for GsvaVizMethod: heatmap (with/without strip) + contrast bar."""

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.comparative.enrichment.viz.viz_methods import GsvaVizMethod


def _context(tmp_path):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    return PipelineContext(
        config=None, paths=paths, manifest=None, adata=ad.AnnData(np.ones((2, 2)))
    )


def _write_gsva(paths, collection, n_paths=6):
    """Matches the REAL upstream writer's orientation (gsva_method.py's `es`):

    sample rows, pathway columns -- "per-sample scores" is sample-indexed by
    design. A test fixture built the other way around would hide any bug in
    code that assumes this orientation, which is exactly what happened here.
    """
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(
        rng.normal(size=(4, n_paths)),
        index=[f"sampleA_{j}" for j in range(2)] + [f"sampleB_{j}" for j in range(2)],
        columns=[f"PATH_{i}" for i in range(n_paths)],
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


def test_gsva_viz_condition_strip_actually_matches_real_sample_names(tmp_path, monkeypatch):
    """sample_condition keys are real sample names -- if the matrix handed to

    annotated_clustermap still has samples as columns (not transposed to
    pathway rows / sample columns), the strip lookup silently degrades to
    "partial sample_condition; strip skipped" instead of ever applying,
    because it's matching sample names against pathway names.
    """
    from cellquorum.stages.comparative.enrichment.viz import viz_methods

    ctx = _context(tmp_path)
    _write_gsva(ctx.paths, "hallmark")
    mapping = {"sampleA_0": "case", "sampleA_1": "case", "sampleB_0": "ctrl", "sampleB_1": "ctrl"}
    captured = {}
    real_clustermap = viz_methods.plots.annotated_clustermap

    def spy(*args, **kwargs):
        captured["col_colors"] = kwargs.get("col_colors")
        captured["matrix"] = args[0] if args else kwargs.get("matrix")
        return real_clustermap(*args, **kwargs)

    monkeypatch.setattr(viz_methods.plots, "annotated_clustermap", spy)
    result = GsvaVizMethod()._run(
        ctx.adata, {"figure_formats": ["png"], "sample_condition": mapping}, ctx
    )
    assert captured["col_colors"] is not None
    assert not any("strip skipped" in str(w) for w in result.warnings)


def test_gsva_viz_heatmap_caps_pathway_count_not_sample_count(tmp_path, monkeypatch):
    """The real upstream CSV is sample-rows x pathway-columns (few samples,

    many pathways) -- annotated_clustermap's top_n caps ROWS, so the matrix
    handed to it must have pathways as rows, or a large collection (hallmark
    alone has 50) never gets capped at all.
    """
    from cellquorum.stages.comparative.enrichment.viz import viz_methods

    ctx = _context(tmp_path)
    _write_gsva(ctx.paths, "hallmark", n_paths=50)
    captured = {}
    real_clustermap = viz_methods.plots.annotated_clustermap

    def spy(*args, **kwargs):
        matrix = args[0] if args else kwargs.get("matrix")
        captured["n_rows"] = matrix.shape[0]
        return real_clustermap(*args, **kwargs)

    monkeypatch.setattr(viz_methods.plots, "annotated_clustermap", spy)
    GsvaVizMethod()._run(ctx.adata, {"figure_formats": ["png"]}, ctx)
    # 50 pathways, 4 samples -- if rows were still samples this would be 4.
    assert captured["n_rows"] == 50


def test_gsva_viz_skips_when_no_csv(tmp_path):
    ctx = _context(tmp_path)
    assert isinstance(GsvaVizMethod()._run(ctx.adata, {}, ctx), MethodSkip)
