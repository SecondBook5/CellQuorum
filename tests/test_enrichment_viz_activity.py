"""Tests for ActivityVizMethod."""

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.enrichment.viz.viz_methods import ActivityVizMethod
from cellquorum.methods.base import MethodSkip


def _context(tmp_path):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    return PipelineContext(
        config=None, paths=paths, manifest=None, adata=ad.AnnData(np.ones((2, 2)))
    )


def _write_activity(paths, resource):
    rows = []
    for ct in ["Tcell", "Bcell", "Myeloid"]:
        for i in range(5):
            rows.append(
                {"cell_type": ct, "source": f"TF_{i}", "mean_score": np.sin(i) + len(ct) * 0.1}
            )
    pd.DataFrame(rows).to_csv(paths.results / f"enrichment_activity_{resource}.csv", index=False)


def test_activity_viz_produces_dotplot(tmp_path):
    ctx = _context(tmp_path)
    _write_activity(ctx.paths, "collectri")
    result = ActivityVizMethod()._run(ctx.adata, {"figure_formats": ["png"]}, ctx)
    names = {p.name for p in (ctx.paths.figures / "enrichment").glob("*.png")}
    assert "activity_dotplot_collectri.png" in names
    assert result.metrics["n_figures"] >= 1


def test_activity_viz_resource_filter(tmp_path):
    ctx = _context(tmp_path)
    _write_activity(ctx.paths, "collectri")
    _write_activity(ctx.paths, "progeny")
    ActivityVizMethod()._run(ctx.adata, {"figure_formats": ["png"], "resources": ["progeny"]}, ctx)
    names = {p.name for p in (ctx.paths.figures / "enrichment").glob("*.png")}
    assert "activity_dotplot_progeny.png" in names
    assert "activity_dotplot_collectri.png" not in names


def test_activity_viz_skips_when_no_csv(tmp_path):
    ctx = _context(tmp_path)
    assert isinstance(ActivityVizMethod()._run(ctx.adata, {}, ctx), MethodSkip)


def test_activity_viz_malformed_csv_skips_instead_of_crash(tmp_path):
    ctx = _context(tmp_path)
    # Write a 0-byte CSV that will trigger EmptyDataError on read.
    (ctx.paths.results / "enrichment_activity_collectri.csv").write_text("")
    out = ActivityVizMethod()._run(ctx.adata, {}, ctx)
    assert isinstance(out, MethodSkip)
    assert any("failed to read" in str(w) for w in out.details.get("warnings", []))
