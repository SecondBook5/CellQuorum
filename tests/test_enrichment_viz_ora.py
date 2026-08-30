"""Tests for OraVizMethod."""

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.comparative.enrichment.viz.viz_methods import OraVizMethod
from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.methods.base import MethodSkip


def _context(tmp_path):
    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    return PipelineContext(
        config=None, paths=paths, manifest=None, adata=ad.AnnData(np.ones((2, 2)))
    )


def _write_ora_csv(paths, collection):
    pd.DataFrame(
        {
            "source": [f"SET_{i}" for i in range(6)],
            "direction": ["up", "up", "up", "down", "down", "down"],
            "score": [1.2, 0.8, 0.3, 1.5, 0.6, 0.1],
            "pvalue": [1e-4, 0.01, 0.2, 1e-5, 0.02, 0.4],
            "padj": [1e-3, 0.03, 0.3, 1e-4, 0.05, 0.5],
            "significant": [True, True, False, True, False, False],
            "collection": collection,
            "count": [12, 6, 2, 15, 5, 1],
            "gene_ratio": [0.24, 0.12, 0.04, 0.3, 0.1, 0.02],
        }
    ).to_csv(paths.results / f"enrichment_ora_{collection}.csv", index=False)


def test_ora_viz_produces_barplot_and_dotplot(tmp_path):
    ctx = _context(tmp_path)
    _write_ora_csv(ctx.paths, "reactome")
    result = OraVizMethod()._run(ctx.adata, {"figure_formats": ["png"]}, ctx)
    names = {p.name for p in (ctx.paths.figures / "enrichment").glob("*.png")}
    assert "ora_barplot_reactome.png" in names
    assert "ora_dotplot_reactome.png" in names
    assert result.metrics["n_figures"] >= 2


def test_ora_viz_skips_when_no_csv(tmp_path):
    ctx = _context(tmp_path)
    assert isinstance(OraVizMethod()._run(ctx.adata, {}, ctx), MethodSkip)


def test_ora_viz_malformed_csv_skips_instead_of_crash(tmp_path):
    ctx = _context(tmp_path)
    # Write a 0-byte CSV that will trigger EmptyDataError on read.
    (ctx.paths.results / "enrichment_ora_reactome.csv").write_text("")
    out = OraVizMethod()._run(ctx.adata, {}, ctx)
    assert isinstance(out, MethodSkip)
    assert any("failed to read" in str(w) for w in out.details.get("warnings", []))
