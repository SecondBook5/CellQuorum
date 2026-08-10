# tests/test_de_viz_stage.py
import matplotlib

matplotlib.use("Agg")
import types

import anndata as ad
import numpy as np

import cellquorum.de_viz  # noqa: F401  (registers the method)
from cellquorum.de_viz.volcano_viz import VolcanoVizMethod
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.registry import METHOD_REGISTRY


def _context(tmp_path, *, case="LE", control="Normal"):
    figures = tmp_path / "figures"
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    design = types.SimpleNamespace(
        case=case,
        control=control,
        condition_col="condition",
        donor_col="patient_id",
        paired=False,
    )
    config = types.SimpleNamespace(design=design, de_viz=None)
    paths = types.SimpleNamespace(results=str(results), figures=str(figures))
    return types.SimpleNamespace(config=config, paths=paths), results, figures


def _adata():
    return ad.AnnData(np.zeros((4, 3), dtype="float32"))


def test_registered():
    assert METHOD_REGISTRY.has("de_viz", "volcano_viz")


def test_skips_when_no_csv(tmp_path):
    ctx, _results, _figures = _context(tmp_path)
    out = VolcanoVizMethod()._run(_adata(), {"case": "LE", "control": "Normal"}, ctx)
    assert isinstance(out, MethodSkip)


def test_renders_volcano(tmp_path):
    ctx, results, figures = _context(tmp_path)
    (results / "de_pseudobulk_edger.csv").write_text(
        "gene,logFC,logCPM,F,PValue,FDR\n"
        "A,2.0,5,3,1e-7,1e-6\nB,-2.5,5,3,1e-5,1e-4\nC,0.1,5,3,0.4,0.5\n"
    )
    out = VolcanoVizMethod()._run(_adata(), {"case": "LE", "control": "Normal"}, ctx)
    assert not isinstance(out, MethodSkip)
    pngs = list((figures / "differential_expression").glob("volcano*.png"))
    pdfs = list((figures / "differential_expression").glob("volcano*.pdf"))
    assert pngs and pdfs
