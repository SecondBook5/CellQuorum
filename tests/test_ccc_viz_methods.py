from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd


def _ctx(tmp_path, uns=None):
    adata = ad.AnnData(np.ones((3, 2)))
    if uns:
        adata.uns.update(uns)
    paths = SimpleNamespace(results=str(tmp_path / "results"), figures=str(tmp_path / "figures"))
    (tmp_path / "results").mkdir(exist_ok=True)
    return adata, SimpleNamespace(paths=paths, config=SimpleNamespace())


def _write_canonical(results_dir, name="mnn_canonical_lr.csv"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "weight": [0.5, 0.7],
            "sample": ["s1", "s2"],
        }
    ).to_csv(Path(results_dir) / name, index=False)


def test_dotplot_method_skips_without_input(tmp_path):
    from cellquorum.ccc_viz.dotplot_viz import DotplotVizMethod
    from cellquorum.methods.base import MethodSkip

    adata, ctx = _ctx(tmp_path)
    out = DotplotVizMethod().run(adata, {}, ctx)
    assert isinstance(out, MethodSkip)


def test_dotplot_method_renders(tmp_path):
    from cellquorum.ccc_viz.dotplot_viz import DotplotVizMethod
    from cellquorum.core.stage import StageResult

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results)
    out = DotplotVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_chord_method_renders(tmp_path):
    from cellquorum.ccc_viz.chord_viz import ChordVizMethod
    from cellquorum.core.stage import StageResult

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results)
    out = ChordVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_dotplot_contract_empty_and_python(tmp_path):
    from cellquorum.ccc_viz.dotplot_viz import DotplotVizMethod

    m = DotplotVizMethod()
    assert m.backend == "python"
    dc = m.input_contract({})
    assert not getattr(dc, "required_obs", []) and not getattr(dc, "required_layers", [])
