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
    from cellquorum.cell_cell_communication.viz.dotplot_viz import DotplotVizMethod
    from cellquorum.methods.base import MethodSkip

    adata, ctx = _ctx(tmp_path)
    out = DotplotVizMethod().run(adata, {}, ctx)
    assert isinstance(out, MethodSkip)


def test_dotplot_method_renders(tmp_path):
    from cellquorum.cell_cell_communication.viz.dotplot_viz import DotplotVizMethod
    from cellquorum.core.stage import StageResult

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results)
    out = DotplotVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_chord_method_renders(tmp_path):
    from cellquorum.cell_cell_communication.viz.chord_viz import ChordVizMethod
    from cellquorum.core.stage import StageResult

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results)
    out = ChordVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_dotplot_contract_empty_and_python(tmp_path):
    from cellquorum.cell_cell_communication.viz.dotplot_viz import DotplotVizMethod

    m = DotplotVizMethod()
    assert m.backend == "python"
    dc = m.input_contract({})
    assert not getattr(dc, "required_obs", []) and not getattr(dc, "required_layers", [])


def _write_curvature(results_dir):
    net = Path(results_dir) / "ccc_network"
    net.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "ricci_curvature": [-0.3, 0.2],
            "weight": [1.0, 2.0],
        }
    ).to_csv(net / "curvature_cci_edges.csv", index=False)
    pd.DataFrame({"node": ["A", "B", "C"], "ricci_curvature": [-0.3, -0.05, 0.2]}).to_csv(
        net / "curvature_cci_nodes.csv", index=False
    )


def _write_topology(results_dir):
    net = Path(results_dir) / "ccc_network"
    net.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "node": ["A", "B", "C"],
            "Listener": [1.0, 2.0, -1.0],
            "Influencer": [0.1, 0.2, 0.3],
            "Mediator": [0.0, 1.0, 2.0],
            "Pagerank": [0.2, 0.3, 0.5],
        }
    ).to_csv(net / "topology_cci.csv", index=False)


def test_sankey_method_renders(tmp_path):
    from cellquorum.cell_cell_communication.viz.sankey_viz import SankeyVizMethod
    from cellquorum.core.stage import StageResult

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results)
    out = SankeyVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_network_method_skips_without_curvature(tmp_path):
    from cellquorum.cell_cell_communication.viz.network_viz import NetworkVizMethod
    from cellquorum.methods.base import MethodSkip

    adata, ctx = _ctx(tmp_path)
    assert isinstance(NetworkVizMethod().run(adata, {}, ctx), MethodSkip)


def test_network_method_renders(tmp_path):
    from cellquorum.cell_cell_communication.viz.network_viz import NetworkVizMethod
    from cellquorum.core.stage import StageResult

    adata, ctx = _ctx(tmp_path)
    _write_curvature(ctx.paths.results)
    out = NetworkVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_summary_method_renders_with_topology(tmp_path):
    from cellquorum.cell_cell_communication.viz.summary_viz import SummaryVizMethod
    from cellquorum.core.stage import StageResult

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results)
    _write_topology(ctx.paths.results)
    out = SummaryVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_summary_method_skips_when_nothing(tmp_path):
    from cellquorum.cell_cell_communication.viz.summary_viz import SummaryVizMethod
    from cellquorum.methods.base import MethodSkip

    adata, ctx = _ctx(tmp_path)
    assert isinstance(SummaryVizMethod().run(adata, {}, ctx), MethodSkip)


def test_summary_method_honors_sources_filter(tmp_path):
    from cellquorum.cell_cell_communication.viz.summary_viz import SummaryVizMethod
    from cellquorum.methods.base import MethodSkip

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results, name="mnn_canonical_lr.csv")
    # Filter excludes the only present source; no topology → MethodSkip
    out = SummaryVizMethod().run(
        adata, {"figure_formats": ["png"], "sources": ["nonexistent"]}, ctx
    )
    assert isinstance(out, MethodSkip)


def test_network_method_honors_levels_filter(tmp_path):
    from cellquorum.cell_cell_communication.viz.network_viz import NetworkVizMethod
    from cellquorum.methods.base import MethodSkip

    adata, ctx = _ctx(tmp_path)
    _write_curvature(ctx.paths.results)  # writes only cci curvature
    # Filter requests only gci, but only cci exists → MethodSkip
    out = NetworkVizMethod().run(adata, {"figure_formats": ["png"], "levels": ["gci"]}, ctx)
    assert isinstance(out, MethodSkip)
