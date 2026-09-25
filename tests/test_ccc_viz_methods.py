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
    from cellquorum.methods.base import MethodSkip
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import DotplotVizMethod

    adata, ctx = _ctx(tmp_path)
    out = DotplotVizMethod().run(adata, {}, ctx)
    assert isinstance(out, MethodSkip)


def test_dotplot_method_renders(tmp_path):
    from cellquorum.core.stage import StageResult
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import DotplotVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results)
    out = DotplotVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_chord_method_renders(tmp_path):
    from cellquorum.core.stage import StageResult
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import ChordVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results)
    out = ChordVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def _write_mnn_prioritization(results_dir):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "group": ["Lymphedema", "Normal"],
            "sender": ["0", "3"],
            "receiver": ["3", "0"],
            "ligand": ["TGFB1", "POSTN"],
            "receptor": ["TGFBR1", "ITGAV"],
            "prioritization_score": [0.95, 0.80],
        }
    ).to_csv(Path(results_dir) / "mnn_prioritization.csv", index=False)


def test_multinichenet_viz_method_skips_without_input(tmp_path):
    from cellquorum.methods.base import MethodSkip
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import MultiNicheNetVizMethod

    adata, ctx = _ctx(tmp_path)
    out = MultiNicheNetVizMethod().run(adata, {}, ctx)
    assert isinstance(out, MethodSkip)


def test_multinichenet_viz_method_renders_one_focused_circos_per_key_cluster(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import MultiNicheNetVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_mnn_prioritization(ctx.paths.results)
    out = MultiNicheNetVizMethod().run(
        adata, {"figure_formats": ["png"], "mnn_key_clusters": ["0"]}, ctx
    )
    names = {a.path.stem for a in out.artifacts}
    # "0" appears as sender in the Normal row and as receiver in neither of
    # the two fixture rows for Lymphedema -- one focused circos per group
    # that actually has an edge touching "0".
    assert any("multinichenet_circos_0" in n for n in names)


def test_multinichenet_viz_method_defaults_key_clusters_to_everything_present(tmp_path):
    # No mnn_key_clusters given -> render for every distinct cluster seen as
    # sender or receiver, not silently limited to whatever the engine
    # guesses is "key" -- config opts in to narrowing, never the other way.
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import MultiNicheNetVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_mnn_prioritization(ctx.paths.results)
    out = MultiNicheNetVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    names = {a.path.stem for a in out.artifacts}
    assert any("multinichenet_circos_0" in n for n in names)
    assert any("multinichenet_circos_3" in n for n in names)


def test_multinichenet_viz_method_renders_one_lollipop_per_group(tmp_path):
    # One lollipop per condition (matching the reference this was built to
    # reproduce), not both groups crammed into one -- a combined figure
    # needs one legend entry per sender->receiver route seen in EITHER
    # group, which becomes illegible past a handful of routes.
    from cellquorum.core.stage import StageResult
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import MultiNicheNetVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_mnn_prioritization(ctx.paths.results)
    out = MultiNicheNetVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    names = {a.path.stem for a in out.artifacts}
    assert "multinichenet_top_lr_Lymphedema" in names
    assert "multinichenet_top_lr_Normal" in names


def _write_nichenet(results_dir, *, with_weights=True, with_expr=True, with_receptor_weights=True):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"test_ligand": ["TGFB2", "IL6"], "aupr_corrected": [0.021, 0.041]}).to_csv(
        Path(results_dir) / "nichenet_activities.csv", index=False
    )
    if with_weights:
        pd.DataFrame({"ligand": ["TGFB2"], "target": ["COL1A1"], "weight": [0.24]}).to_csv(
            Path(results_dir) / "nichenet_ligand_target_weights.csv", index=False
        )
    if with_expr:
        pd.DataFrame(
            {
                "sender": ["1", "3"],
                "ligand": ["TGFB2", "TGFB2"],
                "fraction_expressing": [0.17, 0.40],
                "mean_expression": [0.32, 0.69],
            }
        ).to_csv(Path(results_dir) / "nichenet_sender_expression.csv", index=False)
    if with_receptor_weights:
        pd.DataFrame({"ligand": ["TGFB2"], "receptor": ["TGFBR1"], "weight": [0.9]}).to_csv(
            Path(results_dir) / "nichenet_ligand_receptor_weights.csv", index=False
        )


def test_nichenet_viz_method_skips_without_input(tmp_path):
    from cellquorum.methods.base import MethodSkip
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import NicheNetVizMethod

    adata, ctx = _ctx(tmp_path)
    out = NicheNetVizMethod().run(adata, {}, ctx)
    assert isinstance(out, MethodSkip)


def test_nichenet_viz_method_renders_all_four_figures(tmp_path):
    from cellquorum.core.stage import StageResult
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import NicheNetVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_nichenet(ctx.paths.results)
    out = NicheNetVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert out.metrics["n_figures"] == 4


def test_nichenet_viz_method_renders_only_whats_present(tmp_path):
    from cellquorum.core.stage import StageResult
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import NicheNetVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_nichenet(
        ctx.paths.results, with_weights=False, with_expr=False, with_receptor_weights=False
    )
    out = NicheNetVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert out.metrics["n_figures"] == 1


def test_nichenet_viz_method_relabels_sender_clusters(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import NicheNetVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_nichenet(ctx.paths.results)
    out = NicheNetVizMethod().run(
        adata,
        {"figure_formats": ["png"], "cluster_labels": {"1": "Perivascular", "3": "Reticular"}},
        ctx,
    )
    assert out.metrics["n_figures"] == 4


def test_dotplot_contract_empty_and_python(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import DotplotVizMethod

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
    from cellquorum.core.stage import StageResult
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import SankeyVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results)
    out = SankeyVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_network_method_skips_without_curvature(tmp_path):
    from cellquorum.methods.base import MethodSkip
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import NetworkVizMethod

    adata, ctx = _ctx(tmp_path)
    assert isinstance(NetworkVizMethod().run(adata, {}, ctx), MethodSkip)


def test_network_method_renders(tmp_path):
    from cellquorum.core.stage import StageResult
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import NetworkVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_curvature(ctx.paths.results)
    out = NetworkVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_summary_method_renders_with_topology(tmp_path):
    from cellquorum.core.stage import StageResult
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import SummaryVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results)
    _write_topology(ctx.paths.results)
    out = SummaryVizMethod().run(adata, {"figure_formats": ["png"]}, ctx)
    assert isinstance(out, StageResult)
    assert any(a.kind == "figure" for a in out.artifacts)


def test_summary_method_skips_when_nothing(tmp_path):
    from cellquorum.methods.base import MethodSkip
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import SummaryVizMethod

    adata, ctx = _ctx(tmp_path)
    assert isinstance(SummaryVizMethod().run(adata, {}, ctx), MethodSkip)


def test_summary_method_honors_sources_filter(tmp_path):
    from cellquorum.methods.base import MethodSkip
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import SummaryVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_canonical(ctx.paths.results, name="mnn_canonical_lr.csv")
    # Filter excludes the only present source; no topology → MethodSkip
    out = SummaryVizMethod().run(
        adata, {"figure_formats": ["png"], "sources": ["nonexistent"]}, ctx
    )
    assert isinstance(out, MethodSkip)


def test_network_method_honors_levels_filter(tmp_path):
    from cellquorum.methods.base import MethodSkip
    from cellquorum.stages.cell_cell_communication.viz._viz_methods import NetworkVizMethod

    adata, ctx = _ctx(tmp_path)
    _write_curvature(ctx.paths.results)  # writes only cci curvature
    # Filter requests only gci, but only cci exists → MethodSkip
    out = NetworkVizMethod().run(adata, {"figure_formats": ["png"], "levels": ["gci"]}, ctx)
    assert isinstance(out, MethodSkip)
