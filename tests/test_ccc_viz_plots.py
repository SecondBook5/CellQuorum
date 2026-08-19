import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx  # noqa: F401
import pandas as pd
from matplotlib.figure import Figure


def test_save_figure_writes_formats(tmp_path):
    from cellquorum.cell_cell_communication.viz._io import figure_artifacts, save_figure

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, tmp_path / "sub", "demo", formats=("pdf", "png"), dpi=100)
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert all(p.exists() for p in paths)
    arts = figure_artifacts(paths, name="ccc_figure", description="demo")
    assert all(a.kind == "figure" for a in arts)
    assert len(arts) == 2


def _lr(n_pairs=3):
    rows = []
    cts = ["A", "B", "C"]
    for i in range(n_pairs):
        rows.append(
            {
                "source": cts[i % 3],
                "target": cts[(i + 1) % 3],
                "ligand": f"L{i}",
                "receptor": f"R{i}",
                "weight": 0.1 * (i + 1),
                "sample": f"s{i%2}",
            }
        )
    return pd.DataFrame(rows)


def test_celltype_palette_deterministic_and_entity_keyed():
    from cellquorum.cell_cell_communication.viz._plots import celltype_palette

    p1 = celltype_palette(["B", "A", "C"])
    p2 = celltype_palette(["C", "A", "B"])  # different order
    assert p1 == p2  # color follows entity, not rank
    assert set(p1) == {"A", "B", "C"}


def test_celltype_palette_overflow_to_other():
    from cellquorum.cell_cell_communication.viz._plots import (
        _CELLTYPE_HEXES,
        _OTHER_GRAY,
        celltype_palette,
    )

    many = [f"ct{i}" for i in range(len(_CELLTYPE_HEXES) + 3)]
    pal = celltype_palette(many)
    assert list(pal.values()).count(_OTHER_GRAY) == 3


def test_interaction_dotplot_returns_figure():
    from cellquorum.cell_cell_communication.viz._plots import interaction_dotplot

    assert isinstance(interaction_dotplot(_lr(4), top_k=2), Figure)


def test_interaction_dotplot_empty_guarded():
    from cellquorum.cell_cell_communication.viz._plots import interaction_dotplot

    empty = pd.DataFrame(columns=["source", "target", "ligand", "receptor", "weight", "sample"])
    assert isinstance(interaction_dotplot(empty), Figure)


def test_cci_heatmap_sequential_and_diverging():
    from cellquorum.cell_cell_communication.viz._plots import cci_heatmap

    assert isinstance(cci_heatmap(_lr(3)), Figure)
    diff = _lr(3).assign(weight=[-0.2, 0.1, 0.3])
    assert isinstance(cci_heatmap(diff, diverging=True), Figure)


def test_chord_and_sankey_return_figure():
    from cellquorum.cell_cell_communication.viz._plots import (
        celltype_palette,
        chord_diagram,
        sankey_flow,
    )

    df = _lr(4)
    pal = celltype_palette(list(set(df["source"]) | set(df["target"])))
    assert isinstance(chord_diagram(df, palette=pal, top_k=3), Figure)
    assert isinstance(sankey_flow(df, palette=pal, top_k=3), Figure)


def test_chord_sankey_fallback_when_dep_missing(monkeypatch):
    import builtins

    from cellquorum.cell_cell_communication.viz import _plots

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("pycirclize") or name == "plotly" or name.startswith("plotly."):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = _lr(4)
    pal = _plots.celltype_palette(list(set(df["source"]) | set(df["target"])))
    assert isinstance(_plots.chord_diagram(df, palette=pal), Figure)  # matplotlib fallback
    assert isinstance(_plots.sankey_flow(df, palette=pal), Figure)  # matplotlib fallback


def test_curvature_network_returns_figure():
    from cellquorum.cell_cell_communication.viz._plots import curvature_network

    edges = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "ricci_curvature": [-0.3, 0.2],
            "weight": [1.0, 2.0],
        }
    )
    nodes = pd.DataFrame({"node": ["A", "B", "C"], "ricci_curvature": [-0.3, -0.05, 0.2]})
    assert isinstance(curvature_network(edges, nodes), Figure)


def test_curvature_network_empty_guarded():
    from cellquorum.cell_cell_communication.viz._plots import curvature_network

    e = pd.DataFrame(columns=["source", "target", "ricci_curvature", "weight"])
    n = pd.DataFrame(columns=["node", "ricci_curvature"])
    assert isinstance(curvature_network(e, n), Figure)


def test_curvature_network_without_weight_col():
    from cellquorum.cell_cell_communication.viz._plots import curvature_network

    edges = pd.DataFrame({"source": ["A"], "target": ["B"], "delta_curvature": [-0.1]})
    fig = curvature_network(
        edges, None, curvature_col="delta_curvature", node_curv_col="delta_curvature"
    )
    assert fig is not None


def test_topology_facets_returns_figure():
    from cellquorum.cell_cell_communication.viz._plots import topology_facets

    topo = pd.DataFrame(
        {
            "node": ["A", "B", "C"],
            "Listener": [1.0, 2.0, -1.0],
            "Influencer": [0.1, 0.2, 0.3],
            "Mediator": [0.0, 1.0, 2.0],
            "Pagerank": [0.2, 0.3, 0.5],
        }
    )
    assert isinstance(topology_facets(topo, top_k=2), Figure)


def test_interaction_dotplot_all_nan_weight():
    from cellquorum.cell_cell_communication.viz._plots import interaction_dotplot

    df = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "weight": [float("nan"), float("nan")],
            "sample": ["s1", "s2"],
        }
    )
    # Should return a Figure without raising
    fig = interaction_dotplot(df, top_k=5)
    assert isinstance(fig, Figure)
