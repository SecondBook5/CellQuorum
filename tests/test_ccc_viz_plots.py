import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx  # noqa: F401
import pandas as pd
from matplotlib.figure import Figure


def test_save_figure_writes_formats(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import figure_artifacts, save_figure

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
    from cellquorum.stages.cell_cell_communication.viz._plots import celltype_palette

    p1 = celltype_palette(["B", "A", "C"])
    p2 = celltype_palette(["C", "A", "B"])  # different order
    assert p1 == p2  # color follows entity, not rank
    assert set(p1) == {"A", "B", "C"}


def test_celltype_palette_never_degrades_more_than_8_types_to_flat_gray():
    """The real fibroblast cohort has 9 subclusters -- one more than the old
    fixed 8-hex list, so cluster #9 rendered as flat gray in EVERY CCC figure,
    visually indistinguishable from background. celltype_palette must draw
    from the same unbounded, audited palette every other stage uses
    (figstyle.palette_colors), which never cycles or grays out."""
    from cellquorum.stages.cell_cell_communication.viz._plots import celltype_palette

    many = [f"ct{i}" for i in range(11)]
    pal = celltype_palette(many)
    assert len(set(pal.values())) == 11


def test_celltype_palette_matches_the_validated_categorical_palette():
    """celltype_palette must draw from figstyle's audited CATEGORICAL_PALETTE,
    not a second, never-audited color list -- otherwise a CCC figure can use
    a color the palette-audit tooling has never checked for colorblind
    safety."""
    from cellquorum.stages.cell_cell_communication.viz._plots import celltype_palette
    from cellquorum.visualization.figstyle import CATEGORICAL_PALETTE

    pal = celltype_palette(["A", "B", "C"])
    assert set(pal.values()) <= set(CATEGORICAL_PALETTE)


def test_interaction_dotplot_returns_figure():
    from cellquorum.stages.cell_cell_communication.viz._plots import interaction_dotplot

    assert isinstance(interaction_dotplot(_lr(4), top_k=2), Figure)


def _dense_lr_many_pairs(n_clusters: int, n_lr_pairs: int) -> pd.DataFrame:
    """Every cluster pair carries every LR pair -- the real shape of a
    9-subcluster LIANA run (72 source-target combos), which is what made
    dotplot_liana.png 12,664px wide: it curates LR pairs (rows) but never
    curates source-target combos (columns), so a handful of kept LR pairs
    can still spray across all 72 columns."""
    import numpy as np

    rng = np.random.default_rng(0)
    clusters = [str(i) for i in range(n_clusters)]
    rows = []
    for s in clusters:
        for t in clusters:
            if s == t:
                continue
            for i in range(n_lr_pairs):
                rows.append(
                    {
                        "source": s,
                        "target": t,
                        "ligand": f"L{i}",
                        "receptor": f"R{i}",
                        "weight": float(rng.uniform(0.1, 1.0)),
                        "sample": "s0",
                    }
                )
    return pd.DataFrame(rows)


def test_interaction_dotplot_caps_source_target_columns_too():
    """Real bug, confirmed live: interaction_dotplot curates LR pairs (rows,
    via top_k) but drew a column for EVERY distinct source->target combo
    that survived, uncapped. On a real 9-subcluster CCC run that meant 72
    columns even with only a handful of LR pairs kept -- a 12,664px-wide,
    practically unreadable figure. Columns must be curated the same way
    rows are.
    """
    from cellquorum.stages.cell_cell_communication.viz._plots import _interaction_dotplot_data

    lr = _dense_lr_many_pairs(9, 3)
    data = _interaction_dotplot_data(lr, top_k=3, max_pairs=20)

    assert len(data["xs"]) <= 20
    assert len(data["ys"]) <= 3


def test_interaction_dotplot_empty_guarded():
    from cellquorum.stages.cell_cell_communication.viz._plots import interaction_dotplot

    empty = pd.DataFrame(columns=["source", "target", "ligand", "receptor", "weight", "sample"])
    assert isinstance(interaction_dotplot(empty), Figure)


def test_cci_heatmap_sequential_and_diverging():
    from cellquorum.stages.cell_cell_communication.viz._plots import cci_heatmap

    assert isinstance(cci_heatmap(_lr(3)), Figure)
    diff = _lr(3).assign(weight=[-0.2, 0.1, 0.3])
    assert isinstance(cci_heatmap(diff, diverging=True), Figure)


def _fully_connected_lr(n_clusters: int) -> pd.DataFrame:
    """Every cluster signals to every other cluster -- the real shape of a
    fibroblast_subcluster CCC run (9 subclusters, ~72 source-target pairs all
    carrying some real signal), not a sparse toy graph."""
    import numpy as np

    rng = np.random.default_rng(0)
    clusters = [str(i) for i in range(n_clusters)]
    rows = []
    for s in clusters:
        for t in clusters:
            if s == t:
                continue
            rows.append({"source": s, "target": t, "weight": float(rng.uniform(0.1, 1.0))})
    return pd.DataFrame(rows)


def test_top_edges_default_does_not_drop_nodes_on_a_dense_graph():
    """Real bug, confirmed live: a 9-subcluster fibroblast CCC run's chord
    diagram silently dropped 3 of 9 clusters entirely. top_k=15 is a flat cap
    applied to (source, target) pairs AFTER aggregation; with 9 clusters and
    ~72 real pairs, only the 15 globally-strongest pairs survived, and the 3
    clusters whose own pairs never made that list vanished from the figure
    with no warning. The cap must scale with how many nodes are actually in
    the graph, not stay fixed regardless of cluster count.
    """
    from cellquorum.stages.cell_cell_communication.viz._plots import _top_edges

    lr = _fully_connected_lr(9)
    edges = _top_edges(lr, "source", "target", "weight", top_k=15)

    present_nodes = set(edges["source"]) | set(edges["target"])
    assert present_nodes == {str(i) for i in range(9)}


def test_top_edges_never_drops_a_node_even_with_a_tiny_explicit_top_k():
    """top_k trims SURPLUS edges once every node is represented; it must
    never be the reason a node silently vanishes from the figure, even when
    explicitly set below the node count."""
    from cellquorum.stages.cell_cell_communication.viz._plots import _top_edges

    lr = _fully_connected_lr(6)
    edges = _top_edges(lr, "source", "target", "weight", top_k=2)

    present_nodes = set(edges["source"]) | set(edges["target"])
    assert present_nodes == {str(i) for i in range(6)}


def test_top_edges_does_not_inflate_a_graph_that_already_fits() -> None:
    """When the input already has at most top_k pairs, nothing is added or
    dropped -- the guarantee only ever fills gaps, never pads output."""
    from cellquorum.stages.cell_cell_communication.viz._plots import _top_edges

    lr = _fully_connected_lr(3)  # 6 possible (source, target) pairs
    edges = _top_edges(lr, "source", "target", "weight", top_k=100)

    assert len(edges) == 6


def _fully_connected_lr_with_genes(n_clusters: int) -> pd.DataFrame:
    """Same shape as _fully_connected_lr, plus the ligand/receptor columns
    sankey_flow needs -- the real shape of a canonical LIANA results frame.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    clusters = [str(i) for i in range(n_clusters)]
    rows = []
    for s in clusters:
        for t in clusters:
            if s == t:
                continue
            rows.append(
                {
                    "source": s,
                    "target": t,
                    "ligand": f"L_{s}_{t}",
                    "receptor": f"R_{s}_{t}",
                    "weight": float(rng.uniform(0.1, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def test_select_sankey_flows_does_not_starve_a_cluster_of_all_flow():
    """Real bug found on a rendered figure: sankey_flow took a single global
    top-K cut (df.sort_values(weight).head(top_k)) across the ENTIRE LR
    results table before building the diagram. On a real cohort with
    thousands of source-ligand-receptor-target rows, the top-15 strongest
    individual rows were dominated by one or two source clusters, leaving
    another cluster with literally zero flow -- completely isolated in the
    rendered diagram, with no warning. chord_diagram already guards against
    this exact failure via _top_edges/_guarantee_node_seats (see
    test_top_edges_default_does_not_drop_nodes_on_a_dense_graph); sankey_flow
    must get the same guarantee.
    """
    from cellquorum.stages.cell_cell_communication.viz._plots import _select_sankey_flows

    lr = _fully_connected_lr_with_genes(9)
    kept = _select_sankey_flows(
        lr,
        source_col="source",
        ligand_col="ligand",
        receptor_col="receptor",
        target_col="target",
        weight_col="weight",
        top_k=15,
    )
    present = set(kept["source"]) | set(kept["target"])
    assert present == {str(i) for i in range(9)}


def test_sankey_flow_returns_figure_on_a_dense_graph():
    from cellquorum.stages.cell_cell_communication.viz._plots import (
        celltype_palette,
        sankey_flow,
    )

    lr = _fully_connected_lr_with_genes(9)
    pal = celltype_palette(list(set(lr["source"]) | set(lr["target"])))
    fig = sankey_flow(lr, palette=pal, top_k=15)
    assert isinstance(fig, Figure)


def test_chord_and_sankey_return_figure():
    from cellquorum.stages.cell_cell_communication.viz._plots import (
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

    from cellquorum.stages.cell_cell_communication.viz import _plots

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
    from cellquorum.stages.cell_cell_communication.viz._plots import curvature_network

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
    from cellquorum.stages.cell_cell_communication.viz._plots import curvature_network

    e = pd.DataFrame(columns=["source", "target", "ricci_curvature", "weight"])
    n = pd.DataFrame(columns=["node", "ricci_curvature"])
    assert isinstance(curvature_network(e, n), Figure)


def test_curvature_network_without_weight_col():
    from cellquorum.stages.cell_cell_communication.viz._plots import curvature_network

    edges = pd.DataFrame({"source": ["A"], "target": ["B"], "delta_curvature": [-0.1]})
    fig = curvature_network(
        edges, None, curvature_col="delta_curvature", node_curv_col="delta_curvature"
    )
    assert fig is not None


def test_curvature_network_strips_a_spurious_float_suffix_from_cluster_ids():
    """Real bug found on a rendered figure: node labels read "6.0" instead of
    "6". Cluster ids ("0", "1", ..., "6") all look numeric, so a real CSV
    read without dtype=str silently infers a float64 column -- pandas hands
    curvature_network float values, and str(6.0) == "6.0", not "6". This
    must render the clean integer-looking id regardless of whether the
    caller's DataFrame column happens to be numeric or string dtype.
    """
    from cellquorum.stages.cell_cell_communication.viz._plots import curvature_network

    edges = pd.DataFrame(
        {
            "source": [0.0, 1.0, 6.0],
            "target": [1.0, 6.0, 0.0],
            "ricci_curvature": [0.1, 0.2, 0.3],
            "weight": [1.0, 2.0, 3.0],
        }
    )
    nodes = pd.DataFrame({"node": [0.0, 1.0, 6.0], "ricci_curvature": [0.1, 0.2, 0.3]})
    fig = curvature_network(edges, nodes)
    ax = fig.axes[0]
    labels = {t.get_text() for t in ax.texts}
    assert labels == {"0", "1", "6"}
    assert not any(lbl.endswith(".0") for lbl in labels)


def test_curvature_network_does_not_drop_nodes_on_a_dense_graph():
    """Same bug already fixed for chord/sankey (_top_edges), never applied to
    curvature_network: a flat top_k cap on a real, densely-connected 9-cluster
    CCI network (36 edges) silently dropped 2 of 9 clusters from network_cci.png
    -- confirmed directly against a real fibroblast run's curvature CSVs."""
    from cellquorum.stages.cell_cell_communication.viz._plots import curvature_network

    nodes_list = [str(i) for i in range(9)]
    rows = []
    for i, a in enumerate(nodes_list):
        for b in nodes_list[i + 1 :]:
            rows.append({"source": a, "target": b, "ricci_curvature": 0.1, "weight": 100.0 + i})
    edges = pd.DataFrame(rows)
    nodes = pd.DataFrame({"node": nodes_list, "ricci_curvature": [0.1] * 9})

    fig = curvature_network(edges, nodes, top_k=15)

    ax = fig.axes[0]
    shown = {t.get_text() for t in ax.texts}
    assert shown == set(nodes_list)


def test_topology_facet_figures_returns_one_figure_per_metric():
    """One file per metric (house style: one plot = one file), not a combined
    4-panel grid -- each metric (who receives, who sends, who mediates, who's
    central) answers a different question a reader cares about separately."""
    from cellquorum.stages.cell_cell_communication.viz._plots import topology_facet_figures

    topo = pd.DataFrame(
        {
            "node": ["A", "B", "C"],
            "Listener": [1.0, 2.0, -1.0],
            "Influencer": [0.1, 0.2, 0.3],
            "Mediator": [0.0, 1.0, 2.0],
            "Pagerank": [0.2, 0.3, 0.5],
        }
    )
    figs = topology_facet_figures(topo, top_k=2)
    assert set(figs) == {"Listener", "Influencer", "Mediator", "Pagerank"}
    assert all(isinstance(f, Figure) for f in figs.values())


def test_topology_facet_figures_explains_a_degenerate_metric_instead_of_a_blank_panel():
    """Real bug, confirmed live: a real 9-cluster CCC network is essentially
    complete (every cluster signals every other), so betweenness centrality
    ("Mediator") is mathematically 0 for every node -- no node ever sits on a
    shortest path when direct edges connect everyone. That's a genuine result,
    not a computation bug, but the panel rendered as an empty, unexplained
    blank -- which reads as broken. A metric with zero variance across every
    node must say so, not render silently empty bars.
    """
    from cellquorum.stages.cell_cell_communication.viz._plots import topology_facet_figures

    topo = pd.DataFrame(
        {
            "node": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
            "Listener": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "Mediator": [0.0] * 9,
        }
    )
    figs = topology_facet_figures(topo, metric_cols=("Listener", "Mediator"), top_k=5)

    mediator_ax = figs["Mediator"].axes[0]
    texts = " ".join(t.get_text() for t in mediator_ax.texts)
    assert "no variation" in texts.lower()
    # The real metric renders normally, unaffected.
    listener_ax = figs["Listener"].axes[0]
    assert len(listener_ax.patches) > 0


def _tensor_factor_frames():
    """Matches tensor_c2c_method.py's real written shape: index=entity,
    columns=Factor 1..N."""
    contexts = pd.DataFrame(
        {"Factor 1": [0.2, 0.3, 0.25], "Factor 2": [0.4, 0.1, 0.3]},
        index=["P1_LE", "P1_Normal", "P2_LE"],
    )
    senders = pd.DataFrame(
        {"Factor 1": [0.5, 0.3, 0.2], "Factor 2": [0.1, 0.6, 0.3]}, index=["0", "1", "2"]
    )
    receivers = pd.DataFrame(
        {"Factor 1": [0.4, 0.4, 0.2], "Factor 2": [0.2, 0.3, 0.5]}, index=["0", "1", "2"]
    )
    lr_pairs = pd.DataFrame(
        {"Factor 1": [0.1, 0.05, 0.2, 0.15], "Factor 2": [0.3, 0.2, 0.05, 0.1]},
        index=["A^B", "C^D", "E^F", "G^H"],
    )
    return contexts, senders, receivers, lr_pairs


def test_plot_tensor_factors_returns_a_figure_with_one_row_per_factor():
    """Real gap, confirmed live: tensor_c2c writes 4 real factor-loading
    tables (contexts/senders/receivers/lr_pairs) and NOTHING ever visualized
    them -- zero figures existed for real, computed data. One row per
    factor x 4 columns (contexts/LR-pairs/senders/receivers) is the
    standard tensor-cell2cell factor-plot layout; unlike topology_facets,
    comparing loadings ACROSS factors side by side is the entire analytical
    point, so a combined grid is the deliberate, correct call here.
    """
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_tensor_factors

    contexts, senders, receivers, lr_pairs = _tensor_factor_frames()
    fig = plot_tensor_factors(contexts, senders, receivers, lr_pairs)

    assert isinstance(fig, Figure)
    # 2 factors x 4 columns.
    assert len(fig.axes) == 8


def test_plot_tensor_factors_empty_guarded():
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_tensor_factors

    empty = pd.DataFrame()
    contexts, senders, receivers, lr_pairs = _tensor_factor_frames()
    assert isinstance(plot_tensor_factors(empty, senders, receivers, lr_pairs), Figure)
    assert isinstance(plot_tensor_factors(contexts, empty, receivers, lr_pairs), Figure)


def test_interaction_dotplot_all_nan_weight():
    from cellquorum.stages.cell_cell_communication.viz._plots import interaction_dotplot

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


# ─── NicheNet-specific plots ────────────────────────────────────────────────


def test_ligand_activity_bar_orders_by_score_and_caps_top_n():
    from cellquorum.stages.cell_cell_communication.viz._plots import ligand_activity_bar

    activities = pd.DataFrame(
        {
            "test_ligand": ["TGFB2", "TGFB1", "IL6", "CCN2"],
            "aupr_corrected": [0.021, 0.019, 0.041, 0.005],
        }
    )
    fig = ligand_activity_bar(activities, top_n=2)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # Top-2 by score, and the y-tick labels carry the actual ligand identity.
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert set(labels) == {"IL6", "TGFB2"}


def test_ligand_activity_bar_empty_guarded():
    from cellquorum.stages.cell_cell_communication.viz._plots import ligand_activity_bar

    assert isinstance(ligand_activity_bar(pd.DataFrame()), Figure)
    assert isinstance(ligand_activity_bar(None), Figure)


def test_ligand_target_heatmap_returns_figure_with_ligand_and_target_axes():
    from cellquorum.stages.cell_cell_communication.viz._plots import ligand_target_heatmap

    weights = pd.DataFrame(
        {
            "ligand": ["TGFB2", "TGFB2", "IL6"],
            "target": ["COL1A1", "COL1A2", "COL1A1"],
            "weight": [0.24, 0.25, 0.10],
        }
    )
    fig = ligand_target_heatmap(weights)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert set(t.get_text() for t in ax.get_yticklabels()) == {"TGFB2", "IL6"}
    assert set(t.get_text() for t in ax.get_xticklabels()) == {"COL1A1", "COL1A2"}


def test_ligand_target_heatmap_empty_guarded():
    from cellquorum.stages.cell_cell_communication.viz._plots import ligand_target_heatmap

    assert isinstance(ligand_target_heatmap(pd.DataFrame()), Figure)
    assert isinstance(ligand_target_heatmap(None), Figure)


def test_sender_ligand_dotplot_returns_figure_with_sender_and_ligand_axes():
    from cellquorum.stages.cell_cell_communication.viz._plots import sender_ligand_dotplot

    expr = pd.DataFrame(
        {
            "sender": ["Perivascular", "Perivascular", "Reticular"],
            "ligand": ["TGFB2", "TGFB1", "TGFB2"],
            "fraction_expressing": [0.17, 0.40, 0.05],
            "mean_expression": [0.32, 0.69, 0.11],
        }
    )
    fig = sender_ligand_dotplot(expr)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert set(t.get_text() for t in ax.get_xticklabels()) == {"Perivascular", "Reticular"}
    assert set(t.get_text() for t in ax.get_yticklabels()) == {"TGFB2", "TGFB1"}


def test_sender_ligand_dotplot_empty_guarded():
    from cellquorum.stages.cell_cell_communication.viz._plots import sender_ligand_dotplot

    assert isinstance(sender_ligand_dotplot(pd.DataFrame()), Figure)
    assert isinstance(sender_ligand_dotplot(None), Figure)
