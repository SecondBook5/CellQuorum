import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx  # noqa: F401
import numpy as np
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


def test_chord_diagram_invokes_link_kws_handler_when_given():
    from cellquorum.stages.cell_cell_communication.viz._plots import celltype_palette, chord_diagram

    df = _lr(4)
    pal = celltype_palette(list(set(df["source"]) | set(df["target"])))
    calls = []

    def handler(a, b):
        calls.append((a, b))
        return {}

    fig = chord_diagram(df, palette=pal, link_kws_handler=handler)
    assert isinstance(fig, Figure)
    assert len(calls) > 0


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


def _mnn_sample_and_source_dfs():
    sample_df = pd.DataFrame(
        {
            "id": ["TGFB1_TGFBR1_X0_X3"] * 4 + ["CCN2_FGFR2_X3_X6"] * 4,
            "sample": ["P1_LE", "P2_LE", "P1_N", "P2_N"] * 2,
            "group": ["Lymphedema", "Lymphedema", "Normal", "Normal"] * 2,
            "sender": ["0"] * 4 + ["3"] * 4,
            "receiver": ["3"] * 4 + ["6"] * 4,
            "ligand": ["TGFB1"] * 4 + ["CCN2"] * 4,
            "receptor": ["TGFBR1"] * 4 + ["FGFR2"] * 4,
            "scaled_LR_pb_prod": [1.1, 0.9, -0.5, -0.3, 0.6, 0.4, -0.2, -0.1],
            "keep_sender_receiver": ["Sender & Receiver present"] * 3
            + ["Receiver absent"]
            + ["Sender & Receiver present"] * 4,
        }
    )
    source_df = pd.DataFrame(
        {
            "id": [
                "TGFB1_TGFBR1_X0_X3",
                "TGFB1_TGFBR1_X0_X3",
                "CCN2_FGFR2_X3_X6",
                "CCN2_FGFR2_X3_X6",
            ],
            "sender": ["0", "0", "3", "3"],
            "receiver": ["3", "3", "6", "6"],
            "ligand": ["TGFB1", "TGFB1", "CCN2", "CCN2"],
            "receptor": ["TGFBR1", "TGFBR1", "FGFR2", "FGFR2"],
            "group": ["Lymphedema", "Normal", "Lymphedema", "Normal"],
            "activity": [0.04, 0.01, 0.03, 0.02],
            "activity_scaled": [1.5, -1.5, 0.8, -0.8],
            "direction_regulation": ["up", "down", "up", "down"],
            "scaled_pb_ligand": [1.2, -1.2, 0.5, -0.5],
            "scaled_pb_receptor": [0.9, -0.9, 0.3, -0.3],
            "fraction_ligand_group": [0.8, 0.3, 0.6, 0.4],
            "fraction_receptor_group": [0.7, 0.2, 0.5, 0.35],
            "prioritization_score": [0.95, 0.95, 0.80, 0.80],
        }
    )
    return sample_df, source_df


def _pair_labels(ax):
    """Panel 1's per-row ligand-receptor labels: manually-drawn ax.text()
    calls (not set_yticklabels), identified by " - " with no "->" (route
    labels use "->"; group headers and significance markers don't have
    either pattern).
    """
    return [t.get_text() for t in ax.texts if " - " in t.get_text() and "->" not in t.get_text()]


def test_plot_mnn_sample_activity_returns_a_four_panel_figure():
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df, source_df = _mnn_sample_and_source_dfs()
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=20)
    assert isinstance(fig, Figure)
    # 4 data panels + legend column.
    assert len(fig.axes) >= 5


def test_plot_mnn_sample_activity_rows_name_sender_receiver_and_lr_pair():
    # Both the route (sender->receiver) box label and the ligand-receptor
    # pair label render as manual ax.text() calls in panel 1 (axes[0])'s
    # own negative-x margin, not via set_yticklabels -- a separate
    # route-box axes collided with panel 1's real y-tick labels on a real
    # render, so everything now lives in one coordinate system.
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df, source_df = _mnn_sample_and_source_dfs()
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=20)
    row_text = " ".join(_pair_labels(fig.axes[0]))
    assert "TGFB1 - TGFBR1" in row_text
    assert "CCN2 - FGFR2" in row_text
    route_text = " ".join(t.get_text() for t in fig.axes[0].texts if "->" in t.get_text())
    assert "0 -> 3" in route_text
    assert "3 -> 6" in route_text


def test_plot_mnn_sample_activity_relabels_clusters_in_row_names():
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df, source_df = _mnn_sample_and_source_dfs()
    fig = plot_mnn_sample_activity(
        sample_df, source_df, top_n=20, cluster_labels={"0": "Perivascular", "3": "Progenitor-like"}
    )
    route_text = " ".join(t.get_text() for t in fig.axes[0].texts if "->" in t.get_text())
    assert "Perivascular" in route_text
    assert "Progenitor-like" in route_text


def test_plot_mnn_sample_activity_caps_to_top_n_interactions():
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df, source_df = _mnn_sample_and_source_dfs()
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=1)
    row_text = _pair_labels(fig.axes[0])
    # top_n=1 -> only the strongest id (TGFB1-TGFBR1, score 0.95) survives.
    assert len(row_text) == 1
    assert "TGFB1" in row_text[0]


def test_plot_mnn_sample_activity_groups_samples_by_condition_not_alphabetically():
    # Confirmed against a real render: plain alphabetical sample sort
    # interleaves LE/Normal by patient (P10_LE, P10_Normal, P12_LE, ...),
    # unlike the reference figure's per-group facet columns. Samples for
    # the same group must render as a contiguous block.
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df = pd.DataFrame(
        {
            "id": ["L_R_X0_X3"] * 4,
            "sample": ["P10_LE", "P10_Normal", "P1_LE", "P1_Normal"],
            "group": ["Lymphedema", "Normal", "Lymphedema", "Normal"],
            "sender": ["0"] * 4,
            "receiver": ["3"] * 4,
            "ligand": ["L"] * 4,
            "receptor": ["R"] * 4,
            "scaled_LR_pb_prod": [1.0, -1.0, 0.5, -0.5],
            "keep_sender_receiver": ["Sender & Receiver present"] * 4,
        }
    )
    source_df = pd.DataFrame(
        {
            "id": ["L_R_X0_X3", "L_R_X0_X3"],
            "sender": ["0", "0"],
            "receiver": ["3", "3"],
            "ligand": ["L", "L"],
            "receptor": ["R", "R"],
            "group": ["Lymphedema", "Normal"],
            "activity": [0.03, 0.02],
            "activity_scaled": [1.0, -1.0],
            "scaled_pb_ligand": [1.0, -1.0],
            "scaled_pb_receptor": [0.5, -0.5],
            "fraction_ligand_group": [0.7, 0.4],
            "fraction_receptor_group": [0.6, 0.3],
            "prioritization_score": [0.9, 0.9],
        }
    )
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=20)
    xticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    le_positions = [i for i, s in enumerate(xticklabels) if "LE" in s]
    n_positions = [i for i, s in enumerate(xticklabels) if "Normal" in s]
    # Every LE position comes before every Normal position (or vice versa) --
    # i.e. the two groups are contiguous blocks, not interleaved by patient.
    assert max(le_positions) < min(n_positions) or max(n_positions) < min(le_positions)


def test_plot_mnn_sample_activity_heatmap_panels_share_scatter_panels_row_order():
    # Panel 1 (scatter) puts row 0 at the BOTTOM (matplotlib's default,
    # non-inverted y-axis). imshow's default origin="upper" puts array row 0
    # at the TOP -- confirmed via ax.get_ylim() on a real render. Left
    # unfixed, panels 2/3 (imshow) show each row's data on the opposite
    # horizontal line from its real sample-product/specificity row.
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df, source_df = _mnn_sample_and_source_dfs()
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=20)
    scatter_ylim = fig.axes[0].get_ylim()
    heatmap_ylim = fig.axes[1].get_ylim()
    # Both non-inverted (row 0 at the bottom): ylim[0] < ylim[1] on both.
    assert scatter_ylim[0] < scatter_ylim[1]
    assert heatmap_ylim[0] < heatmap_ylim[1]


def test_plot_mnn_sample_activity_groups_rows_by_sender_receiver_with_separators():
    # Rows for the same sender->receiver route should sit contiguously
    # (already true, sorted rows), with a visible separator marking where
    # one route's block ends and the next begins -- confirmed against a
    # real render as too flat/cramped to tell routes apart at a glance.
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df, source_df = _mnn_sample_and_source_dfs()
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=20)
    ax = fig.axes[1]
    # Two distinct routes in the fixture (0->3 and 3->6) -> at least one
    # separator line between them.
    assert len(ax.lines) >= 1


def test_plot_mnn_sample_activity_labels_sample_groups_above_panel_one():
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df, source_df = _mnn_sample_and_source_dfs()
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=20)
    header_text = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "Lymphedema" in header_text
    assert "Normal" in header_text


def test_plot_mnn_sample_activity_sorts_rows_by_score_within_route_block():
    # Within the same sender->receiver route, the stronger interaction
    # (higher prioritization_score) should render closer to the top of
    # that route's block, not just alphabetically by ligand-receptor name.
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    def _rows(id_, sender, receiver, ligand, receptor, score):
        s = pd.DataFrame(
            {
                "id": [id_],
                "sample": ["P1_LE"],
                "group": ["Lymphedema"],
                "sender": [sender],
                "receiver": [receiver],
                "ligand": [ligand],
                "receptor": [receptor],
                "scaled_LR_pb_prod": [0.5],
                "keep_sender_receiver": ["Sender & Receiver present"],
            }
        )
        g = pd.DataFrame(
            {
                "id": [id_],
                "sender": [sender],
                "receiver": [receiver],
                "ligand": [ligand],
                "receptor": [receptor],
                "group": ["Lymphedema"],
                "activity": [0.02],
                "activity_scaled": [0.5],
                "scaled_pb_ligand": [0.5],
                "scaled_pb_receptor": [0.5],
                "fraction_ligand_group": [0.5],
                "fraction_receptor_group": [0.5],
                "prioritization_score": [score],
            }
        )
        return s, g

    # Names deliberately reversed vs. score, so a plain alphabetical-label
    # sort would rank them the OPPOSITE way from a score-based sort --
    # only the real fix passes this.
    s1, g1 = _rows("Z_Y_X0_X3", "0", "3", "Z", "Y", 0.5)  # alphabetically last, lower score
    s2, g2 = _rows("A_B_X0_X3", "0", "3", "A", "B", 0.9)  # alphabetically first, higher score
    sample_df = pd.concat([s1, s2], ignore_index=True)
    source_df = pd.concat([g1, g2], ignore_index=True)
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=20)
    labels = _pair_labels(fig.axes[0])
    assert labels.index("A - B") > labels.index("Z - Y")


def test_plot_mnn_sample_activity_marks_significant_cells():
    # p_val_ligand/p_val_receptor are real DE significance columns
    # multinichenetr already computes but nothing surfaced anywhere.
    # Uses raw (unadjusted) p-values with an OR across ligand/receptor,
    # not p_adj with AND: confirmed against real data that the adjusted-AND
    # criterion is too conservative to ever fire on a top-N shortlist
    # (140k+ total LR-sender-receiver tests means heavy multiple-testing
    # correction; 0 of a real top-20 passed adjusted-OR, 14 of 20 passed
    # raw-OR).
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df, source_df = _mnn_sample_and_source_dfs()
    source_df["p_val_ligand"] = [0.001, 0.5, 0.5, 0.5]
    source_df["p_val_receptor"] = [0.5, 0.5, 0.5, 0.5]
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=20)
    marker_text = "".join(t.get_text() for t in fig.axes[1].texts)
    assert "*" in marker_text


def test_plot_mnn_sample_activity_legend_has_labeled_colorbars_and_size_keys():
    # Confirmed against the reference figure: a reader had no way to tell
    # what a specific color or dot size actually MEANT quantitatively --
    # inline panel titles alone don't give that. Each colorbar needs its
    # own title and real numeric tick values, not just a bare gradient.
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df, source_df = _mnn_sample_and_source_dfs()
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=20)
    legend_text = " ".join(t.get_text() for t in fig.axes[4].texts)
    assert "pseudobulk" in legend_text.lower()
    assert "specificity" in legend_text.lower()
    assert "fraction of expression" in legend_text.lower()
    assert "sufficient presence" in legend_text.lower()
    # Colorbars are inset axes (children of the legend axes, not top-level
    # entries in fig.axes) -- at least 4 of them, one per color scale.
    assert len(fig.axes[4].child_axes) >= 4


def test_plot_mnn_sample_activity_colorbars_show_real_numeric_ticks():
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df, source_df = _mnn_sample_and_source_dfs()
    fig = plot_mnn_sample_activity(sample_df, source_df, top_n=20)
    # Colorbars are inset axes (children of the legend axes), not top-level
    # entries in fig.axes.
    colorbar_axes = fig.axes[4].child_axes
    assert len(colorbar_axes) >= 4
    tick_texts = [t.get_text() for cax in colorbar_axes for t in cax.get_yticklabels()]
    assert any(t.strip() for t in tick_texts)


def test_plot_mnn_sample_activity_long_route_names_do_not_collide_with_pair_labels():
    # Real bug, confirmed on a real render: a box-width margin sized with
    # too small a per-character estimate let LONG route names (e.g.
    # "Perivascular -> Inflammatory") overflow their box and visually
    # merge with the neighboring row's own "ligand - receptor" label.
    # Rendered here with canvas.draw() so get_window_extent() reflects
    # real text metrics, not just the box-width constant.
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    sample_df = pd.DataFrame(
        {
            "id": ["L_R_X0_X3"],
            "sample": ["P1_LE"],
            "group": ["Lymphedema"],
            "sender": ["0"],
            "receiver": ["3"],
            "ligand": ["ANGPTL4"],
            "receptor": ["ITGB3"],
            "scaled_LR_pb_prod": [0.5],
            "keep_sender_receiver": ["Sender & Receiver present"],
        }
    )
    source_df = pd.DataFrame(
        {
            "id": ["L_R_X0_X3"],
            "sender": ["0"],
            "receiver": ["3"],
            "ligand": ["ANGPTL4"],
            "receptor": ["ITGB3"],
            "group": ["Lymphedema"],
            "activity": [0.02],
            "activity_scaled": [0.5],
            "scaled_pb_ligand": [0.5],
            "scaled_pb_receptor": [0.5],
            "fraction_ligand_group": [0.5],
            "fraction_receptor_group": [0.5],
            "prioritization_score": [0.9],
        }
    )
    fig = plot_mnn_sample_activity(
        sample_df,
        source_df,
        top_n=20,
        cluster_labels={"0": "Perivascular", "3": "Inflammatory"},  # a genuinely long route name
    )
    fig.canvas.draw()
    ax = fig.axes[0]
    route_text = next(t for t in ax.texts if "->" in t.get_text())
    pair_text = next(t for t in ax.texts if "ANGPTL4 - ITGB3" == t.get_text())
    route_bbox = route_text.get_window_extent()
    pair_bbox = pair_text.get_window_extent()
    assert (
        route_bbox.x1 <= pair_bbox.x0
    ), f"route label {route_bbox} overlaps pair label {pair_bbox}"


def test_plot_mnn_sample_activity_empty_guarded():
    from cellquorum.stages.cell_cell_communication.viz._plots import plot_mnn_sample_activity

    empty = pd.DataFrame()
    sample_df, source_df = _mnn_sample_and_source_dfs()
    assert isinstance(plot_mnn_sample_activity(empty, source_df), Figure)
    assert isinstance(plot_mnn_sample_activity(sample_df, empty), Figure)
    assert isinstance(plot_mnn_sample_activity(None, None), Figure)


# ─── MultiNicheNet-specific plots ───────────────────────────────────────────


def _mnn_prioritization_df():
    return pd.DataFrame(
        {
            "group": ["Lymphedema", "Lymphedema", "Normal", "Normal"],
            "sender": ["0", "3", "0", "6"],
            "receiver": ["3", "6", "3", "0"],
            "ligand": ["TGFB1", "CCN2", "IL6", "POSTN"],
            "receptor": ["TGFBR1", "FGFR2", "IL6R", "ITGAV"],
            "prioritization_score": [0.95, 0.92, 0.80, 0.78],
        }
    )


def test_mnn_top_lr_lollipop_orders_by_score_and_caps_top_n():
    from cellquorum.stages.cell_cell_communication.viz._plots import mnn_top_lr_lollipop

    df = _mnn_prioritization_df()
    fig = mnn_top_lr_lollipop(df, top_n=1)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # top_n=1 per group (Lymphedema, Normal) -> the single strongest row each:
    # TGFB1-TGFBR1 (0.95 > 0.92 CCN2-FGFR2), IL6-IL6R (0.80 > 0.78 POSTN-ITGAV).
    assert set(labels) == {"TGFB1 - TGFBR1", "IL6 - IL6R"}


def test_mnn_top_lr_lollipop_names_the_actual_lr_pair_not_just_cell_types():
    # This is the whole point over the circos plot: a reader must be able to
    # tell WHICH ligand-receptor pair each row is, not just which cell types.
    from cellquorum.stages.cell_cell_communication.viz._plots import mnn_top_lr_lollipop

    df = _mnn_prioritization_df()
    fig = mnn_top_lr_lollipop(df, top_n=10)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert "TGFB1 - TGFBR1" in labels
    assert "CCN2 - FGFR2" in labels


def test_mnn_top_lr_lollipop_empty_guarded():
    from cellquorum.stages.cell_cell_communication.viz._plots import mnn_top_lr_lollipop

    assert isinstance(mnn_top_lr_lollipop(pd.DataFrame()), Figure)
    assert isinstance(mnn_top_lr_lollipop(None), Figure)


def test_mnn_top_lr_lollipop_height_stays_reasonable_at_real_scale():
    # A real run's top_n=30 rendered at 9in tall (0.3in/row) read as an
    # awkwardly stretched column next to the wide legend -- height per row
    # needs a lower ceiling.
    from cellquorum.stages.cell_cell_communication.viz._plots import mnn_top_lr_lollipop

    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame(
        {
            "sender": [str(i % 6) for i in range(n)],
            "receiver": [str((i + 1) % 6) for i in range(n)],
            "ligand": [f"L{i}" for i in range(n)],
            "receptor": [f"R{i}" for i in range(n)],
            "prioritization_score": rng.uniform(0.8, 0.95, size=n),
        }
    )
    fig = mnn_top_lr_lollipop(df, top_n=30)
    assert fig.get_size_inches()[1] <= 7.0


def test_mnn_top_lr_lollipop_legend_sits_outside_the_axes():
    # A real MultiNicheNet run's top-N routinely has 20-30 distinct
    # sender->receiver routes -- loc="lower right" placed that legend ON TOP
    # of the data (confirmed by rendering against real project data), not
    # beside it. The legend must anchor outside the axes bounding box.
    from cellquorum.stages.cell_cell_communication.viz._plots import mnn_top_lr_lollipop

    df = _mnn_prioritization_df()
    fig = mnn_top_lr_lollipop(df, top_n=10)
    legend = fig.axes[0].get_legend()
    anchor = legend.get_bbox_to_anchor()._bbox
    assert anchor.x0 >= 1.0


def test_mnn_top_lr_lollipop_relabels_sender_receiver_in_legend():
    from cellquorum.stages.cell_cell_communication.viz._plots import mnn_top_lr_lollipop

    df = _mnn_prioritization_df()
    fig = mnn_top_lr_lollipop(
        df, top_n=10, cluster_labels={"0": "Perivascular", "3": "Progenitor-like"}
    )
    legend_labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("Perivascular" in lab for lab in legend_labels)


def _mnn_focused_df():
    return pd.DataFrame(
        {
            "sender": ["6", "6", "0", "1", "2", "3"],
            "receiver": ["3", "0", "1", "2", "3", "4"],
            "ligand": ["WIF1", "COL18A1", "TGFB1", "IL6", "POSTN", "CCN2"],
            "receptor": ["FZD5", "ITGAV", "TGFBR1", "IL6R", "ITGAV", "FGFR2"],
            "prioritization_score": [0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
        }
    )


def test_cluster_focused_circos_filters_to_only_edges_touching_focus():
    from cellquorum.stages.cell_cell_communication.viz._plots import cluster_focused_circos

    df = _mnn_focused_df()
    palette = {c: "#123456" for c in "0123456"}
    fig = cluster_focused_circos(df, "6", palette=palette)
    text = " ".join(t.get_text() for t in fig.texts)
    # Rows touching "6" (sender or receiver): the WIF1/COL18A1 pair. Rows
    # that never touch "6" (IL6-IL6R, POSTN-ITGAV, CCN2-FGFR2) must not
    # appear -- this is the whole point of focusing.
    assert "WIF1" in text and "COL18A1" in text
    assert "IL6" not in text
    assert "POSTN" not in text
    assert "CCN2" not in text


def test_cluster_focused_circos_lists_the_lr_pair_for_each_ribbon():
    from cellquorum.stages.cell_cell_communication.viz._plots import cluster_focused_circos

    df = _mnn_focused_df()
    palette = {c: "#123456" for c in "0123456"}
    fig = cluster_focused_circos(df, "6", palette=palette)
    text = " ".join(t.get_text() for t in fig.texts)
    assert "WIF1 - FZD5" in text
    assert "COL18A1 - ITGAV" in text


def test_cluster_focused_circos_caps_to_top_k_partners():
    from cellquorum.stages.cell_cell_communication.viz._plots import cluster_focused_circos

    rng_rows = [
        {
            "sender": "6",
            "receiver": str(i),
            "ligand": f"L{i}",
            "receptor": f"R{i}",
            "prioritization_score": 1.0 - 0.01 * i,
        }
        for i in range(10)
    ]
    df = pd.DataFrame(rng_rows)
    palette = {str(i): "#123456" for i in range(10)} | {"6": "#654321"}
    fig = cluster_focused_circos(df, "6", palette=palette, top_k=3)
    text = " ".join(t.get_text() for t in fig.texts)
    assert text.count(" - ") == 3


def test_cluster_focused_circos_empty_guarded():
    from cellquorum.stages.cell_cell_communication.viz._plots import cluster_focused_circos

    palette = {"6": "#123456"}
    assert isinstance(cluster_focused_circos(pd.DataFrame(), "6", palette=palette), Figure)
    assert isinstance(cluster_focused_circos(None, "6", palette=palette), Figure)


def test_cluster_focused_circos_relabels_the_ring_nodes(monkeypatch):
    # Confirmed against a real render: passing cluster_labels only reached
    # the side legend, not the ring's own sector labels -- the ring kept
    # showing bare codes ("6") even with a real display-name mapping given.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("pycirclize"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    from cellquorum.stages.cell_cell_communication.viz._plots import cluster_focused_circos

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = _mnn_focused_df()
    palette = {c: "#123456" for c in "0123456"}
    fig = cluster_focused_circos(
        df, "6", palette=palette, cluster_labels={"6": "Papillary", "3": "Progenitor-like"}
    )
    node_text = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "Papillary" in node_text
    assert "Progenitor-like" in node_text


def test_cluster_focused_circos_has_a_title():
    from cellquorum.stages.cell_cell_communication.viz._plots import cluster_focused_circos

    df = _mnn_focused_df()
    palette = {c: "#123456" for c in "0123456"}
    fig = cluster_focused_circos(
        df, "6", palette=palette, cluster_labels={"6": "Papillary"}, title="Papillary (Lymphedema)"
    )
    assert fig._suptitle is not None
    assert "Papillary" in fig._suptitle.get_text()


def test_cluster_focused_circos_styles_outgoing_and_incoming_ribbons_differently():
    # Direction (is the focus cluster sending or receiving) isn't visible
    # from ribbon color alone (color = route, not direction) -- confirmed
    # against a real render. The handler chord_diagram() is called with
    # must style a (focus, other) outgoing link differently from an
    # (other, focus) incoming one.
    from cellquorum.stages.cell_cell_communication.viz._plots import cluster_focused_circos

    df = _mnn_focused_df()
    palette = {c: "#123456" for c in "0123456"}
    captured = {}

    import cellquorum.stages.cell_cell_communication.viz._plots as plots_mod

    real_chord = plots_mod.chord_diagram

    def spy(*args, **kwargs):
        captured["handler"] = kwargs.get("link_kws_handler")
        return real_chord(*args, **kwargs)

    import unittest.mock as mock

    with mock.patch.object(plots_mod, "chord_diagram", spy):
        cluster_focused_circos(df, "6", palette=palette, cluster_labels={"6": "Papillary"})

    handler = captured["handler"]
    assert handler is not None
    outgoing_style = handler("Papillary", "Progenitor-like")
    incoming_style = handler("Progenitor-like", "Papillary")
    assert outgoing_style != incoming_style


def test_cluster_focused_circos_legend_explains_width_and_direction_encoding():
    from cellquorum.stages.cell_cell_communication.viz._plots import cluster_focused_circos

    df = _mnn_focused_df()
    palette = {c: "#123456" for c in "0123456"}
    fig = cluster_focused_circos(df, "6", palette=palette, cluster_labels={"6": "Papillary"})
    legend_text = " ".join(t.get_text() for t in fig.texts).lower()
    assert "prioritization" in legend_text or "score" in legend_text
    assert "receiv" in legend_text or "sen" in legend_text


def test_cluster_focused_circos_empty_when_focus_has_no_edges():
    from cellquorum.stages.cell_cell_communication.viz._plots import cluster_focused_circos

    df = _mnn_focused_df()
    palette = {c: "#123456" for c in "0123456789"}
    fig = cluster_focused_circos(df, "9", palette=palette)  # "9" touches nothing
    assert fig.texts == [] or all(not t.get_text() for t in fig.texts)


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


def test_ligand_activity_bar_bars_are_colored_by_their_own_score():
    from cellquorum.stages.cell_cell_communication.viz._plots import ligand_activity_bar

    activities = pd.DataFrame(
        {
            "test_ligand": ["TGFB2", "TGFB1", "IL6"],
            "aupr_corrected": [0.021, 0.019, 0.041],
        }
    )
    fig = ligand_activity_bar(activities, top_n=3)
    ax = fig.axes[0]
    colors = {tuple(p.get_facecolor()) for p in ax.patches}
    # A flat single color (the old behavior) collapses to one distinct RGBA;
    # a real gradient keyed on score gives each differently-scored bar its
    # own color.
    assert len(colors) == 3


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


def test_ligand_target_heatmap_uses_a_single_hue_purple_scale():
    from cellquorum.stages.cell_cell_communication.viz._plots import ligand_target_heatmap

    weights = pd.DataFrame(
        {
            "ligand": ["TGFB2", "TGFB2"],
            "target": ["COL1A1", "COL1A2"],
            "weight": [0.24, 0.25],
        }
    )
    fig = ligand_target_heatmap(weights)
    im = fig.axes[0].images[0]
    # viridis (the shared house sequential map) reads dark-purple-to-yellow
    # and drowns a small, mostly-near-zero regulatory-potential matrix; a
    # single-hue white->purple ramp (nichenetr's own convention) is what
    # reads cleanly here.
    assert im.get_cmap().name == "Purples"


def test_ligand_target_heatmap_empty_guarded():
    from cellquorum.stages.cell_cell_communication.viz._plots import ligand_target_heatmap

    assert isinstance(ligand_target_heatmap(pd.DataFrame()), Figure)
    assert isinstance(ligand_target_heatmap(None), Figure)


def test_ligand_receptor_heatmap_returns_figure_with_ligand_and_receptor_axes():
    from cellquorum.stages.cell_cell_communication.viz._plots import ligand_receptor_heatmap

    weights = pd.DataFrame(
        {
            "ligand": ["TGFB1", "TGFB1", "IL6"],
            "receptor": ["TGFBR1", "TGFBR2", "IL6R"],
            "weight": [0.9, 0.7, 1.5],
        }
    )
    fig = ligand_receptor_heatmap(weights)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert set(t.get_text() for t in ax.get_yticklabels()) == {"TGFB1", "IL6"}
    assert set(t.get_text() for t in ax.get_xticklabels()) == {"TGFBR1", "TGFBR2", "IL6R"}


def test_ligand_receptor_heatmap_uses_a_distinct_single_hue_scale_from_ligand_target():
    from cellquorum.stages.cell_cell_communication.viz._plots import (
        ligand_receptor_heatmap,
        ligand_target_heatmap,
    )

    weights = pd.DataFrame({"ligand": ["TGFB1"], "receptor": ["TGFBR1"], "weight": [0.9]})
    targets = pd.DataFrame({"ligand": ["TGFB1"], "target": ["COL1A1"], "weight": [0.24]})
    lr_cmap = ligand_receptor_heatmap(weights).axes[0].images[0].get_cmap().name
    lt_cmap = ligand_target_heatmap(targets).axes[0].images[0].get_cmap().name
    assert lr_cmap != lt_cmap


def test_ligand_receptor_heatmap_empty_guarded():
    from cellquorum.stages.cell_cell_communication.viz._plots import ligand_receptor_heatmap

    assert isinstance(ligand_receptor_heatmap(pd.DataFrame()), Figure)
    assert isinstance(ligand_receptor_heatmap(None), Figure)


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


def test_drawing_a_chord_never_changes_matplotlibs_save_settings():
    """pycirclize rewrites savefig.pad_inches (and others) when it is first imported.

    Left alone, whichever figure imports it first is saved with wide margins and every later
    one is not, so a chord's margins would depend on what ran before it.
    """
    import sys

    import matplotlib as mpl

    from cellquorum.stages.cell_cell_communication.viz._plots import chord_diagram

    for name in [m for m in sys.modules if m == "pycirclize" or m.startswith("pycirclize.")]:
        del sys.modules[name]  # force the first-import path again
    mpl.rcParams["savefig.pad_inches"] = 0.1
    mpl.rcParams["savefig.bbox"] = "tight"
    before = (mpl.rcParams["savefig.pad_inches"], mpl.rcParams["savefig.bbox"])
    edges = pd.DataFrame(
        {"source": ["A", "B", "C"], "target": ["B", "C", "A"], "weight": [1.0, 0.8, 0.6]}
    )
    chord_diagram(edges, palette={"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}, top_k=5)
    assert (mpl.rcParams["savefig.pad_inches"], mpl.rcParams["savefig.bbox"]) == before
