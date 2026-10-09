"""Tests for the radial DoRothEA/CollecTRI GRN figure and its curvature variant.

Ports the reference lab figure (plot_grn_radial.py / grn_subcluster.py, the
"cool circle plot" the collaborator specifically asked CellQuorum to match)
into engine primitives: TF activity computed via decoupler ULM on a DE
contrast (not per-cell AUCell), sectors clustered by shared targets, and a
radial layout shared with the curvature variant of the same figure.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest
from matplotlib.figure import Figure
from optional_deps import requires

from cellquorum.stages.gene_regulation.grn.radial_figures import (
    _bezier,
    _cluster_by_shared_children,
    _layout_sectors,
    radial_grn,
    radial_regulon_curvature,
    select_curvature_targets,
    select_tf_targets,
    tf_activity_from_de,
)


def _tiny_de() -> pd.DataFrame:
    genes = [f"G{i}" for i in range(20)]
    return pd.DataFrame(
        {
            "gene": genes,
            "logFC": [2.0 - 0.1 * i for i in range(20)],
            "PValue": [0.001] * 10 + [0.5] * 10,
            "FDR": [0.01] * 10 + [0.6] * 10,
        }
    )


def _tiny_net() -> pd.DataFrame:
    genes = [f"G{i}" for i in range(20)]
    return pd.DataFrame(
        {
            "source": ["TF1"] * 6 + ["TF2"] * 6 + ["TF3"] * 4,
            "target": genes[0:6] + genes[4:10] + genes[10:14],
            "weight": [1.0] * 16,
        }
    )


# ---------------------------------------------------------------------------
# tf_activity_from_de
# ---------------------------------------------------------------------------


@requires("decoupler")
def test_tf_activity_from_de_returns_one_row_per_tf_in_the_net():
    tf = tf_activity_from_de(_tiny_de(), _tiny_net(), tmin=2)

    assert set(tf.index) == {"TF1", "TF2", "TF3"}
    assert {"score", "pvalue", "padj", "significant"}.issubset(tf.columns)


@requires("decoupler")
def test_tf_activity_from_de_scores_the_tf_driving_the_strongest_up_genes_highest():
    """TF1 regulates G0-G5, the most strongly up-regulated genes in the DE
    table; TF3 regulates G10-G13, near the bottom of the ranking. A real
    contrast-level activity score should rank TF1 above TF3."""
    tf = tf_activity_from_de(_tiny_de(), _tiny_net(), tmin=2)

    assert tf.loc["TF1", "score"] > tf.loc["TF3", "score"]


@requires("decoupler")
def test_tf_activity_from_de_empty_de_returns_empty_frame():
    empty = pd.DataFrame(columns=["gene", "logFC", "PValue"])

    tf = tf_activity_from_de(empty, _tiny_net(), tmin=2)

    assert tf.empty


# ---------------------------------------------------------------------------
# select_tf_targets
# ---------------------------------------------------------------------------


@requires("decoupler")
def test_select_tf_targets_picks_top_n_tfs_by_significance_then_activity():
    tf = tf_activity_from_de(_tiny_de(), _tiny_net(), tmin=2)
    de = _tiny_de().set_index("gene")

    tfs, edges, targets = select_tf_targets(tf, de, _tiny_net(), n_tf=2, per_tf=10)

    assert len(tfs) == 2
    assert tfs[0] == "TF1"  # strongest real signal, must sort first


@requires("decoupler")
def test_select_tf_targets_caps_targets_per_tf():
    tf = tf_activity_from_de(_tiny_de(), _tiny_net(), tmin=2)
    de = _tiny_de().set_index("gene")

    tfs, edges, targets = select_tf_targets(tf, de, _tiny_net(), n_tf=3, per_tf=2)

    per_tf_counts: dict[str, int] = {}
    for t, _g, _sign, _lfc in edges:
        per_tf_counts[t] = per_tf_counts.get(t, 0) + 1
    assert all(c <= 2 for c in per_tf_counts.values())


@requires("decoupler")
def test_select_tf_targets_drops_targets_absent_from_de_table():
    tf = tf_activity_from_de(_tiny_de(), _tiny_net(), tmin=2)
    de = _tiny_de().set_index("gene")
    net_with_ghost = pd.concat(
        [_tiny_net(), pd.DataFrame({"source": ["TF1"], "target": ["GHOST"], "weight": [1.0]})]
    )

    _tfs, edges, _targets = select_tf_targets(tf, de, net_with_ghost, n_tf=3, per_tf=10)

    assert all(g != "GHOST" for _t, g, _s, _l in edges)


@requires("decoupler")
def test_select_tf_targets_empty_inputs_return_empty():
    empty_tf = pd.DataFrame(columns=["score", "pvalue", "padj", "significant"])
    tfs, edges, targets = select_tf_targets(
        empty_tf, pd.DataFrame(columns=["logFC", "FDR"]), _tiny_net(), n_tf=3, per_tf=5
    )
    assert tfs == []
    assert edges == []
    assert targets == {}


# ---------------------------------------------------------------------------
# shared radial layout helpers
# ---------------------------------------------------------------------------


@requires("decoupler")
def test_cluster_by_shared_children_groups_members_with_overlapping_children():
    """TF1 and TF2 share 2/10 targets (weak overlap); TF3 shares none with
    either -- with n_sectors=2, TF3 must land in its own sector."""
    children_of = {
        "TF1": {"G0", "G1", "G2", "G3"},
        "TF2": {"G2", "G3", "G4", "G5"},
        "TF3": {"G10", "G11"},
    }
    sec_of, k = _cluster_by_shared_children(["TF1", "TF2", "TF3"], children_of, n_sectors=2)

    assert sec_of["TF3"] != sec_of["TF1"]
    assert k <= 2


@requires("decoupler")
def test_cluster_by_shared_children_falls_back_when_target_sets_are_fully_disjoint():
    """Real bug found on the real fibroblast pySCENIC regulon graph: TFs
    pulled from a data-driven regulon graph often share NO targets at all
    (unlike DoRothEA's curated net, where shared pathway targets are
    common), which makes every pairwise Jaccard distance exactly 1.0 -- a
    degenerate, constant distance matrix. Average-linkage + fcluster's
    maxclust criterion collapses that into ONE cluster regardless of
    n_sectors (confirmed directly against scipy), producing a figure with
    only one giant sector and a misleading '4 target-sharing modules'
    subtitle. Must fall back to a real multi-sector split instead."""
    children_of = {f"TF{i}": {f"G{i}_{j}" for j in range(3)} for i in range(12)}

    sec_of, k = _cluster_by_shared_children(list(children_of), children_of, n_sectors=4)

    assert len(set(sec_of.values())) > 1
    assert k > 1


@requires("decoupler")
def test_cluster_by_shared_children_never_exceeds_member_count():
    children_of = {"TF1": {"G0"}}
    sec_of, k = _cluster_by_shared_children(["TF1"], children_of, n_sectors=4)

    assert k == 1
    assert sec_of == {"TF1": 0} or sec_of == {"TF1": 1}


@requires("decoupler")
def test_layout_sectors_places_every_member_and_child_at_a_finite_angle():
    sec_ids = [0, 1]
    sec_members = {0: ["TF1", "TF2"], 1: ["TF3"]}
    sec_children = {0: ["G1", "G2", "G3"], 1: ["G4"]}

    ang_member, ang_child, sec_span = _layout_sectors(sec_ids, sec_members, sec_children)

    assert set(ang_member) == {"TF1", "TF2", "TF3"}
    assert set(ang_child) == {"G1", "G2", "G3", "G4"}
    for a in [*ang_member.values(), *ang_child.values()]:
        assert -10 < a < 10  # finite, sane radian range
    assert set(sec_span) == {0, 1}


@requires("decoupler")
def test_bezier_returns_a_path_from_p0_to_p1():
    path = _bezier((0.0, 0.0), (1.0, 1.0))

    assert path.vertices[0] == pytest.approx((0.0, 0.0))
    assert path.vertices[-1] == pytest.approx((1.0, 1.0))


# ---------------------------------------------------------------------------
# radial_grn (full figure)
# ---------------------------------------------------------------------------


@requires("decoupler")
def test_radial_grn_renders_a_figure_for_real_synthetic_data():
    tf = tf_activity_from_de(_tiny_de(), _tiny_net(), tmin=2)
    de = _tiny_de().set_index("gene")

    fig = radial_grn(tf, de, _tiny_net(), title="test", n_tf=3, per_tf=4, n_sectors=2)

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


@requires("decoupler")
def test_radial_grn_labels_case_and_control_generically_not_hardcoded():
    """A prior version of this figure (the reference lab script) hardcoded
    'up in LE' / 'up in Normal' -- CellQuorum must read case/control from
    config, not bake in one study's condition names."""
    tf = tf_activity_from_de(_tiny_de(), _tiny_net(), tmin=2)
    de = _tiny_de().set_index("gene")

    fig = radial_grn(
        tf, de, _tiny_net(), title="test", case="Tumor", control="Adjacent", n_tf=3, per_tf=4
    )

    all_text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    all_text += " " + " ".join(t.get_text() for t in fig.texts)
    assert "Tumor" in all_text
    assert "Adjacent" in all_text
    assert "LE" not in all_text.replace("legend", "")  # no leaked study-specific text


@requires("decoupler")
def test_radial_grn_returns_empty_fig_when_no_edges_survive():
    empty_tf = pd.DataFrame(columns=["score", "pvalue", "padj", "significant"])
    fig = radial_grn(
        empty_tf, pd.DataFrame(columns=["logFC", "FDR"]), _tiny_net(), title="t", n_tf=3, per_tf=4
    )

    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# curvature-on-radial-layout
# ---------------------------------------------------------------------------


def _tiny_regulon_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edge("TF1", "G1", weight=0.9)
    G.add_edge("TF1", "G2", weight=0.5)
    G.add_edge("TF2", "G2", weight=0.7)
    G.add_edge("TF2", "G3", weight=0.3)
    return G


def _tiny_curvature_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_df = pd.DataFrame(
        {
            "source": ["TF1", "TF1", "TF2", "TF2"],
            "target": ["G1", "G2", "G2", "G3"],
            "ricci_curvature": [-0.4, 0.1, 0.3, -0.1],
            "weight": [0.9, 0.5, 0.7, 0.3],
        }
    )
    node_df = pd.DataFrame(
        {
            "node": ["TF1", "TF2", "G1", "G2", "G3"],
            "ricci_curvature": [-0.15, 0.1, -0.4, 0.2, -0.1],
        }
    )
    return edge_df, node_df


@requires("decoupler")
def test_select_curvature_targets_orients_edges_tf_first_even_if_edge_df_swapped():
    """compute_ricci_curvature runs on an UNDIRECTED graph internally, so its
    edge_df's source/target order is not guaranteed to match the original
    TF->target direction -- select_curvature_targets must re-derive the real
    orientation from the directed regulon graph G, not trust edge_df as-is."""
    G = _tiny_regulon_graph()
    edge_df, node_df = _tiny_curvature_tables()
    # simulate the undirected-conversion swap for one row
    swapped = edge_df.copy()
    swapped.loc[0, ["source", "target"]] = ["G1", "TF1"]

    tfs, edges, _tf_curv, _tgt_curv = select_curvature_targets(
        swapped, node_df, G, n_tf=2, per_tf=5
    )

    assert all(t in {"TF1", "TF2"} for t, _g, _c, _w in edges)
    assert all(g not in {"TF1", "TF2"} for _t, g, _c, _w in edges)


@requires("decoupler")
def test_select_curvature_targets_caps_targets_per_tf():
    G = _tiny_regulon_graph()
    edge_df, node_df = _tiny_curvature_tables()

    _tfs, edges, _tf_curv, _tgt_curv = select_curvature_targets(
        edge_df, node_df, G, n_tf=2, per_tf=1
    )

    per_tf_counts: dict[str, int] = {}
    for t, _g, _c, _w in edges:
        per_tf_counts[t] = per_tf_counts.get(t, 0) + 1
    assert all(c <= 1 for c in per_tf_counts.values())


@requires("decoupler")
def test_radial_regulon_curvature_renders_a_figure():
    G = _tiny_regulon_graph()
    edge_df, node_df = _tiny_curvature_tables()

    fig = radial_regulon_curvature(edge_df, node_df, G, n_tf=2, per_tf=5, n_sectors=2)

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


@requires("decoupler")
def test_radial_regulon_curvature_empty_edges_returns_empty_fig():
    fig = radial_regulon_curvature(
        pd.DataFrame(columns=["source", "target", "ricci_curvature", "weight"]),
        pd.DataFrame(columns=["node", "ricci_curvature"]),
        nx.DiGraph(),
    )

    assert isinstance(fig, Figure)


@requires("decoupler")
def test_a_failure_scoring_tf_activity_is_not_disguised_as_no_tfs(monkeypatch):
    """An empty result means "nothing to score"; an error must stay an error.

    Otherwise a malformed DE table or a decoupler failure reads as "no TF activity scored"
    and the user chases the wrong cause.
    """
    import decoupler as dc
    import pytest

    def broken(*_args, **_kwargs):
        raise RuntimeError("ulm exploded")

    monkeypatch.setattr(dc.mt, "ulm", broken)
    with pytest.raises(RuntimeError, match="ulm exploded"):
        tf_activity_from_de(_tiny_de(), _tiny_net(), tmin=2)


@requires("decoupler")
def test_a_failure_ranking_the_de_table_is_not_disguised_as_no_tfs(monkeypatch):
    import pytest

    import cellquorum.stages.comparative.enrichment.ranking as ranking

    def broken(*_args, **_kwargs):
        raise ValueError("bad DE table")

    monkeypatch.setattr(ranking, "de_table_to_ranking", broken)
    with pytest.raises(ValueError, match="bad DE table"):
        tf_activity_from_de(_tiny_de(), _tiny_net(), tmin=2)


@requires("decoupler")
def test_scoring_tf_activity_never_changes_the_figure_theme():
    """decoupler rewrites matplotlib's legend settings the first time it is imported.

    The radial figure imports it after the theme is applied, so in a fresh process it would
    override the theme's legend size and spacing, while in a process that had already imported
    decoupler it would not: the same figure, two looks. First import can only be tested in a
    fresh interpreter.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib as mpl
        import pandas as pd
        from cellquorum.visualization.figstyle import apply_theme
        from cellquorum.stages.gene_regulation.grn.radial_figures import tf_activity_from_de

        apply_theme("compact")
        keys = ("legend.fontsize", "legend.borderpad", "legend.frameon")
        before = {k: mpl.rcParams[k] for k in keys}
        genes = [f"G{i}" for i in range(20)]
        de = pd.DataFrame({"gene": genes, "logFC": [2.0 - 0.1 * i for i in range(20)],
                           "PValue": [0.001] * 10 + [0.5] * 10, "FDR": [0.01] * 10 + [0.6] * 10})
        net = pd.DataFrame({"source": ["TF1"] * 6 + ["TF2"] * 6, "target": genes[:6] + genes[4:10],
                            "weight": 1.0})
        tf_activity_from_de(de, net, tmin=2)
        after = {k: mpl.rcParams[k] for k in before}
        assert after == before, (before, after)
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "OK" in result.stdout, result.stdout + result.stderr
