# tests/test_ccc_viz_discovery.py

import pandas as pd


def test_relabel_clusters_overrides_display_only():
    """Real ask: every CCC figure (dotplot, chord, sankey, network, topology)
    showed bare numeric subcluster codes ("0","1",...) with no way to tell
    what they mean. cluster_labels is a display-only {raw code -> name}
    override, same pattern as trajectory_viz's fate_labels: a code with no
    entry stays unchanged, and columns not named in `cols` are untouched.
    """
    from cellquorum.stages.cell_cell_communication.viz._io import relabel_clusters

    df = pd.DataFrame({"source": ["0", "1"], "target": ["1", "6"], "ligand": ["L1", "L2"]})
    out = relabel_clusters(df, ["source", "target"], {"0": "Perivascular", "1": "Mechanosensory"})
    assert list(out["source"]) == ["Perivascular", "Mechanosensory"]
    assert list(out["target"]) == ["Mechanosensory", "6"]  # "6" has no entry, stays bare
    assert list(out["ligand"]) == ["L1", "L2"]  # untouched column


def test_cluster_labels_is_in_the_stage_whitelist():
    """A config field that exists on CccVizConfig but is missing from
    _VIZ_CONFIG_KEYS is silently dropped before any method ever sees it --
    the exact bug already found live in trajectory_viz (fate_labels existed
    on the model but wasn't forwarded by _augment_config).
    """
    from cellquorum.stages.cell_cell_communication.viz.stage import _VIZ_CONFIG_KEYS

    assert "cluster_labels" in _VIZ_CONFIG_KEYS


def test_relabel_clusters_no_labels_is_a_noop():
    from cellquorum.stages.cell_cell_communication.viz._io import relabel_clusters

    df = pd.DataFrame({"source": ["0"], "target": ["1"]})
    out = relabel_clusters(df, ["source", "target"], None)
    assert list(out["source"]) == ["0"]
    assert out is df  # no-op returns the same object, not a needless copy


def _canon(sample="s1"):
    return pd.DataFrame(
        {
            "source": ["A"],
            "target": ["B"],
            "ligand": ["L1"],
            "receptor": ["R1"],
            "weight": [0.5],
            "sample": [sample],
        }
    )


def test_load_canonical_prefers_uns_liana(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import load_canonical_lr_sources

    liana_res = pd.DataFrame(
        {
            "sample": ["s1"],
            "source": ["A"],
            "target": ["B"],
            "ligand_complex": ["L1"],
            "receptor_complex": ["R1"],
            "magnitude_rank": [0.1],
        }
    )
    out = load_canonical_lr_sources(tmp_path, {"liana_res": liana_res})
    labels = [lab for lab, _ in out]
    assert labels == ["liana"]
    assert list(out[0][1].columns[:6]) == [
        "source",
        "target",
        "ligand",
        "receptor",
        "weight",
        "sample",
    ]


def test_load_canonical_reads_csvs_in_order(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import load_canonical_lr_sources

    (tmp_path / "cell_cell_communication").mkdir()
    _canon().to_csv(tmp_path / "cell_cell_communication" / "liana_ranks.csv", index=False)
    _canon().to_csv(tmp_path / "mnn_canonical_lr.csv", index=False)
    _canon().to_csv(tmp_path / "nichenet_canonical_lr.csv", index=False)
    out = load_canonical_lr_sources(tmp_path, None)
    assert [lab for lab, _ in out] == ["liana", "multinichenet", "nichenet"]


def test_load_canonical_empty_when_none(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import load_canonical_lr_sources

    assert load_canonical_lr_sources(tmp_path, None) == []


def test_load_canonical_skips_unreadable(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import load_canonical_lr_sources

    (tmp_path / "mnn_canonical_lr.csv").write_bytes(b"\xff\xfe\x00\x01 not a csv \x00\x00")
    # unreadable/misparsed -> omitted, never raises
    out = load_canonical_lr_sources(tmp_path, None)
    assert "multinichenet" not in [lab for lab, _ in out]


def test_load_nichenet_returns_whichever_tables_are_present(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import load_nichenet

    pd.DataFrame({"test_ligand": ["TGFB2"], "aupr_corrected": [0.02]}).to_csv(
        tmp_path / "nichenet_activities.csv", index=False
    )
    # weights/sender_expression deliberately absent -- a partial result set
    # (e.g. weights_csv never wrote) should still surface what did land,
    # not be treated as all-or-nothing like tensor_factors.
    out = load_nichenet(tmp_path)
    assert set(out) == {"activities"}


def test_load_nichenet_reads_all_four_tables(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import load_nichenet

    pd.DataFrame({"test_ligand": ["TGFB2"], "aupr_corrected": [0.02]}).to_csv(
        tmp_path / "nichenet_activities.csv", index=False
    )
    pd.DataFrame({"ligand": ["TGFB2"], "target": ["COL1A1"], "weight": [0.24]}).to_csv(
        tmp_path / "nichenet_ligand_target_weights.csv", index=False
    )
    pd.DataFrame(
        {
            "sender": ["1"],
            "ligand": ["TGFB2"],
            "fraction_expressing": [0.17],
            "mean_expression": [0.32],
        }
    ).to_csv(tmp_path / "nichenet_sender_expression.csv", index=False)
    pd.DataFrame({"ligand": ["TGFB2"], "receptor": ["TGFBR1"], "weight": [0.9]}).to_csv(
        tmp_path / "nichenet_ligand_receptor_weights.csv", index=False
    )
    out = load_nichenet(tmp_path)
    assert set(out) == {"activities", "weights", "sender_expression", "receptor_weights"}


def test_load_nichenet_empty_dict_when_none_present(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import load_nichenet

    assert load_nichenet(tmp_path) == {}


def test_load_mnn_prioritization_reads_the_native_table(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import load_mnn_prioritization

    pd.DataFrame(
        {
            "group": ["Lymphedema"],
            "sender": ["0"],
            "receiver": ["3"],
            "ligand": ["TGFB1"],
            "receptor": ["TGFBR1"],
            "prioritization_score": [0.95],
        }
    ).to_csv(tmp_path / "mnn_prioritization.csv", index=False)
    out = load_mnn_prioritization(tmp_path)
    assert out is not None
    assert list(out["ligand"]) == ["TGFB1"]


def test_load_mnn_prioritization_none_when_absent(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import load_mnn_prioritization

    assert load_mnn_prioritization(tmp_path) is None


def test_load_topology_and_curvature(tmp_path):
    from cellquorum.stages.cell_cell_communication.viz._io import load_curvature, load_topology

    d = tmp_path / "ccc_network"
    d.mkdir()
    pd.DataFrame(
        {
            "node": ["A"],
            "Listener": [1.0],
            "Influencer": [0.0],
            "Mediator": [0.0],
            "Pagerank": [0.5],
        }
    ).to_csv(d / "topology_cci.csv", index=False)
    pd.DataFrame(
        {"source": ["A"], "target": ["B"], "ricci_curvature": [-0.2], "weight": [1.0]}
    ).to_csv(d / "curvature_cci_edges.csv", index=False)
    topo = load_topology(tmp_path)
    curv = load_curvature(tmp_path)
    assert "cci" in topo
    assert "cci_edges" in curv
    assert load_topology(tmp_path / "nope") == {}
