# tests/test_ccc_viz_discovery.py

import pandas as pd


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
    from cellquorum.ccc_viz.discovery import load_canonical_lr_sources

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
    from cellquorum.ccc_viz.discovery import load_canonical_lr_sources

    (tmp_path / "cell_cell_communication").mkdir()
    _canon().to_csv(tmp_path / "cell_cell_communication" / "liana_ranks.csv", index=False)
    _canon().to_csv(tmp_path / "mnn_canonical_lr.csv", index=False)
    _canon().to_csv(tmp_path / "nichenet_canonical_lr.csv", index=False)
    out = load_canonical_lr_sources(tmp_path, None)
    assert [lab for lab, _ in out] == ["liana", "multinichenet", "nichenet"]


def test_load_canonical_empty_when_none(tmp_path):
    from cellquorum.ccc_viz.discovery import load_canonical_lr_sources

    assert load_canonical_lr_sources(tmp_path, None) == []


def test_load_canonical_skips_unreadable(tmp_path):
    from cellquorum.ccc_viz.discovery import load_canonical_lr_sources

    (tmp_path / "mnn_canonical_lr.csv").write_bytes(b"\xff\xfe\x00\x01 not a csv \x00\x00")
    # unreadable/misparsed -> omitted, never raises
    out = load_canonical_lr_sources(tmp_path, None)
    assert "multinichenet" not in [lab for lab, _ in out]


def test_load_topology_and_curvature(tmp_path):
    from cellquorum.ccc_viz.discovery import load_curvature, load_topology

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
