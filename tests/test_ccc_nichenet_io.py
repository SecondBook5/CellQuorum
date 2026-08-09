from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from cellquorum.cell_cell_communication._nichenet_io import (
    CANONICAL_COLUMNS,
    de_to_geneset,
    export_sce_inputs,
    ligand_activity_to_canonical,
    mnn_prioritization_to_canonical,
)


def _toy_adata():
    X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]))
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "B", "A"],
            "sample_id": ["s1", "s1", "s2"],
            "condition": ["ctrl", "ctrl", "case"],
        },
        index=["c1", "c2", "c3"],
    )
    var = pd.DataFrame(index=["Gene1", "Gene2", "Gene3"])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_export_sce_inputs_writes_all_files(tmp_path):
    adata = _toy_adata()
    paths = export_sce_inputs(adata, ["cell_type", "sample_id", "condition"], tmp_path)
    assert set(paths) == {"counts", "genes", "barcodes", "obs"}
    for p in paths.values():
        assert p.is_file()
    # counts.mtx is genes x cells (3 genes x 3 cells)
    mat = scipy.io.mmread(paths["counts"])
    assert mat.shape == (3, 3)
    genes = pd.read_csv(paths["genes"])
    assert list(genes["gene"]) == ["Gene1", "Gene2", "Gene3"]
    obs = pd.read_csv(paths["obs"])
    assert list(obs.columns) == ["barcode", "cell_type", "sample_id", "condition"]
    assert list(obs["barcode"]) == ["c1", "c2", "c3"]


def test_de_to_geneset_filters_and_caps():
    de = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G3", "G4", "G5"],
            "logFC": [3.0, -2.0, 0.5, 5.0, 1.0],
            "logCPM": [1, 1, 1, 1, 1],
            "F": [1, 1, 1, 1, 1],
            "PValue": [0.001, 0.001, 0.2, 0.001, 0.001],
            "FDR": [0.01, 0.02, 0.30, 0.005, 0.04],
        }
    )
    geneset, background = de_to_geneset(de, fdr=0.05, top_n=2)
    # G3 fails FDR; among {G1,G2,G4,G5} top-2 by |logFC| = G4(5), G1(3)
    assert geneset == ["G1", "G4"]  # sorted alphabetically for determinism
    assert background == ["G1", "G2", "G3", "G4", "G5"]


def test_de_to_geneset_empty_when_none_significant():
    de = pd.DataFrame(
        {
            "gene": ["G1", "G2"],
            "logFC": [1.0, 2.0],
            "logCPM": [1, 1],
            "F": [1, 1],
            "PValue": [0.5, 0.5],
            "FDR": [0.5, 0.6],
        }
    )
    geneset, background = de_to_geneset(de, fdr=0.05, top_n=10)
    assert geneset == []
    assert background == ["G1", "G2"]


def test_mnn_prioritization_to_canonical_maps_and_drops():
    native = pd.DataFrame(
        {
            "sender": ["A", "B"],
            "receiver": ["B", "A"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "prioritization_score": [0.9, 0.4],
            "group": ["case", "case"],
        }
    )
    out = mnn_prioritization_to_canonical(native)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert list(out["source"]) == ["A", "B"]
    assert list(out["target"]) == ["B", "A"]
    assert list(out["weight"]) == [0.9, 0.4]
    assert list(out["condition"]) == ["case", "case"]
    assert (out["weight"] >= 0).all()


def test_mnn_drops_rows_missing_required():
    native = pd.DataFrame(
        {
            "sender": ["A"],
            "receiver": ["B"],
            "ligand": [None],
            "receptor": ["R1"],
            "prioritization_score": [0.9],
            "group": ["case"],
        }
    )
    out = mnn_prioritization_to_canonical(native)
    assert len(out) == 0


def test_ligand_activity_to_canonical_clamps_negative():
    native = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "aupr_corrected": [0.3, -0.05],
        }
    )
    out = ligand_activity_to_canonical(native, sender="LEC", receiver="Fib", condition="LE")
    assert list(out.columns) == CANONICAL_COLUMNS
    assert list(out["source"]) == ["LEC", "LEC"]
    assert list(out["target"]) == ["Fib", "Fib"]
    assert list(out["weight"]) == [0.3, 0.0]  # negative clamped to 0
    assert list(out["condition"]) == ["LE", "LE"]


def test_canonical_output_feeds_ccc_network():
    """Cross-spec contract: a canonical frame from spec #2 builds a spec #3 network."""
    from cellquorum.ccc_network._networks import build_cci_network
    from cellquorum.cell_cell_communication._nichenet_io import mnn_prioritization_to_canonical

    native = pd.DataFrame(
        {
            "sender": ["A", "A", "B"],
            "receiver": ["B", "C", "C"],
            "ligand": ["L1", "L2", "L3"],
            "receptor": ["R1", "R2", "R3"],
            "prioritization_score": [0.9, 0.5, 0.7],
            "group": ["case"] * 3,
        }
    )
    canonical = mnn_prioritization_to_canonical(native)
    G = build_cci_network(canonical)
    assert G.number_of_nodes() >= 3
    assert G.number_of_edges() >= 1
