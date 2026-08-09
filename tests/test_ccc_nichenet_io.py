from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from cellquorum.cell_cell_communication._nichenet_io import de_to_geneset, export_sce_inputs


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
