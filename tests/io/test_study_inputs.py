from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite
from typer.testing import CliRunner

from cellquorum.cli.app import app
from cellquorum.config.loader import load_config
from cellquorum.io.study_inputs import import_libraries


def write_library(path: Path, genes=("A", "B"), counts=(1, 2)):
    directory = path / "outs" / "filtered_feature_bc_matrix"
    directory.mkdir(parents=True)
    (directory / "barcodes.tsv").write_text("BC-1\n")
    (directory / "features.tsv").write_text(
        "".join(f"id-{g}\t{g}\tGene Expression\n" for g in genes)
    )
    mmwrite(directory / "matrix.mtx", sparse.coo_matrix(np.array([counts]).T))


def test_10x_sheet_preserves_library_identity_feature_order_and_optional_files(tmp_path):
    write_library(tmp_path / "one")
    write_library(tmp_path / "two", ("B", "A"), (3, 4))
    (tmp_path / "one/outs/raw_feature_bc_matrix").mkdir()
    (tmp_path / "one/outs/possorted_genome_bam.bam").touch()
    (tmp_path / "one/cells.csv").write_text("barcode,cell_type\nBC-1,B\n")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("sample_id,path,donor,metadata_path\nL1,one,D1,one/cells.csv\nL2,two,D1,\n")
    manifest = import_libraries(sheet, tmp_path / "result.h5ad")
    obj = ad.read_h5ad(tmp_path / "result.h5ad")
    np.testing.assert_array_equal(obj.X.toarray(), [[1, 2], [4, 3]])
    assert list(obj.obs_names) == ["L1:BC-1", "L2:BC-1"]
    assert list(obj.obs.barcode) == ["BC-1", "BC-1"] and obj.obs.cell_type.iloc[0] == "B"
    assert list(obj.var.gene_ids) == ["id-A", "id-B"]
    assert (
        Path(manifest.raw_matrix_path.iloc[0]).is_dir()
        and Path(manifest.bam_path.iloc[0]).is_file()
    )
    # A guided run consumes the same portable sheet without scripting.
    target = tmp_path / "study"
    cli = CliRunner().invoke(
        app,
        ["init", str(target), "--input", str(sheet)],
        input="explore\nX\nhuman\ncells\nsample_id\n1\n",
    )
    assert cli.exit_code == 0, cli.output
    cfg = load_config(target / "config.yaml")
    assert cfg.paths.manifest == target / "samples.csv" and cfg.input.h5ad == target / "input.h5ad"
    paths = pd.read_csv(cfg.paths.manifest)
    assert not Path(paths.raw_matrix_path.iloc[0]).is_absolute()
    assert (target / paths.raw_matrix_path.iloc[0]).is_dir()


def test_h5ad_layer_provenance_and_mismatched_refs(tmp_path):
    data = ad.AnnData(
        sparse.csr_matrix([[0.4, 0.6]]),
        obs=pd.DataFrame(index=["A"]),
        var=pd.DataFrame(index=["X", "Y"]),
    )
    data.layers["counts"] = sparse.csr_matrix([[4, 6]])
    data.layers["spliced"] = data.layers["counts"].copy()
    data.write_h5ad(tmp_path / "a.h5ad")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("sample_id,path,counts_layer\ns,a.h5ad,counts\n")
    import_libraries(sheet, tmp_path / "result.h5ad")
    obj = ad.read_h5ad(tmp_path / "result.h5ad")
    assert obj.uns["cellquorum_count_sources"]["s"] == "counts"
    np.testing.assert_allclose(obj.layers["cellquorum_input_X"].toarray(), [[0.4, 0.6]])
    np.testing.assert_array_equal(obj.layers["spliced"].toarray(), obj.X.toarray())
    write_library(tmp_path / "one")
    write_library(tmp_path / "two", ("C", "B"))
    sheet.write_text("sample_id,path\nx,one\ny,two\n")
    with pytest.raises(ValueError, match="different feature reference"):
        import_libraries(sheet, tmp_path / "invalid.h5ad")
    assert not (tmp_path / "invalid.h5ad").exists()
