"""A frozen display layout stands in for a computed UMAP, aligned by cell ID."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.integration.embeddings import compute
from cellquorum.stages.integration.embeddings.methods import UmapMethod


def _adata(n: int = 6) -> ad.AnnData:
    a = ad.AnnData(X=np.zeros((n, 3), dtype="float32"))
    a.obs_names = [f"cell{i}" for i in range(n)]
    return a


def _layout(tmp_path: Path, cells: list[str], *, name: str = "layout.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(
        {"cell": cells, "umap1": [float(i) for i in range(len(cells))], "umap2": 10.0}
    ).to_csv(path, index=False)
    return path


def test_coordinates_are_matched_to_cells_by_id_not_by_row_order(tmp_path):
    a = _adata()
    shuffled = ["cell3", "cell0", "cell5", "cell1", "cell4", "cell2"]
    path = _layout(tmp_path, shuffled)

    compute.load_layout(a, path)

    # cell3 is row 0 of the file, so its umap1 is 0.0; cell0 is row 1, so 1.0.
    assert a.obsm["X_umap"][3].tolist() == [0.0, 10.0]
    assert a.obsm["X_umap"][0].tolist() == [1.0, 10.0]
    assert a.obsm["X_umap"].shape == (6, 2)


def test_a_gzipped_layout_is_read(tmp_path):
    a = _adata()
    path = tmp_path / "layout.csv.gz"
    pd.DataFrame({"cell": a.obs_names, "umap1": range(6), "umap2": range(6)}).to_csv(
        path, index=False
    )

    compute.load_layout(a, path)

    assert a.obsm["X_umap"].shape == (6, 2)


def test_a_layout_missing_some_cells_is_an_error_that_counts_them(tmp_path):
    a = _adata()
    path = _layout(tmp_path, ["cell0", "cell1", "cell2", "cell3"])

    with pytest.raises(compute.LayoutMismatch, match="2 of 6"):
        compute.load_layout(a, path)
    assert "X_umap" not in a.obsm


def test_extra_cells_in_the_layout_are_ignored(tmp_path):
    a = _adata(4)
    path = _layout(tmp_path, [f"cell{i}" for i in range(9)])

    compute.load_layout(a, path)

    assert a.obsm["X_umap"].shape == (4, 2)


def test_a_layout_without_the_expected_columns_names_what_is_missing(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"cell": ["cell0"], "x": [1.0]}).to_csv(path, index=False)

    with pytest.raises(compute.LayoutMismatch, match="umap1"):
        compute.load_layout(_adata(), path)


def test_a_layout_with_duplicate_cells_is_refused(tmp_path):
    path = _layout(tmp_path, ["cell0", "cell0", "cell1", "cell2", "cell3", "cell4", "cell5"])

    with pytest.raises(compute.LayoutMismatch, match="duplicate"):
        compute.load_layout(_adata(), path)


def test_the_umap_method_uses_the_layout_and_needs_no_neighbors_graph(tmp_path):
    a = _adata()
    assert "neighbors" not in a.uns
    path = _layout(tmp_path, list(a.obs_names))

    res = UmapMethod()._run(a, {"layout_file": str(path)}, SimpleNamespace(random_seed=1))

    assert not isinstance(res, MethodSkip)
    assert a.obsm["X_umap"][2].tolist() == [2.0, 10.0]
    assert res.metrics["compute"] == "layout_file"


def test_a_bad_layout_skips_and_never_falls_back_to_computing_a_different_umap(tmp_path):
    a = _adata()
    path = _layout(tmp_path, ["cell0", "cell1"])

    res = UmapMethod()._run(a, {"layout_file": str(path)}, SimpleNamespace(random_seed=1))

    assert isinstance(res, MethodSkip)
    assert "4 of 6" in res.reason
    assert "X_umap" not in a.obsm


def test_a_missing_layout_file_skips_with_its_path(tmp_path):
    res = UmapMethod()._run(
        _adata(), {"layout_file": str(tmp_path / "nope.csv")}, SimpleNamespace(random_seed=1)
    )

    assert isinstance(res, MethodSkip)
    assert "nope.csv" in res.reason


def test_the_config_field_defaults_to_computing():
    from cellquorum.stages.integration.embeddings.config import EmbeddingsConfig

    assert EmbeddingsConfig().layout_file is None


def test_read_layout_returns_coordinates_without_touching_the_object(tmp_path):
    a = _adata()
    path = _layout(tmp_path, ["cell5", "cell4", "cell3", "cell2", "cell1", "cell0"])

    xy = compute.read_layout(a.obs_names, path)

    assert xy.shape == (6, 2)
    assert xy[5].tolist() == [0.0, 10.0]  # cell5 is row 0 of the file
    assert "X_umap" not in a.obsm
