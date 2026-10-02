# tests/stages/comparative/differential_expression/test_volcano_subcluster_method.py
"""Tests for the per-subcluster volcano method (highlights identity markers)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.comparative.differential_expression.viz import (
    volcano_subcluster_method as mod,
)
from cellquorum.stages.comparative.differential_expression.viz.volcano_subcluster_method import (
    VolcanoBySubclusterMethod,
)


def _de_df():
    """2 clusters, 4 genes each -- 2 significant, 2 not, per cluster."""
    return pd.DataFrame(
        {
            "cell_type": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "gene": ["UP0A", "DN0A", "FLAT0", "FLAT0B", "UP1A", "DN1A", "FLAT1", "FLAT1B"],
            "logFC": [3.0, -3.0, 0.05, -0.02, 4.0, -4.0, 0.03, 0.01],
            "FDR": [0.001, 0.001, 0.9, 0.95, 0.001, 0.001, 0.9, 0.95],
        }
    )


def _adata(n=40, cluster_key="fibroblast_subcluster"):
    rng = np.random.default_rng(0)
    genes = ["UP0A", "DN0A", "FLAT0", "FLAT0B", "UP1A", "DN1A", "FLAT1", "FLAT1B"]
    X = rng.poisson(2.0, size=(n, len(genes))).astype("float32")
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {cluster_key: ["0" if i % 2 == 0 else "1" for i in range(n)]},
            index=[f"cell_{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=genes),
    )
    a.layers["cellquorum_normalized"] = X.copy()
    return a


def _ctx(
    tmp_path: Path, de_csv: pd.DataFrame | None, csv_name="de_pseudobulk_edger_by_subcluster.csv"
):
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    if de_csv is not None:
        de_csv.to_csv(results / csv_name, index=False)
    paths = SimpleNamespace(
        results=results, figures=tmp_path / "figs", scratch=tmp_path / "scratch"
    )
    return SimpleNamespace(paths=paths, config=None)


def test_skips_when_no_de_csv_present(tmp_path):
    ctx = _ctx(tmp_path, de_csv=None)
    res = VolcanoBySubclusterMethod()._run(_adata(), {}, ctx)
    assert isinstance(res, MethodSkip)
    assert "de_pseudobulk_edger_by_subcluster.csv" in res.reason


def test_skips_when_cell_type_col_missing_from_adata(tmp_path):
    ctx = _ctx(tmp_path, de_csv=_de_df())
    res = VolcanoBySubclusterMethod()._run(_adata(), {"cell_type_col": "not_a_real_column"}, ctx)
    assert isinstance(res, MethodSkip)
    assert "not_a_real_column" in res.reason


def test_writes_one_figure_per_subcluster(tmp_path):
    ctx = _ctx(tmp_path, de_csv=_de_df())
    res = VolcanoBySubclusterMethod()._run(
        _adata(),
        {"cell_type_col": "fibroblast_subcluster", "figure_formats": ["png"]},
        ctx,
    )
    assert not isinstance(res, MethodSkip)
    assert len(res.artifacts) == 2  # one per cluster (0 and 1), png only
    for artifact in res.artifacts:
        assert artifact.path.exists()


def test_cluster_labels_rename_output_filenames(tmp_path):
    ctx = _ctx(tmp_path, de_csv=_de_df())
    res = VolcanoBySubclusterMethod()._run(
        _adata(),
        {
            "cell_type_col": "fibroblast_subcluster",
            "figure_formats": ["png"],
            "cluster_labels": {"0": "Papillary", "1": "Reticular"},
        },
        ctx,
    )
    names = [p.path.name for p in res.artifacts]
    assert any("Papillary" in n for n in names)
    assert any("Reticular" in n for n in names)
    assert not any("volcano_subcluster_0" in n for n in names)


def test_passes_each_clusters_own_markers_as_highlight_genes(tmp_path, monkeypatch):
    """The 0-cluster volcano highlights 0's markers, not 1's, and vice versa."""
    calls = []
    real_volcano = mod.plots.volcano

    def _spy_volcano(df, **kwargs):
        calls.append((df["gene"].tolist(), kwargs.get("highlight_genes")))
        return real_volcano(df, **kwargs)

    monkeypatch.setattr(mod.plots, "volcano", _spy_volcano)
    monkeypatch.setattr(
        mod,
        "select_marker_genes",
        lambda adata, *, cluster_key, layer, n_top: {"0": ["FLAT0"], "1": ["FLAT1"]},
    )

    ctx = _ctx(tmp_path, de_csv=_de_df())
    VolcanoBySubclusterMethod()._run(
        _adata(),
        {"cell_type_col": "fibroblast_subcluster", "figure_formats": ["png"]},
        ctx,
    )
    assert len(calls) == 2
    for genes, hl in calls:
        if "FLAT0" in genes:
            assert hl == {"FLAT0"}
        else:
            assert hl == {"FLAT1"}


def test_skip_reason_when_no_groups_survive_intersection(tmp_path):
    df = _de_df()
    df["cell_type"] = "9"  # a cluster id absent from the adata's obs column
    ctx = _ctx(tmp_path, de_csv=df)
    res = VolcanoBySubclusterMethod()._run(
        _adata(), {"cell_type_col": "fibroblast_subcluster"}, ctx
    )
    assert isinstance(res, MethodSkip)
