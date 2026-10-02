# tests/stages/state_scoring/test_state_scoring_viz_methods.py
"""Tests for StateScoreViolinMethod (real matplotlib rendering, no fakes needed)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.state_scoring.viz._viz_methods import StateScoreViolinMethod


def _adata(n=60):
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((n, 8)).astype("float32"))
    a.obs["fibroblast_subcluster"] = ["0"] * 30 + ["1"] * 30
    a.obs["fibroblast_subcluster"] = a.obs["fibroblast_subcluster"].astype("category")
    a.obs["condition"] = (["Normal"] * 15 + ["Lymphedema"] * 15) * 2
    a.obsm["X_state_aucell"] = rng.random((n, 2)).astype(float)
    a.uns["state_aucell"] = {
        "programs": ["proliferating", "myofibroblast"],
        "genes": {"proliferating": ["G1"], "myofibroblast": ["G2"]},
    }
    return a


def _ctx(tmp_path: Path, *, design=None):
    paths = SimpleNamespace(
        root=tmp_path,
        results=tmp_path / "results",
        figures=tmp_path / "figs",
        scratch=tmp_path / "scratch",
    )
    config = SimpleNamespace(design=design)
    return SimpleNamespace(paths=paths, config=config)


def _design():
    return SimpleNamespace(condition_col="condition", case="Lymphedema", control="Normal")


def test_skips_when_score_obsm_absent(tmp_path):
    a = _adata()
    del a.obsm["X_state_aucell"]
    ctx = _ctx(tmp_path, design=_design())
    res = StateScoreViolinMethod()._run(a, {}, ctx)
    assert isinstance(res, MethodSkip)


def test_skips_when_condition_design_unresolvable(tmp_path):
    ctx = _ctx(tmp_path, design=None)
    res = StateScoreViolinMethod()._run(_adata(), {"cluster_key": "fibroblast_subcluster"}, ctx)
    assert isinstance(res, MethodSkip)
    assert "condition" in res.reason.lower()


def test_skips_when_requested_program_not_scored(tmp_path):
    ctx = _ctx(tmp_path, design=_design())
    res = StateScoreViolinMethod()._run(_adata(), {"programs": ["not_a_real_program"]}, ctx)
    assert isinstance(res, MethodSkip)


def test_success_renders_one_figure_per_program(tmp_path):
    ctx = _ctx(tmp_path, design=_design())
    res = StateScoreViolinMethod()._run(
        _adata(),
        {"cluster_key": "fibroblast_subcluster", "figure_formats": ["png"]},
        ctx,
    )
    assert not isinstance(res, MethodSkip)
    assert len(res.artifacts) == 2  # proliferating + myofibroblast, png only
    for artifact in res.artifacts:
        assert artifact.path.exists()
        assert artifact.path.parent.name == "state_scoring"


def test_cluster_labels_rename_output_without_changing_filenames(tmp_path):
    ctx = _ctx(tmp_path, design=_design())
    res = StateScoreViolinMethod()._run(
        _adata(),
        {
            "cluster_key": "fibroblast_subcluster",
            "figure_formats": ["png"],
            "cluster_labels": {"0": "Papillary", "1": "Reticular"},
            "programs": ["proliferating"],
        },
        ctx,
    )
    assert len(res.artifacts) == 1
    assert "proliferating" in res.artifacts[0].path.name
