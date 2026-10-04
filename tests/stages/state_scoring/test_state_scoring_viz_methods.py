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


def test_violin_stars_are_tested_on_the_design_donor_column(tmp_path, monkeypatch):
    """The significance marks must be computed over donors named by ``design``: the stage used to
    hand the plot only cells, so its marks were a cell-level test whatever the design said."""
    from cellquorum.stages.state_scoring.viz import _viz_methods

    seen = {}
    real = _viz_methods.state_score_violin

    def spy(df, **kwargs):
        seen["columns"] = set(df.columns)
        seen.update(donor_col=kwargs.get("donor_col"), paired=kwargs.get("paired"))
        return real(df, **kwargs)

    monkeypatch.setattr(_viz_methods, "state_score_violin", spy)
    a = _adata()
    a.obs["donor_id"] = ["P1", "P2", "P3"] * 20
    design = SimpleNamespace(
        condition_col="condition",
        case="Lymphedema",
        control="Normal",
        donor_col="donor_id",
        paired=True,
    )
    res = StateScoreViolinMethod()._run(
        a,
        {
            "cluster_key": "fibroblast_subcluster",
            "figure_formats": ["png"],
            "programs": ["proliferating"],
        },
        _ctx(tmp_path, design=design),
    )
    assert not isinstance(res, MethodSkip)
    assert seen["donor_col"] == "donor"
    assert "donor" in seen["columns"]
    assert seen["paired"] is True
