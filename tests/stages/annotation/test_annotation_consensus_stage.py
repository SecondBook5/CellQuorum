"""Tests for AnnotationConsensusStage."""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.stages.annotation.consensus.stage import AnnotationConsensusStage


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.reports = tmp / "reports"


class _Ctx:
    def __init__(self, adata, config, paths):
        self._adata = adata
        self.config = config
        self.paths = paths

    def require_adata(self):
        return self._adata


def _labeled_adata():
    n = 6
    a = ad.AnnData(X=np.ones((n, 2), dtype=np.float32))
    a.obs_names = [f"c{i}" for i in range(n)]
    # 3 methods; row 0 all-agree, row 1 2/3, row 2 3-way split, row 3 one method only.
    a.obs["m1"] = ["T cell", "T/NK", "T/NK", None, "Mast", "Mast"]
    a.obs["m2"] = ["T/NK", "T/NK", "Fibroblasts", None, "Mast", "Mast"]
    a.obs["ref_state"] = ["T/NK", "Fibroblasts", "DC", "B cells", "Mast", "Mast"]
    return a


def _config(tmp, **overrides):
    base = dict(
        method_label_keys=["m1", "m2", "ref_state"],
        backbone_aliases={"T cell": "T/NK"},
        granular_source_key="ref_state",
    )
    base.update(overrides)
    return {"annotation_consensus": base}


def test_stage_writes_consensus_columns(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "reports").mkdir()
    a = _labeled_adata()
    ctx = _Ctx(a, _config(tmp_path), _Paths(tmp_path))
    result = AnnotationConsensusStage().run(ctx)
    obs = result.adata.obs
    assert "cell_type" in obs.columns
    assert "annotation_confidence" in obs.columns
    assert "needs_review" in obs.columns
    # Row 0: T cell/T/NK/T/NK -> all become T/NK after alias -> high.
    assert obs["cell_type"].iloc[0] == "T/NK"
    assert obs["annotation_confidence"].iloc[0] == "high"
    assert bool(obs["needs_review"].iloc[0]) is False
    # Row 2: T/NK, Fibroblasts, DC -> low, needs review.
    assert obs["annotation_confidence"].iloc[2] == "low"
    assert bool(obs["needs_review"].iloc[2]) is True


def test_stage_skips_when_no_label_columns(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "reports").mkdir()
    a = _labeled_adata()
    ctx = _Ctx(a, _config(tmp_path, method_label_keys=["absent1", "absent2"]), _Paths(tmp_path))
    result = AnnotationConsensusStage().run(ctx)
    assert result.metrics.get("skipped") is True
    assert "cell_type" not in result.adata.obs.columns


def test_stage_writes_granular_for_high_confidence(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "reports").mkdir()
    a = _labeled_adata()
    ctx = _Ctx(a, _config(tmp_path), _Paths(tmp_path))
    result = AnnotationConsensusStage().run(ctx)
    # Rows 4/5 are unanimous Mast -> granular copied from ref_state.
    assert result.adata.obs["cell_type_granular"].iloc[4] == "Mast"
