# tests/test_trajectory_stage.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.trajectory.stage import TrajectoryStage


class _Paths:
    def __init__(self, tmp):
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Vel:
    # mimics VelocityConfig attribute access for the flatten step
    grouping_col = "cell_type"
    sample_col = "sample_id"
    loom_path_col = "loom_path"
    groups = None
    use_rep = None
    use_rep_fallback = ["X_pca"]
    mode = "dynamical"
    min_shared_counts = 0
    n_top_genes = 3
    n_pcs = 2
    n_neighbors = 2
    min_cells = 1
    n_jobs = 1
    seed = 0
    enabled = True

    def model_dump(self):
        return {
            k: getattr(self, k) for k in dir(self) if not k.startswith("_") and k != "model_dump"
        }


class _TrajCfg:
    enabled = True
    methods: list = []
    velocity = _Vel()


class _Cfg:
    def __init__(self):
        self.trajectory = _TrajCfg()
        self.cohort = None


class _Ctx:
    def __init__(self, tmp, adata, manifest):
        self.config = _Cfg()
        self.paths = _Paths(tmp)
        self._adata = adata
        self._manifest = manifest

    def require_adata(self):
        return self._adata

    def require_manifest(self):
        if self._manifest is None:
            raise RuntimeError("no manifest")
        return self._manifest


def test_velocity_registered():
    assert METHOD_REGISTRY.has("trajectory", "velocity")


def test_stage_injects_default_velocity_method():
    stage = TrajectoryStage()
    augmented = stage._augment_config(_Ctx.__new__(_Ctx), {"enabled": True})
    assert [m["method"] for m in augmented["methods"]] == ["velocity"]


def test_stage_skips_when_no_looms(tmp_path):
    # No manifest → the velocity method MethodSkips; stage records it as a warning.
    genes = ["GENE_A", "GENE_B", "GENE_C"]
    a = ad.AnnData(
        X=np.ones((2, 3), dtype="float32"),
        obs=pd.DataFrame(
            {"sample_id": ["s1", "s1"], "cell_type": ["T", "T"]}, index=["s1_AAAA-1", "s1_CCCC-1"]
        ),
    )
    a.var_names = genes
    result = TrajectoryStage().run(_Ctx(tmp_path, a, manifest=None))
    assert result.status == "success"
    assert any("skipped" in w for w in result.warnings)
