"""End-to-end MetacellsStage.run() with a fake backend: confirms figures are
wired in and the full artifact/metrics contract holds through the stage
dispatch layer, not just the method in isolation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.stages.metacells.stage import MetacellsStage


def _adata(n: int = 6000, g: int = 20) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n)])
    obs["ref_cell_type_granular_coarse"] = [
        "Fibroblast" if i % 2 else "Endothelial" for i in range(n)
    ]
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(g)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["counts"] = X.copy()
    # The stage's input_contract expects "counts" to carry a provenance tag
    # (real pipeline runs get this from an earlier stage); the brief's fixture
    # omitted it, which trips MethodDispatchStage's contract check that the
    # method-level unit tests (which call `_run` directly) never exercise.
    set_layer_tag(a, "counts", kind="counts")
    a.obsm["X_pca_harmony"] = rng.normal(size=(n, 10)).astype("float32")
    return a


def _write_canned_outputs(out_dir: Path, adata: ad.AnnData) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = adata.n_obs
    metacell_ids = ["MC1"] * (n // 2) + ["MC2"] * (n - n // 2 - 4) + ["Outliers"] * 4
    groups = adata.obs["ref_cell_type_granular_coarse"].to_numpy()
    assignments = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "metacell_id": metacell_ids,
            "group": groups,
            "n_umis": np.asarray(adata.layers["counts"].sum(axis=1)).ravel(),
        }
    )
    assignments.to_csv(out_dir / "metacell_assignments.csv", index=False)

    mc_obs = pd.DataFrame(index=["MC1", "MC2"])
    mc_var = pd.DataFrame(index=adata.var_names)
    mc_X = np.random.default_rng(1).normal(size=(2, adata.n_vars)).astype("float32")
    mc_adata = ad.AnnData(X=mc_X, obs=mc_obs, var=mc_var)
    mc_adata.write_h5ad(out_dir / "metacell_adata.h5ad")


class _FakeBackend:
    def __init__(self, adata: ad.AnnData) -> None:
        self._adata = adata

    def _py_module_available(self, _m):  # noqa: ANN001
        return True

    def run_script(self, _script, args, *, timeout=None):  # noqa: ANN001, ANN003
        out_dir = Path(args[1])
        _write_canned_outputs(out_dir, self._adata)
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")


def _ctx(tmp_path: Path, adata: ad.AnnData):
    paths = SimpleNamespace(
        results=tmp_path / "res", scratch=tmp_path / "scr", figures=tmp_path / "figs"
    )
    reg = SimpleNamespace(get=lambda name: _FakeBackend(adata))
    # `resolve_stage_config` only special-cases a dict sub-block or a pydantic
    # model with `model_dump()` (see cellquorum.methods.context_access); a bare
    # SimpleNamespace falls through to `{}`, silently discarding every key
    # below (including marker_genes) instead of raising. A plain dict is the
    # convention every other stage's dispatch-level test already uses (see
    # tests/test_de_end_to_end.py's `_Cfg`).
    stage_cfg = {
        "enabled": True,
        "method": "metacells",
        "rep_key": "X_pca_harmony",
        "layer": "counts",
        "group_by": "ref_cell_type_granular_coarse",
        "target_metacell_size": 96,
        "min_cells": 100,
        "marker_genes": ["gene_0"],
        "seed": 0,
        "env_name": "metacells_env",
        "launcher": "python",
        "timeout_seconds": 60,
    }
    config = SimpleNamespace(metacells=stage_cfg, cohort=None)
    return SimpleNamespace(
        paths=paths,
        backend_registry=reg,
        config=config,
        adata=adata,
        require_adata=lambda: adata,
    )


def test_stage_run_produces_figures_and_metacell_id(tmp_path: Path) -> None:
    adata = _adata()
    stage = MetacellsStage()
    result = stage.run(_ctx(tmp_path, adata))

    assert result.status == "success"
    assert "metacell_id" in result.adata.obs.columns
    figure_artifacts = [a for a in result.artifacts if a.kind == "figure"]
    # network, heatmap, composition, 2 qc scatters, 1 marker trace = 6 distinct figures
    assert len(figure_artifacts) >= 5
    assert any("gene_0" in a.name for a in figure_artifacts)
