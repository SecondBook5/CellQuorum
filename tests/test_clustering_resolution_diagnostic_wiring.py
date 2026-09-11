"""The resolution diagnostic is opt-in and wired through LeidenMethod, not a separate stage.

Off by default: no extra cost, no figure, no note -- the whole point of making it a
config flag rather than always-on.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum.stages.clustering.neighbors_leiden import LeidenMethod


class _Paths:
    def __init__(self, figures):
        self.figures = figures


class _Context:
    def __init__(self, figures):
        self.paths = _Paths(figures)


def _blobs(n_per_blob: int = 30, n_blobs: int = 3, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n = n_per_blob * n_blobs
    coords = rng.normal(scale=0.4, size=(n, 5)).astype(np.float32)
    for i in range(n_blobs):
        coords[i * n_per_blob : (i + 1) * n_per_blob, :2] += rng.normal(
            loc=i * 15.0, scale=0.5, size=2
        )
    adata = ad.AnnData(X=rng.normal(size=(n, 10)).astype(np.float32))
    adata.obsm["X_pca"] = coords
    return adata


def _config(**overrides) -> dict:
    cfg = {"n_neighbors": 10, "resolution": 1.0, "random_state": 0, "key_added": "leiden"}
    cfg.update(overrides)
    return cfg


def test_disabled_by_default_produces_no_diagnostic_note_or_artifact(tmp_path):
    adata = _blobs()
    result = LeidenMethod()._run(adata, _config(), context=_Context(tmp_path))

    assert not any("Resolution diagnostic" in n for n in result.notes)
    assert result.artifacts == []


def test_enabled_writes_a_figure_and_a_note_without_changing_the_used_resolution(tmp_path):
    adata = _blobs()
    config = _config(
        resolution_diagnostic={
            "enabled": True,
            "resolutions": [0.6, 1.0],
            "n_bootstraps": 2,
            "subsample_fraction": 0.8,
            "adaptive": False,
        }
    )
    result = LeidenMethod()._run(adata, config, context=_Context(tmp_path))

    assert any("Resolution diagnostic" in n for n in result.notes)
    assert result.metrics["resolution"] == 1.0  # unchanged by the diagnostic
    assert len(result.artifacts) == 1
    assert result.artifacts[0].kind == "figure"
    assert Path(result.artifacts[0].path).exists()


def test_enabled_without_a_figures_dir_still_adds_the_note_but_no_artifact():
    adata = _blobs()
    config = _config(
        resolution_diagnostic={"enabled": True, "resolutions": [1.0], "n_bootstraps": 2}
    )

    class _NoPathsContext:
        pass

    result = LeidenMethod()._run(adata, config, context=_NoPathsContext())

    assert any("Resolution diagnostic" in n for n in result.notes)
    assert result.artifacts == []
