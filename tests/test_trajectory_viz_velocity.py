import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.stage import StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.trajectory.viz._kernel_plots import VelocityVizMethod


class _Ctx:
    def __init__(self, tmp):
        class P:
            results = tmp
            figures = tmp / "f"

        self.paths = P()


def test_skips_without_velocity_h5ads(tmp_path):
    res = VelocityVizMethod()._run(ad.AnnData(np.zeros((2, 2))), {}, _Ctx(tmp_path))
    assert isinstance(res, MethodSkip)


def test_resolves_group_key_from_cluster_key_not_the_generic_fallback(tmp_path, monkeypatch):
    """Real bug: velocity_viz read a config['group_key'] that no config schema
    field or stage whitelist ever set, so it always fell through to its own
    default candidate chain and landed on 'cell_type_granular' -- a coarse,
    auto-named atlas column, dominated by one value, instead of the project's
    real, correctly-populated 'fibroblast_subcluster' identities that every
    other trajectory_viz method already honors via config['cluster_key'].
    """
    n = 200
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    xy = np.c_[6 * np.cos(theta), 6 * np.sin(theta)].astype(np.float32)
    adata = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32))
    adata.obsm["X_umap"] = xy
    adata.obsm["velocity_umap"] = np.c_[-np.sin(theta), np.cos(theta)].astype(np.float32)
    # The real trap: dominated by one value, but still "2+ levels".
    adata.obs["cell_type_granular"] = pd.Categorical(
        ["Fibroblast SLPI+"] * (n - 6) + [f"stray_{i}" for i in range(6)]
    )
    adata.obs["fibroblast_subcluster"] = pd.Categorical([str(i % 4) for i in range(n)])

    vel_dir = tmp_path / "trajectory" / "velocity"
    vel_dir.mkdir(parents=True)
    adata.write_h5ad(vel_dir / "whole_object.h5ad")

    import cellquorum.visualization.velocity as velocity_viz

    real_figure = velocity_viz.velocity_stream_figure
    captured = {}

    def _spy(sub, *, group_key, group_labels=None, **kwargs):
        captured["group_key"] = group_key
        captured["group_labels"] = group_labels
        return real_figure(sub, group_key=group_key, group_labels=group_labels, **kwargs)

    monkeypatch.setattr(velocity_viz, "velocity_stream_figure", _spy)

    config = {
        "cluster_key": "fibroblast_subcluster",
        "fate_labels": {"0": "Papillary", "1": "Mechanosensory"},
    }
    out = VelocityVizMethod()._run(ad.AnnData(np.zeros((2, 2))), config, _Ctx(tmp_path))

    assert isinstance(out, StageResult)
    assert captured["group_key"] == "fibroblast_subcluster"
    assert captured["group_labels"] == {"0": "Papillary", "1": "Mechanosensory"}
