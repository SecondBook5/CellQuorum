import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.context import PipelineContext, PipelinePaths
from cellquorum.trajectory_viz.stage import TrajectoryVizStage


def test_config_exposes_trajectory_viz_defaults():
    cfg = CellQuorumConfig()
    assert cfg.stages.trajectory_viz is True
    assert cfg.trajectory_viz.enabled is True


def test_stage_dispatch_renders_pseudotime(tmp_path):
    a = ad.AnnData(np.zeros((30, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(30)]
    a.obsm["X_umap"] = np.random.RandomState(0).rand(30, 2)
    a.obs["dpt_pseudotime"] = np.linspace(0, 1, 30)

    cfg = CellQuorumConfig()
    cfg.trajectory_viz.figure_formats = ["png"]
    cfg.trajectory_viz.dpi = 72

    paths = PipelinePaths.from_output_dir(tmp_path)
    paths.ensure_directories()
    ctx = PipelineContext(config=cfg, paths=paths, manifest=None, adata=a)

    TrajectoryVizStage().run(ctx)
    figs = list((paths.figures / "trajectory").glob("pseudotime_dpt_pseudotime.png"))
    assert len(figs) == 1
