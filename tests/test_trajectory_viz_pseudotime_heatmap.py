# tests/test_trajectory_viz_pseudotime_heatmap.py
import matplotlib

matplotlib.use("Agg")
import types

import anndata as ad
import numpy as np
import pandas as pd

import cellquorum.trajectory_viz  # noqa: F401  (registers methods)
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.trajectory_viz.pseudotime_heatmap_viz import PseudotimeHeatmapVizMethod


def _adata(with_condition=True, with_pt=True):
    rng = np.random.default_rng(0)
    n, g = 60, 8
    X = rng.random((n, g)).astype("float32")
    obs = {}
    if with_pt:
        obs["dpt_pseudotime"] = np.linspace(0, 1, n)
    obs["G2M_score"] = rng.random(n)
    obs["cell_type"] = pd.Categorical(["A"] * (n // 2) + ["B"] * (n - n // 2))
    if with_condition:
        obs["condition"] = pd.Categorical(["Normal"] * (n // 2) + ["LE"] * (n - n // 2))
    adata = ad.AnnData(X, obs=pd.DataFrame(obs))
    adata.var_names = [f"G{i}" for i in range(g)]
    return adata


def _context(tmp_path, case="LE", control="Normal"):
    figures = tmp_path / "figures"
    design = types.SimpleNamespace(case=case, control=control, condition_col="condition")
    config = types.SimpleNamespace(design=design, trajectory_viz=None)
    paths = types.SimpleNamespace(results=str(tmp_path / "results"), figures=str(figures))
    return types.SimpleNamespace(config=config, paths=paths), figures


def test_registered():
    assert METHOD_REGISTRY.has("trajectory_viz", "pseudotime_heatmap")


def test_skips_when_no_pseudotime(tmp_path):
    ctx, _ = _context(tmp_path)
    out = PseudotimeHeatmapVizMethod()._run(
        _adata(with_pt=False), {"case": "LE", "control": "Normal"}, ctx
    )
    assert isinstance(out, MethodSkip)


def test_renders_condition_split(tmp_path):
    ctx, figures = _context(tmp_path)
    cfg = {
        "case": "LE",
        "control": "Normal",
        "condition_col": "condition",
        "heatmap_genes": [f"G{i}" for i in range(6)],
        "heatmap_score_key": "G2M_score",
        "heatmap_state_key": "cell_type",
    }
    out = PseudotimeHeatmapVizMethod()._run(_adata(), cfg, ctx)
    assert not isinstance(out, MethodSkip)
    pngs = list((figures / "trajectory").glob("pseudotime_heatmap*.png"))
    assert pngs


def test_single_panel_when_no_condition(tmp_path):
    ctx, figures = _context(tmp_path)
    cfg = {"heatmap_genes": [f"G{i}" for i in range(6)]}
    out = PseudotimeHeatmapVizMethod()._run(_adata(with_condition=False), cfg, ctx)
    assert not isinstance(out, MethodSkip)


def test_skips_on_non_numeric_pseudotime(tmp_path):
    """Regression: non-numeric pseudotime must skip, not crash."""
    ctx, _ = _context(tmp_path)
    rng = np.random.default_rng(0)
    n, g = 60, 8
    X = rng.random((n, g)).astype("float32")
    obs = pd.DataFrame(
        {
            "dpt_pseudotime": pd.Categorical(["early"] * (n // 2) + ["late"] * (n - n // 2)),
            "G2M_score": rng.random(n),
            "cell_type": pd.Categorical(["A"] * (n // 2) + ["B"] * (n - n // 2)),
            "condition": pd.Categorical(["Normal"] * (n // 2) + ["LE"] * (n - n // 2)),
        }
    )
    adata = ad.AnnData(X, obs=obs)
    adata.var_names = [f"G{i}" for i in range(g)]
    cfg = {"heatmap_genes": [f"G{i}" for i in range(6)], "condition_col": "condition"}
    out = PseudotimeHeatmapVizMethod()._run(adata, cfg, ctx)
    assert isinstance(out, MethodSkip)
    assert "input/render failed" in out.reason
