# tests/test_trajectory_viz_pseudotime_heatmap.py
import matplotlib

matplotlib.use("Agg")
import types

import anndata as ad
import numpy as np
import pandas as pd

import cellquorum.stages.trajectory.viz  # noqa: F401  (registers methods)
from cellquorum.methods.base import MethodSkip
from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.trajectory.viz._pseudotime_plots import PseudotimeHeatmapVizMethod


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


def test_does_not_densify_the_full_matrix_for_gene_selection_or_extraction(tmp_path):
    """Both the correlation-prefilter fallback and the gene-value extraction must
    not densify the WHOLE expression matrix.

    On a real cohort (201,871 x 33,417) that's ~54 GB, to either (a) compute a
    per-gene correlation that only ever needs one n_cells-length column at a
    time, or (b) keep <= heatmap_max_genes (default 60) columns after throwing
    the rest away. Same class of bug already fixed several times elsewhere
    this session (decoupler densification, adata.copy() before write_h5ad).
    """
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n, g = 60, 8
    dense_X = rng.random((n, g)).astype("float32")
    sparse_X = sp.csr_matrix(dense_X)
    obs = pd.DataFrame(
        {
            "dpt_pseudotime": np.linspace(0, 1, n),
            "G2M_score": rng.random(n),
            "cell_type": pd.Categorical(["A"] * (n // 2) + ["B"] * (n - n // 2)),
        }
    )
    adata = ad.AnnData(sparse_X.copy(), obs=obs)
    adata.var_names = [f"G{i}" for i in range(g)]

    original_toarray = sp.csr_matrix.toarray

    def _guarded_toarray(self, *args, **kwargs):
        if self.shape == sparse_X.shape:
            raise AssertionError(
                "pseudotime_heatmap densified the FULL expression matrix instead "
                "of a single column or the already-subset gene selection."
            )
        return original_toarray(self, *args, **kwargs)

    sp.csr_matrix.toarray = _guarded_toarray
    try:
        ctx, figures = _context(tmp_path)
        # No heatmap_genes configured: exercises the correlation-prefilter
        # fallback in _select_genes, not just the gene-value extraction.
        cfg = {"heatmap_max_genes": 4}
        out = PseudotimeHeatmapVizMethod()._run(adata, cfg, ctx)
    finally:
        sp.csr_matrix.toarray = original_toarray

    assert not isinstance(out, MethodSkip)
    pngs = list((figures / "trajectory").glob("pseudotime_heatmap*.png"))
    assert pngs
