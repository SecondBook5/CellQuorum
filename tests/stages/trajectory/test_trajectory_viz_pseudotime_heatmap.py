# tests/stages/trajectory/test_trajectory_viz_pseudotime_heatmap.py
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


def test_categorical_track_colors_use_the_muted_atlas_palette(tmp_path):
    """Real complaint: the bright CATEGORICAL_PALETTE (meant for a handful of
    bars or a volcano plot) reads as cartoonish/clownish for a categorical
    identity track -- exactly the failure MUTED_PALETTE's own docstring was
    written to fix ("saturated primaries reads as cartoonish -- the clusters
    look like poster clip-art"), just never wired into this call site.
    """
    from cellquorum.visualization.figstyle import MUTED_PALETTE

    ctx, figures = _context(tmp_path)
    adata = _adata()
    adata.obs["fibroblast_subcluster"] = pd.Categorical(["0"] * 30 + ["1"] * 30)
    cfg = {
        "case": "LE",
        "control": "Normal",
        "condition_col": "condition",
        "heatmap_genes": [f"G{i}" for i in range(6)],
        "heatmap_categorical_keys": ["fibroblast_subcluster"],
    }
    out = PseudotimeHeatmapVizMethod()._run(adata, cfg, ctx)
    assert not isinstance(out, MethodSkip)
    tracks = PseudotimeHeatmapVizMethod()._resolve_categorical_tracks(adata, cfg)
    assert tracks[0]["colors"] == MUTED_PALETTE[:2]


def test_renders_two_categorical_tracks_via_heatmap_categorical_keys(tmp_path):
    """The real ask: subcluster AND cell-cycle-phase stacked as two separate
    tracks, not the single legacy heatmap_state_key row.
    """
    ctx, figures = _context(tmp_path)
    adata = _adata()
    adata.obs["fibroblast_subcluster"] = pd.Categorical(["0"] * 30 + ["1"] * 30)
    cfg = {
        "case": "LE",
        "control": "Normal",
        "condition_col": "condition",
        "heatmap_genes": [f"G{i}" for i in range(6)],
        "heatmap_categorical_keys": ["fibroblast_subcluster", "cell_type"],
        "heatmap_categorical_labels": {
            "fibroblast_subcluster": "Subcluster",
            "cell_type": "Group",
        },
        "heatmap_value_labels": {"0": "Perivascular", "1": "Mechanosensory"},
    }
    out = PseudotimeHeatmapVizMethod()._run(adata, cfg, ctx)
    assert not isinstance(out, MethodSkip)
    pngs = list((figures / "trajectory").glob("pseudotime_heatmap*.png"))
    assert pngs


def test_computes_cell_cycle_phase_when_requested(tmp_path):
    """heatmap_compute_cell_cycle + 'phase' in heatmap_categorical_keys must
    score real cell-cycle genes and use the result as a track, not silently
    drop 'phase' because no such obs column exists yet.
    """
    from cellquorum.stages.integration.embeddings.overlay import (
        TIROSH_G2M_GENES,
        TIROSH_S_GENES,
    )

    ctx, figures = _context(tmp_path)
    rng = np.random.default_rng(0)
    n = 60
    cc_genes = list(TIROSH_S_GENES[:5]) + list(TIROSH_G2M_GENES[:5])
    # score_genes_cell_cycle bins ALL genes by expression quantile to build a
    # control set for each scored gene (scanpy's score_genes machinery) -- too
    # few total genes leaves some bins empty ("No control genes found in any
    # cut"). 8 filler genes isn't enough; give it real headroom.
    n_filler = 200
    g = n_filler + len(cc_genes)
    X = rng.random((n, g)).astype("float32")
    obs = pd.DataFrame(
        {
            "dpt_pseudotime": np.linspace(0, 1, n),
            "condition": pd.Categorical(["Normal"] * (n // 2) + ["LE"] * (n - n // 2)),
        }
    )
    adata = ad.AnnData(X, obs=obs)
    adata.var_names = [f"G{i}" for i in range(n_filler)] + cc_genes
    assert "phase" not in adata.obs.columns

    cfg = {
        "case": "LE",
        "control": "Normal",
        "condition_col": "condition",
        "heatmap_genes": [f"G{i}" for i in range(6)],
        "heatmap_categorical_keys": ["phase"],
        "heatmap_compute_cell_cycle": True,
    }
    out = PseudotimeHeatmapVizMethod()._run(adata, cfg, ctx)
    assert not isinstance(out, MethodSkip)
    assert "phase" in adata.obs.columns
    assert set(adata.obs["phase"].unique()) <= {"G1", "S", "G2M"}
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


def test_gene_groups_drive_row_order_and_labels(tmp_path, monkeypatch):
    """heatmap_gene_groups is the gene list AND the grouping: the figure must receive rows grouped
    in the declared order, each row tagged with its group."""
    from cellquorum.stages.trajectory.viz import _pseudotime_plots

    seen = {}
    real = _pseudotime_plots.heatmap.condition_split_heatmap

    def spy(profiles, score_tracks, categorical_tracks, gene_labels, gene_order, **kwargs):
        seen["rows"] = [gene_labels[i] for i in gene_order]
        seen["groups"] = kwargs.get("gene_group_labels")
        return real(profiles, score_tracks, categorical_tracks, gene_labels, gene_order, **kwargs)

    monkeypatch.setattr(_pseudotime_plots.heatmap, "condition_split_heatmap", spy)
    ctx, _ = _context(tmp_path)
    cfg = {
        "case": "LE",
        "control": "Normal",
        "condition_col": "condition",
        "heatmap_gene_groups": {"First": ["G3", "G1"], "Second": ["G5", "G0", "NOT_A_GENE"]},
    }
    out = PseudotimeHeatmapVizMethod()._run(_adata(), cfg, ctx)
    assert not isinstance(out, MethodSkip)
    assert seen["groups"] == ["First", "First", "Second", "Second"]
    assert set(seen["rows"][:2]) == {"G3", "G1"}
    assert set(seen["rows"][2:]) == {"G5", "G0"}


def test_gene_groups_and_a_flat_gene_list_together_are_a_config_error(tmp_path):
    """One switch per decision: two settings that both name the genes would silently disagree."""
    import pytest

    from cellquorum.core.exceptions import CellQuorumConfigError

    ctx, _ = _context(tmp_path)
    cfg = {
        "case": "LE",
        "control": "Normal",
        "condition_col": "condition",
        "heatmap_genes": ["G0", "G1"],
        "heatmap_gene_groups": {"A": ["G2"]},
    }
    with pytest.raises(CellQuorumConfigError, match="heatmap_gene_groups"):
        PseudotimeHeatmapVizMethod()._run(_adata(), cfg, ctx)
