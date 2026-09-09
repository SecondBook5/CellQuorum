import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.stages.integration.embeddings import plots
from cellquorum.visualization import figstyle


def test_palette_is_shared_with_figstyle():
    # Categorical colors come from figstyle's shared authority, not a private
    # hardcoded list in the plotter — one source of truth for the house look.
    # The plotter now uses palette_colors (validated CATEGORICAL_PALETTE first,
    # then the generator past its size) rather than the raw generator, so the
    # atlas gets the audited hues instead of raw golden-angle vivids.
    assert plots.palette_colors is figstyle.palette_colors


def test_continuous_overlay_respects_explicit_vmin_vmax():
    coords = np.random.default_rng(0).random((30, 2))
    values = np.linspace(0, 10, 30)
    fig = plots.continuous_overlay(
        coords,
        values,
        title="t",
        axis_labels=("U1", "U2"),
        cmap="viridis",
        vmin=-2,
        vmax=2,
    )
    coll = fig.axes[0].collections[0]
    assert coll.norm.vmin == -2
    assert coll.norm.vmax == 2


def test_continuous_overlay_clip_pct_sets_symmetric_limits():
    coords = np.random.default_rng(0).random((100, 2))
    values = np.concatenate([np.linspace(0, 1, 99), [1000.0]])  # one outlier
    fig = plots.continuous_overlay(
        coords,
        values,
        title="t",
        axis_labels=("U1", "U2"),
        clip_pct=2.0,
    )
    coll = fig.axes[0].collections[0]
    # Outlier is clipped out of the color scale.
    assert coll.norm.vmax < 1000.0


def test_magic_zscore_layer_writes_when_present():
    rng = np.random.default_rng(0)
    adata = ad.AnnData(rng.random((20, 5)).astype("float32"))
    adata.layers["magic"] = rng.random((20, 5)).astype("float32")
    assert plots.magic_zscore_layer(adata) is True
    z = adata.layers["magic_z"]
    # Per-gene mean ~0.
    assert np.allclose(z.mean(0), 0, atol=1e-5)


def test_magic_zscore_layer_skips_when_absent():
    adata = ad.AnnData(np.zeros((5, 3), dtype="float32"))
    assert plots.magic_zscore_layer(adata) is False
    assert "magic_z" not in adata.layers


def test_atlas_subset_excludes_probable_multiplets():
    """A flagged doublet is not a cell type, so it is dropped from the restricted atlas

    panel but kept in the _allcells twin. This clears the central salt-and-pepper
    mixing zone, which is ~12% doublets, without touching the genuinely intermediate
    cells around it.
    """
    import pandas as pd

    from cellquorum.stages.integration.embeddings.methods import CategoricalEmbeddingMethod

    n = 1000
    adata = ad.AnnData(np.zeros((n, 2), dtype="float32"))
    state = np.array(["core"] * 800 + ["rescued"] * 100 + ["unresolved_borderline"] * 100)
    mult = np.zeros(n, dtype=bool)
    mult[:50] = True  # 50 core cells are also probable multiplets
    adata.obs["qc_state_final"] = pd.Categorical(state)
    adata.obs["qc_probable_multiplet"] = mult

    method = CategoricalEmbeddingMethod()
    cfg = {"qc_state_column": "qc_state_final", "atlas_states": ["core", "rescued"]}
    subsets = method._resolve_subsets(adata, cfg)

    suffix, mask = subsets[0]
    assert "nodoublet" in suffix
    assert int(mask.sum()) == 850  # 900 core+rescued minus 50 doublets
    assert subsets[1] == ("_allcells", None)  # twin keeps everything

    # Opt-out restores the full core+rescued set.
    off = method._resolve_subsets(adata, {**cfg, "exclude_multiplets": False})
    assert int(off[0][1].sum()) == 900
