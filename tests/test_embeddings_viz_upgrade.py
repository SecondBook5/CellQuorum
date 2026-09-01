import matplotlib

matplotlib.use("Agg")
import anndata as ad
import numpy as np

from cellquorum.stages.integration.embeddings import plots
from cellquorum.visualization import figstyle


def test_palette_is_shared_with_figstyle():
    # Categorical colors come from figstyle's shared generator, not a private
    # hardcoded list in the plotter — one source of truth for the house look.
    assert plots.distinct_palette is figstyle.distinct_palette


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
