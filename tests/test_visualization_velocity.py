"""Regression tests for the publication velocity renderer.

Both properties here were defects found on the real LEC arm, not hypotheticals:
the flagship velocity figure failed to save at all (and left a truncated PDF
behind), and when it did save it was colored by a column that put 98.7% of the
object in one category.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.visualization.velocity import (
    VelocityRenderError,
    resolve_group_key,
    velocity_stream_figure,
)


def _adata_with_obs(obs: dict[str, list], n: int) -> ad.AnnData:
    frame = pd.DataFrame(obs, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=np.zeros((n, 2), dtype=np.float32), obs=frame)


# ---------------------------------------------------------------------------
# resolve_group_key
# ---------------------------------------------------------------------------


def test_resolve_group_key_rejects_a_column_that_does_not_partition():
    """The real LEC case: 13 levels, 1840/1864 cells in one of them.

    ``cell_type_granular`` passed a naive "2+ levels" test and produced a figure
    that was one uniform color plus a twelve-entry legend for invisible
    singletons, while ``leiden`` — last in the candidate list — had 15 populated
    clusters showing the within-lineage structure the figure exists for.
    """
    n = 1864
    granular = ["LEC"] * 1840 + [f"stray_{i // 2}" for i in range(24)]
    leiden = [str(i % 15) for i in range(n)]
    adata = _adata_with_obs({"cell_type_granular": granular, "leiden": leiden}, n)

    key = resolve_group_key(adata, None, ("cell_type_granular", "leiden"))
    assert key == "leiden"


def test_resolve_group_key_honors_candidate_order_between_usable_columns():
    """Order encodes the caller's preference for the most specific naming."""
    n = 200
    adata = _adata_with_obs(
        {
            "cell_type_granular": ["LEC_a"] * 100 + ["LEC_b"] * 100,
            "leiden": [str(i % 4) for i in range(n)],
        },
        n,
    )
    assert resolve_group_key(adata, None, ("cell_type_granular", "leiden")) == "cell_type_granular"


def test_resolve_group_key_honors_a_configured_key_it_would_not_have_chosen():
    """A named key is the caller's call; this must not second-guess it."""
    n = 1000
    adata = _adata_with_obs(
        {
            "ref_state": ["LEC"] * 998 + ["other", "other"],
            "leiden": [str(i % 8) for i in range(n)],
        },
        n,
    )
    assert resolve_group_key(adata, "ref_state", ("leiden",)) == "ref_state"
    # But a configured key with only one level is not a grouping at all, so it
    # falls through to the candidates rather than drawing a single-color figure.
    adata.obs["ref_state"] = ["LEC"] * n
    assert resolve_group_key(adata, "ref_state", ("leiden",)) == "leiden"


def test_resolve_group_key_falls_back_weakly_rather_than_returning_nothing():
    """A weakly colored figure still beats no figure."""
    n = 500
    adata = _adata_with_obs({"cell_type": ["LEC"] * 498 + ["x", "y"]}, n)
    assert resolve_group_key(adata, None, ("cell_type",)) == "cell_type"
    # Nothing usable at all.
    assert resolve_group_key(adata, None, ("absent", "")) is None


# ---------------------------------------------------------------------------
# velocity_stream_figure
# ---------------------------------------------------------------------------


def _velocity_adata(seed: int = 0, n: int = 1500) -> ad.AnnData:
    """A ring of cells with rotational velocity: a large hole and a curved rim.

    Geometry is the whole point of this fixture. ``compute_velocity_on_grid``
    NaNs out every grid cell below ``min_mass`` — that mask is how streamplot
    knows not to draw in empty regions — and the failure mode only fires where a
    drawn streamline's interpolation stencil reaches into a masked cell. Two
    tidy separated blobs do NOT reproduce it (tried: streamlines stay well
    inside the dense region), whereas a thin annulus, whose valid band is
    everywhere adjacent to mask on both sides, does.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    radius = 6.0 + rng.normal(0.0, 0.5, n)
    xy = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    adata = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32))
    # Four arcs, so the figure has something to color and label.
    quadrant = np.floor(theta / (np.pi / 2)).astype(int).clip(0, 3)
    adata.obs["leiden"] = pd.Categorical([str(q) for q in quadrant])
    adata.obsm["X_umap"] = xy.astype(np.float32)
    # Tangential flow around the ring.
    adata.obsm["velocity_umap"] = np.c_[-np.sin(theta), np.cos(theta)].astype(np.float32)
    return adata


def test_velocity_stream_figure_hands_streamplot_a_finite_linewidth(monkeypatch):
    """The linewidth passed to streamplot must be finite everywhere.

    streamplot masks non-finite *u/v* itself, but interpolates ``linewidth`` per
    streamline vertex WITHOUT masking, so a NaN speed propagates into the
    linewidth of any streamline whose stencil touches a masked cell. This asserts
    the contract directly rather than through a backend, so it holds regardless
    of matplotlib's integration details.
    """
    from matplotlib.axes import Axes

    captured: list[np.ndarray] = []
    real_streamplot = Axes.streamplot

    def _capture(self, x, y, u, v, **kwargs):
        captured.append(np.asarray(kwargs["linewidth"], dtype=float))
        return real_streamplot(self, x, y, u, v, **kwargs)

    monkeypatch.setattr(Axes, "streamplot", _capture)
    velocity_stream_figure(_velocity_adata(), group_key="leiden", density=3.0)

    assert captured, "streamplot was never called"
    linewidth = captured[0]
    assert np.isfinite(linewidth).all(), "NaN linewidth reached streamplot"
    # And it is still a real speed encoding, not a flattened constant.
    assert linewidth.min() < linewidth.max()


def test_velocity_stream_figure_saves_to_pdf_with_a_mostly_masked_grid(tmp_path):
    """End-to-end guard: the real failure was invisible until PDF export.

    On the LEC arm this raised ``ValueError: Can only output finite numbers in
    PDF`` from the PDF backend *after* it had written part of the stream, leaving
    a truncated 38 KB ``velocity_stream.pdf``.

    The ``(density, stream_density)`` pair below is not arbitrary — it was chosen
    by checking which combinations reproduce the pre-fix failure through this
    whole function, and most do not. That is the nature of the bug rather than a
    weak fixture: reproducing needs one streamline vertex whose interpolation
    stencil reaches a masked cell, and ONE such vertex out of several hundred is
    enough to lose the entire figure. Because that makes the end-to-end
    reproduction knife-edge across matplotlib versions, the finite-linewidth test
    above is the primary guard and this one is the export smoke test.
    """
    from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid

    adata = _velocity_adata()
    _, v_grid = compute_velocity_on_grid(
        X_emb=np.asarray(adata.obsm["X_umap"], dtype=float),
        V_emb=np.asarray(adata.obsm["velocity_umap"], dtype=float),
        density=3.0,
        smooth=0.6,
        min_mass=1.5,
        adjust_for_stream=True,
    )
    # Guard the guard: without masked grid cells this test exercises nothing.
    assert not np.isfinite(np.asarray(v_grid)).all(), "fixture no longer masks any grid cell"

    fig = velocity_stream_figure(adata, group_key="leiden", density=3.0, stream_density=2.0)
    out = tmp_path / "velocity_stream.pdf"
    fig.savefig(out, format="pdf")
    assert out.stat().st_size > 0


def test_velocity_stream_figure_raises_a_named_error_without_an_embedding():
    adata = ad.AnnData(X=np.zeros((20, 2), dtype=np.float32))
    adata.obs["leiden"] = pd.Categorical(["0"] * 10 + ["1"] * 10)
    with pytest.raises(VelocityRenderError, match="X_umap"):
        velocity_stream_figure(adata, group_key="leiden")


def test_velocity_stream_figure_raises_when_the_group_key_is_absent():
    adata = _velocity_adata()
    with pytest.raises(VelocityRenderError, match="not in obs"):
        velocity_stream_figure(adata, group_key="nope")
