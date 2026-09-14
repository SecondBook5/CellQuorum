"""Tests for per-coarse-type granular (fine-label) embedding panels.

One figure per coarse cell type showing its OWN fine subtypes, not one
crowded global map of all ~80 fine labels at once. A coarse type whose fine
label never differs from its coarse label (no real subdivision in the
reference atlas) is skipped, not drawn as a flat single-color panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.visualization.granular_panels import plot_granular_panels_per_coarse_type


def _coords(n: int) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, 2))


def test_writes_one_panel_per_coarse_type_with_real_subdivision(tmp_path):
    n = 12
    coarse = pd.Series((["Fibroblasts"] * 6) + (["LEC"] * 6))
    fine = pd.Series((["Fibroblast_A"] * 3 + ["Fibroblast_B"] * 3) + (["LEC"] * 6))

    written = plot_granular_panels_per_coarse_type(_coords(n), fine, coarse, out_dir=tmp_path)

    # Fibroblasts has 2 fine labels -> gets a panel. LEC has 1 (== its own
    # coarse label) -> no real subdivision, skipped.
    assert set(written) == {"Fibroblasts"}
    assert written["Fibroblasts"].exists()
    assert written["Fibroblasts"].with_suffix(".pdf").exists()


def test_returns_empty_when_no_coarse_type_has_subdivision(tmp_path):
    n = 6
    coarse = pd.Series(["LEC"] * 3 + ["Neutrophils"] * 3)
    fine = pd.Series(["LEC"] * 3 + ["Neutrophils"] * 3)

    written = plot_granular_panels_per_coarse_type(_coords(n), fine, coarse, out_dir=tmp_path)

    assert written == {}


def test_crop_is_robust_to_a_realistic_outlier_fraction(tmp_path, monkeypatch):
    """A real coarse type can have ~10% of its cells scattered near OTHER
    clusters (genuine signal -- e.g. cells whose expression doesn't match
    their label), confirmed on the real atlas. A plain 1st/99th-percentile
    crop is not aggressive enough to stay tight against that much
    contamination; the crop must not stretch out to accommodate it."""
    from cellquorum.visualization import granular_panels as panels_module

    captured = {}

    def fake_save(fig, out_dir, stem, **kwargs):
        captured["fig"] = fig
        return [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]

    monkeypatch.setattr(panels_module, "save_figure", fake_save)

    n_main = 180
    n_outliers = 20  # 10% of the type, matching the real atlas's scattered fraction
    rng = np.random.default_rng(0)
    main_cluster = rng.normal(loc=0.0, scale=0.5, size=(n_main, 2))
    # Scattered near several OTHER (far-apart) clusters, not one single point.
    outliers = rng.normal(loc=0.0, scale=1.0, size=(n_outliers, 2)) + rng.choice(
        [[-50.0, -50.0], [50.0, 50.0], [50.0, -50.0]], size=n_outliers
    )
    coords = np.vstack([main_cluster, outliers])

    coarse = pd.Series(["Fibroblasts"] * (n_main + n_outliers))
    half = (n_main + n_outliers) // 2
    fine = pd.Series((["Fibroblast_A"] * half) + (["Fibroblast_B"] * (n_main + n_outliers - half)))

    plot_granular_panels_per_coarse_type(coords, fine, coarse, out_dir=tmp_path)

    ax = captured["fig"].axes[0]
    xlim = ax.get_xlim()
    # The crop must stay near the dense main cluster (scale ~1), not stretch
    # out to accommodate 10% of the type sitting near +/-50.
    assert xlim[1] - xlim[0] < 10


def test_no_resolution_types_are_reported(tmp_path):
    n = 12
    coarse = pd.Series((["Fibroblasts"] * 6) + (["LEC"] * 6))
    fine = pd.Series((["Fibroblast_A"] * 3 + ["Fibroblast_B"] * 3) + (["LEC"] * 6))

    written, no_resolution = plot_granular_panels_per_coarse_type(
        _coords(n), fine, coarse, out_dir=tmp_path, return_skipped=True
    )

    assert no_resolution == ["LEC"]
