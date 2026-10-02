"""Tests for the cross-compartment ambient-fix before/after embedding figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.visualization.ambient_specificity import plot_panel_before_after


def _coords(n=20):
    return np.random.default_rng(0).normal(size=(n, 2))


def _scores(n=20):
    rng = np.random.default_rng(1)
    before = pd.Series(rng.uniform(0, 4, size=n))
    after = before.copy()
    after.iloc[: n // 2] = 0.0  # simulate ambient removal for half the cells
    return before, after


def test_plot_panel_before_after_writes_two_standalone_files(tmp_path):
    before, after = _scores()

    written = plot_panel_before_after(
        _coords(),
        before,
        after,
        title_before="Keratin score — BEFORE",
        title_after="Keratin score — AFTER",
        out_dir=tmp_path,
    )

    assert set(written) == {"before", "after"}
    for path in written.values():
        assert path.exists()
        assert path.with_suffix(".pdf").exists()


def test_plot_panel_before_after_applies_the_house_theme(tmp_path, monkeypatch):
    from cellquorum.visualization import ambient_specificity as viz_module

    calls = []
    monkeypatch.setattr(viz_module, "apply_theme", lambda style, **kw: calls.append(style))
    before, after = _scores()

    plot_panel_before_after(
        _coords(), before, after, title_before="B", title_after="A", out_dir=tmp_path
    )

    assert calls == ["notebook"], "plot_panel_before_after must apply the notebook house theme"


def test_plot_panel_before_after_shares_one_color_scale(tmp_path, monkeypatch):
    """Before and after must be visually comparable -- same vmax, not each
    auto-scaled to its own max (which would make 'after' look unchanged even
    when most of its mass was actually removed)."""
    from cellquorum.visualization import ambient_specificity as viz_module

    captured = []

    def fake_save(fig, out_dir, stem, **kwargs):
        captured.append((stem, fig))
        return [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]

    monkeypatch.setattr(viz_module, "save_figure", fake_save)
    before, after = _scores()

    plot_panel_before_after(
        _coords(), before, after, title_before="B", title_after="A", out_dir=tmp_path
    )

    vmaxes = {}
    for stem, fig in captured:
        sc = fig.axes[0].collections[0]
        vmaxes[stem] = sc.norm.vmax

    assert len(set(vmaxes.values())) == 1, f"expected one shared vmax, got {vmaxes}"
