"""Tests for the PCA scree/elbow diagnostic figure.

Never had direct tests before -- it was only ever exercised incidentally inside full
pipeline runs (test_generalization_smoke.py, test_feature_selection_e2e.py), which write
it but assert nothing about it.
"""

from __future__ import annotations

import numpy as np

from cellquorum.stages.preprocessing.dimensionality.pca import write_scree_plot


def _captured_figure(monkeypatch, call):
    """Capture the Figure passed to save_cellquorum_figure before the caller closes it."""
    from cellquorum.visualization import figstyle

    captured = {}

    def fake_save(fig, path, **kwargs):
        captured["fig"] = fig
        return [path]

    monkeypatch.setattr(figstyle, "save_cellquorum_figure", fake_save)
    call()
    return captured["fig"]


def test_write_scree_plot_applies_the_house_theme(tmp_path, monkeypatch):
    """Every other CellQuorum figure calls apply_cellquorum_theme first; this one didn't."""
    from cellquorum.visualization import figstyle

    calls = []
    monkeypatch.setattr(figstyle, "apply_cellquorum_theme", lambda: calls.append(True))
    monkeypatch.setattr(figstyle, "save_cellquorum_figure", lambda fig, path, **kw: [path])

    variance_ratio = np.array([0.4, 0.2, 0.1, 0.05, 0.03])
    write_scree_plot(variance_ratio, chosen_n=3, output_path=tmp_path / "scree.png")

    assert calls, "write_scree_plot must apply the shared CellQuorum theme"


def test_write_scree_plot_writes_png_and_a_vector_twin(tmp_path):
    variance_ratio = np.array([0.5, 0.25, 0.1, 0.08, 0.07])
    out = tmp_path / "scree.png"

    write_scree_plot(variance_ratio, chosen_n=2, output_path=out)

    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_write_scree_plot_shows_at_most_30_components(tmp_path, monkeypatch):
    """Legibility cap: a 200-PC scree plot showing every bar is unreadable."""
    variance_ratio = np.linspace(0.1, 0.001, 200)

    fig = _captured_figure(
        monkeypatch,
        lambda: write_scree_plot(variance_ratio, chosen_n=10, output_path=tmp_path / "scree.png"),
    )

    (ax,) = [a for a in fig.axes if a.patches]
    assert len(ax.patches) == 30
