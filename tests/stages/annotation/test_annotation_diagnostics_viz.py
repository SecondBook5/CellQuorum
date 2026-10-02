"""Tests for annotation_diagnostics figures (scDiagnostics anomaly/entropy/kNN).

One plot per file, not a combined grid -- a standing preference over this
pipeline's default multi-panel figures.
"""

from __future__ import annotations

import pandas as pd

from cellquorum.stages.annotation.diagnostics.viz import (
    plot_anomaly_by_cell_type,
    plot_entropy_by_cell_type,
    plot_knn_confidence,
    write_annotation_diagnostics_figures,
)


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_type": ["Fibroblasts", "Fibroblasts", "DC", "DC", "T/NK"],
            "scdiag_anomaly": [0.1, 0.2, 0.6, 0.7, 0.3],
            "scdiag_anomaly_flag": [False, False, True, True, False],
            "scdiag_entropy": [0.05, 0.1, 0.8, 0.9, 0.4],
            "scdiag_knn_prob_group": [0.95, 0.95, 0.40, 0.40, 0.80],
            "scdiag_knn_prob_group_n": [1000, 1000, 50, 50, 300],
        }
    )


def test_plot_anomaly_by_cell_type_writes_png_and_a_vector_twin(tmp_path):
    out = tmp_path / "anomaly.png"
    result = plot_anomaly_by_cell_type(_obs(), "cell_type", out)

    assert result == out
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_anomaly_by_cell_type_returns_none_without_the_column(tmp_path):
    obs = pd.DataFrame({"cell_type": ["Fibroblasts"], "unrelated": [1]})
    out = tmp_path / "anomaly.png"

    result = plot_anomaly_by_cell_type(obs, "cell_type", out)

    assert result is None
    assert not out.exists()


def test_plot_entropy_by_cell_type_writes_png_and_a_vector_twin(tmp_path):
    out = tmp_path / "entropy.png"
    result = plot_entropy_by_cell_type(_obs(), "cell_type", out)

    assert result == out
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_entropy_by_cell_type_returns_none_without_the_column(tmp_path):
    obs = pd.DataFrame({"cell_type": ["Fibroblasts"], "unrelated": [1]})
    out = tmp_path / "entropy.png"

    result = plot_entropy_by_cell_type(obs, "cell_type", out)

    assert result is None
    assert not out.exists()


def test_plot_knn_confidence_writes_png_and_a_vector_twin(tmp_path):
    out = tmp_path / "knn.png"
    result = plot_knn_confidence(_obs(), "cell_type", out)

    assert result == out
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_knn_confidence_reduces_to_one_bar_per_cell_type(tmp_path, monkeypatch):
    """scdiag_knn_prob_group is a population-level stat broadcast onto every
    member cell -- the figure must reduce it back to one bar per cell type
    (3), not one bar per row (5).
    """
    from cellquorum.stages.annotation.diagnostics import viz as viz_module

    captured = {}

    def fake_save(fig, path, **kwargs):
        captured["fig"] = fig
        return [path]

    monkeypatch.setattr(viz_module, "save_cellquorum_figure", fake_save)

    plot_knn_confidence(_obs(), "cell_type", tmp_path / "knn.png")

    ax = captured["fig"].axes[0]
    bar_patches = [p for p in ax.patches]
    assert len(bar_patches) == 3
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert sorted(labels) == ["DC", "Fibroblasts", "T/NK"]


def test_plot_knn_confidence_returns_none_without_the_column(tmp_path):
    obs = pd.DataFrame({"cell_type": ["Fibroblasts"], "unrelated": [1]})
    out = tmp_path / "knn.png"

    result = plot_knn_confidence(obs, "cell_type", out)

    assert result is None
    assert not out.exists()


def test_write_annotation_diagnostics_figures_writes_all_three(tmp_path):
    written = write_annotation_diagnostics_figures(_obs(), "cell_type", tmp_path)

    assert set(written) == {"anomaly", "entropy", "knn_confidence"}
    for path in written.values():
        assert path.exists()


def test_write_annotation_diagnostics_figures_skips_missing_diagnostics(tmp_path):
    obs = pd.DataFrame({"cell_type": ["Fibroblasts", "DC"], "scdiag_entropy": [0.1, 0.5]})

    written = write_annotation_diagnostics_figures(obs, "cell_type", tmp_path)

    assert set(written) == {"entropy"}
    assert (tmp_path / "scdiagnostics_entropy.png").exists()
    assert not (tmp_path / "scdiagnostics_anomaly.png").exists()
    assert not (tmp_path / "scdiagnostics_knn_confidence.png").exists()
