"""Tests for reference-mapping diagnostics (loss curves + uncertainty histograms).

Had zero test coverage before -- not even a "does it run" check.
"""

from __future__ import annotations

import pandas as pd

from cellquorum.stages.annotation.reference_mapping.diagnostics import (
    plot_loss_curves,
    plot_uncertainty,
)


def _captured_figure(monkeypatch, call):
    """Capture the Figure passed to save_cellquorum_figure before the caller closes it."""
    from cellquorum.stages.annotation.reference_mapping import diagnostics as diag_module

    captured = {}

    def fake_save(fig, path, **kwargs):
        captured["fig"] = fig
        return [path]

    monkeypatch.setattr(diag_module, "save_cellquorum_figure", fake_save)
    call()
    return captured["fig"]


def test_plot_loss_curves_applies_the_house_theme(tmp_path, monkeypatch):
    from cellquorum.stages.annotation.reference_mapping import diagnostics as diag_module

    calls = []
    monkeypatch.setattr(diag_module, "apply_cellquorum_theme", lambda: calls.append(True))
    monkeypatch.setattr(diag_module, "save_cellquorum_figure", lambda fig, path, **kw: [path])

    loss_history = {"scvi": {"elbo": [10.0, 8.0, 6.0]}}
    plot_loss_curves(loss_history, tmp_path / "loss.png")

    assert calls, "plot_loss_curves must apply the shared CellQuorum theme"


def test_plot_uncertainty_applies_the_house_theme(tmp_path, monkeypatch):
    from cellquorum.stages.annotation.reference_mapping import diagnostics as diag_module

    calls = []
    monkeypatch.setattr(diag_module, "apply_cellquorum_theme", lambda: calls.append(True))
    monkeypatch.setattr(diag_module, "save_cellquorum_figure", lambda fig, path, **kw: [path])

    obs = pd.DataFrame({"transfer_knn_entropy": [0.1, 0.5, 0.9]})
    plot_uncertainty(obs, "transfer", tmp_path / "uncertainty.png")

    assert calls, "plot_uncertainty must apply the shared CellQuorum theme"


def test_plot_loss_curves_writes_png_and_a_vector_twin(tmp_path):
    loss_history = {"scvi": {"elbo": [10.0, 8.0, 6.0], "reconstruction": [5.0, 4.0, 3.0]}}
    out = tmp_path / "loss.png"

    plot_loss_curves(loss_history, out)

    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_loss_curves_handles_a_missing_phase_without_raising(tmp_path):
    # Only scvi provided; scanvi/query_surgery must render as "No data", not raise.
    plot_loss_curves({"scvi": {"elbo": [1.0, 0.5]}}, tmp_path / "loss.png")
    assert (tmp_path / "loss.png").exists()


def test_plot_uncertainty_writes_png_and_a_vector_twin(tmp_path):
    obs = pd.DataFrame(
        {
            "transfer_knn_entropy": [0.1, 0.4, 0.9],
            "transfer_knn_agreement": [0.9, 0.6, 0.2],
            "transfer_consensus_frac": [1.0, 0.8, 0.5],
        }
    )
    out = tmp_path / "uncertainty.png"

    plot_uncertainty(obs, "transfer", out)

    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_uncertainty_handles_missing_columns_without_raising(tmp_path):
    obs = pd.DataFrame({"unrelated": [1, 2, 3]})
    plot_uncertainty(obs, "transfer", tmp_path / "uncertainty.png")
    assert (tmp_path / "uncertainty.png").exists()
