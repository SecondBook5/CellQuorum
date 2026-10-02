"""Tests for the rare-population-triage condition-enrichment figure."""

from __future__ import annotations

import pandas as pd

from cellquorum.visualization.rare_population_triage import plot_condition_enrichment_bar


def _triage_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["A", "B", "C", "D"],
            "n_cells": [100, 50, 200, 10],
            "condition_enrichment": [0.3, -0.2, 0.05, 0.9],
            "category": ["candidate", "candidate", "candidate", "noise"],
        }
    )


def test_writes_png_and_a_vector_twin(tmp_path):
    out = plot_condition_enrichment_bar(_triage_df(), out_path=tmp_path / "enrich.png")

    assert out == tmp_path / "enrich.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_only_candidates_are_plotted(tmp_path, monkeypatch):
    from cellquorum.visualization import rare_population_triage as viz_module

    captured = {}

    def fake_save(fig, out_dir, stem, **kwargs):
        captured["fig"] = fig
        return [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]

    monkeypatch.setattr(viz_module, "save_figure", fake_save)

    plot_condition_enrichment_bar(_triage_df(), out_path=tmp_path / "enrich.png")

    ax = captured["fig"].axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "D" not in labels  # noise excluded
    assert set(labels) == {"A", "B", "C"}


def test_returns_none_when_no_candidates(tmp_path):
    df = pd.DataFrame(
        {
            "label": ["X"],
            "n_cells": [5],
            "condition_enrichment": [0.1],
            "category": ["noise"],
        }
    )

    out = plot_condition_enrichment_bar(df, out_path=tmp_path / "enrich.png")

    assert out is None
    assert not (tmp_path / "enrich.png").exists()
