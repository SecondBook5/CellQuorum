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


def test_condition_enrichment_exports_only_candidate_populations(tmp_path):
    from unittest.mock import patch

    from cellquorum.visualization import rare_population_triage as viz

    with patch.object(viz, "save_figure", wraps=viz.save_figure) as save:
        out = plot_condition_enrichment_bar(_triage_df(), out_path=tmp_path / "enrich.png")

    assert out == tmp_path / "enrich.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    fig = save.call_args.args[0]
    labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
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
