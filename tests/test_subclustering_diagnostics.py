"""Tests for subclustering diagnostic plots."""

from __future__ import annotations

from cellquorum.stages.clustering.subclustering.diagnostics import (
    plot_technical_qc_consistency,
)


def _consistency_result() -> dict:
    return {
        "clusters": {
            "0": {
                "n_cells": 100,
                "qc_pass": True,
                "qc_reason": "within normal range on all checked metrics",
                "metrics": {
                    "pct_counts_mito": {"value": 2.5, "z": 0.1, "flagged": False},
                    "total_counts": {"value": 8000, "z": -0.2, "flagged": False},
                },
            },
            "6": {
                "n_cells": 40,
                "qc_pass": False,
                "qc_reason": "outlier vs. sibling subclusters on: total_counts",
                "metrics": {
                    "pct_counts_mito": {"value": 5.5, "z": 3.1, "flagged": True},
                    "total_counts": {"value": 1800, "z": -3.4, "flagged": True},
                },
            },
        },
        "summary": {"n_pass": 1, "n_fail": 1, "metrics_used": ["pct_counts_mito", "total_counts"]},
    }


def test_plot_technical_qc_consistency_writes_png_and_a_vector_twin(tmp_path) -> None:
    out = plot_technical_qc_consistency(_consistency_result(), tmp_path / "qc_consistency.png")

    assert out == tmp_path / "qc_consistency.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_technical_qc_consistency_skips_when_no_clusters(tmp_path) -> None:
    empty_result = {"clusters": {}, "summary": {"n_pass": 0, "n_fail": 0, "metrics_used": []}}

    out = plot_technical_qc_consistency(empty_result, tmp_path / "qc_consistency.png")

    assert not out.exists()
