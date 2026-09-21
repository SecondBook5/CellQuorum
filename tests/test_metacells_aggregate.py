"""Tests for pure metacell aggregation math (no AnnData, no subprocess)."""

from __future__ import annotations

import pandas as pd
import pytest

from cellquorum.stages.metacells.aggregate import (
    dominant_group,
    median_size_per_metacell,
    outlier_summary,
    purity_fraction,
)


def _assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(8)],
            "metacell_id": ["MC1", "MC1", "MC1", "MC2", "MC2", "MC2", "Outliers", "Outliers"],
            "group": ["A", "A", "B", "C", "C", "C", "A", "B"],
            "n_umis": [1000, 1200, 900, 500, 600, 550, 200, 150],
        }
    )


def test_dominant_group_picks_the_majority_label() -> None:
    dom = dominant_group(_assignments())
    assert dom["MC1"] == "A"  # 2 of 3
    assert dom["MC2"] == "C"  # 3 of 3


def test_purity_fraction_matches_majority_share() -> None:
    pur = purity_fraction(_assignments())
    assert pur["MC1"] == pytest.approx(2 / 3)
    assert pur["MC2"] == pytest.approx(1.0)


def test_median_size_per_metacell() -> None:
    sizes = median_size_per_metacell(_assignments())
    assert sizes["MC1"] == 1000.0
    assert sizes["MC2"] == 550.0


def test_outlier_summary_counts_the_outliers_bucket() -> None:
    summary = outlier_summary(_assignments())
    assert summary["n_outlier_cells"] == 2
    assert summary["pct_outlier"] == pytest.approx(2 / 8 * 100)
    assert summary["n_metacells"] == 2  # MC1, MC2 -- Outliers is not a real metacell


def test_outlier_summary_with_no_outliers() -> None:
    df = _assignments()
    df = df[df["metacell_id"] != "Outliers"]
    summary = outlier_summary(df)
    assert summary["n_outlier_cells"] == 0
    assert summary["pct_outlier"] == 0.0
    assert summary["n_metacells"] == 2
