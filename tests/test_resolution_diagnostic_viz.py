"""The resolution-diagnostic figure: cluster count and bootstrap stability vs resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.clustering.resolution_diagnostic_viz import (
    write_resolution_diagnostic_figure,
)


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "resolution": [0.4, 0.6, 0.8, 1.0, 1.2],
            "n_clusters": [3, 4, 5, 7, 9],
            "median_jaccard": [0.95, 0.93, 0.88, 0.55, 0.40],
            "q25": [0.9, 0.88, 0.8, 0.4, 0.3],
            "q75": [0.98, 0.97, 0.93, 0.7, 0.5],
            "n_bootstrap_observations": [40, 40, 40, 40, 40],
        }
    )


def test_writes_a_png_and_returns_its_path(tmp_path: Path):
    output = tmp_path / "clustering_resolution_diagnostic.png"

    result = write_resolution_diagnostic_figure(_summary(), output)

    assert result == output
    assert output.exists()


def test_returns_none_and_writes_nothing_for_an_empty_summary(tmp_path: Path):
    output = tmp_path / "clustering_resolution_diagnostic.png"
    empty = pd.DataFrame(
        columns=[
            "resolution",
            "n_clusters",
            "median_jaccard",
            "q25",
            "q75",
            "n_bootstrap_observations",
        ]
    )

    result = write_resolution_diagnostic_figure(empty, output)

    assert result is None
    assert not output.exists()
