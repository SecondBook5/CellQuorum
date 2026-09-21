"""Tests for the per-gene marker-trace figures (on-demand, config-gated)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellquorum.stages.metacells.viz.marker_traces import plot_metacell_marker_traces


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metacell_id": ["MC1", "MC2", "MC3"],
            "dominant_group": ["Fibroblast", "Endothelial", "Fibroblast"],
            "n_cells": [80, 95, 60],
        }
    )


def _expr() -> pd.DataFrame:
    return pd.DataFrame(
        {"MC1": [1.0, 2.0], "MC2": [0.1, 0.2], "MC3": [1.5, 1.8]},
        index=["PECAM1", "PDGFRB"],
    )


def test_no_genes_configured_returns_empty(tmp_path: Path) -> None:
    assert plot_metacell_marker_traces(_expr(), _nodes(), tmp_path, []) == []


def test_one_file_pair_per_configured_gene(tmp_path: Path) -> None:
    paths = plot_metacell_marker_traces(_expr(), _nodes(), tmp_path, ["PECAM1", "PDGFRB"])
    assert len(paths) == 4  # 2 genes x (png + pdf)
    stems = {p.stem for p in paths}
    assert stems == {"PECAM1", "PDGFRB"}


def test_gene_not_in_expr_is_skipped_not_crashed(tmp_path: Path) -> None:
    paths = plot_metacell_marker_traces(_expr(), _nodes(), tmp_path, ["PECAM1", "NOT_A_REAL_GENE"])
    assert len(paths) == 2  # only PECAM1
