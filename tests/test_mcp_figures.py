"""Smoke test for the MCP summary figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from cellquorum.comparative.multicellular_programs.mcp_figures import plot_mcp_summary


def test_plot_mcp_summary_writes_figure(tmp_path):
    programs = pd.DataFrame(
        {
            "program": ["MCP1", "MCP1", "MCP2"],
            "cell_type": ["A", "B", "A"],
            "gene": ["G1", "G2", "G3"],
            "loading": [0.8, 0.5, -0.4],
            "direction": ["up", "up", "down"],
        }
    )
    scores = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(6)],
            "sample": ["s0", "s1"] * 3,
            "cell_type": ["A", "B"] * 3,
            "program": ["MCP1"] * 3 + ["MCP2"] * 3,
            "score": [1.0, -1.0, 0.5, 0.2, -0.3, 0.1],
        }
    )
    ds = pd.DataFrame(
        {
            "program": ["MCP1", "MCP2"],
            "n_donors": [3, 1],
            "donor_fraction": [1.0, 0.33],
            "supported": [True, False],
        }
    )
    out = plot_mcp_summary(
        programs, scores, ds, cell_type_col_values=["A", "B"], out_dir=tmp_path, name="mcp_summary"
    )
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0
