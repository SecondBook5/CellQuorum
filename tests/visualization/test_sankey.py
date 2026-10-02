"""Tests for the two-column categorical-flow Sankey diagram."""

from __future__ import annotations

import pandas as pd
import pytest

from cellquorum.visualization.sankey import two_column_sankey


def _crosstab() -> pd.DataFrame:
    # rows = left (automated), columns = right (manual). 3 nonzero cells.
    return pd.DataFrame(
        {
            "Fibroblasts": [90, 0],
            "DC": [10, 40],
        },
        index=["Fibroblasts", "DC"],
    )


def test_two_column_sankey_writes_png_and_a_vector_twin(tmp_path):
    out = two_column_sankey(
        _crosstab(),
        left_title="Automated",
        right_title="Manual",
        out_path=tmp_path / "sankey.png",
    )

    assert out == tmp_path / "sankey.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_two_column_sankey_draws_one_ribbon_per_nonzero_flow(tmp_path, monkeypatch):
    from cellquorum.visualization import sankey as sankey_module

    captured = {}

    def fake_save(fig, out_dir, stem, **kwargs):
        captured["fig"] = fig
        return [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]

    monkeypatch.setattr(sankey_module, "save_figure", fake_save)

    ct = _crosstab()
    # 3 nonzero cells: (Fibroblasts,Fibroblasts)=90, (Fibroblasts,DC)=10, (DC,DC)=40.
    two_column_sankey(ct, left_title="Automated", right_title="Manual", out_path=tmp_path / "s.png")

    ax = captured["fig"].axes[0]
    # 2 left node rects + 2 right node rects + 3 ribbon patches = 7.
    assert len(ax.patches) == 7


def test_two_column_sankey_skips_zero_flows(tmp_path, monkeypatch):
    """A crosstab cell of 0 (Fibroblasts never called DC-by-manual here) must not
    draw a zero-width ribbon."""
    from cellquorum.visualization import sankey as sankey_module

    captured = {}

    def fake_save(fig, out_dir, stem, **kwargs):
        captured["fig"] = fig
        return [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]

    monkeypatch.setattr(sankey_module, "save_figure", fake_save)

    ct = pd.DataFrame({"A": [5, 0], "B": [0, 7]}, index=["A", "B"])
    two_column_sankey(ct, left_title="L", right_title="R", out_path=tmp_path / "s.png")

    ax = captured["fig"].axes[0]
    # 2 left + 2 right node rects + 2 nonzero ribbons (the two zero cells excluded) = 6.
    assert len(ax.patches) == 6


def test_two_column_sankey_raises_on_an_all_zero_crosstab(tmp_path):
    ct = pd.DataFrame({"A": [0, 0]}, index=["A", "B"])
    with pytest.raises(ValueError, match="no flow"):
        two_column_sankey(ct, left_title="L", right_title="R", out_path=tmp_path / "s.png")
