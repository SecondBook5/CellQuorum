"""Tests for the group_by resolution shared by hdWGCNA and pySCENIC.

Both methods independently implemented "configured OR cell_type OR leiden OR
<default>" as an inline if/elif chain; the two copies had already drifted
(one defaulted to "all", the other to "cell_type") without either author
intending a difference. One function, each caller supplies its own default.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.gene_regulation._shared import resolve_group_by


def _adata(**obs_cols: list[str]) -> ad.AnnData:
    n = len(next(iter(obs_cols.values()))) if obs_cols else 3
    obs = pd.DataFrame(obs_cols, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=np.zeros((n, 2)), obs=obs)


def test_configured_value_wins_even_when_absent_from_obs():
    """An explicit config value is trusted as-is -- resolution never overrides it."""
    a = _adata(cell_type=["A", "B", "C"])
    assert resolve_group_by(a, "custom_col", default="all") == "custom_col"


def test_falls_back_to_cell_type_when_present():
    a = _adata(cell_type=["A", "B", "C"], leiden=["0", "1", "0"])
    assert resolve_group_by(a, None, default="all") == "cell_type"


def test_falls_back_to_leiden_when_cell_type_absent():
    a = _adata(leiden=["0", "1", "0"])
    assert resolve_group_by(a, None, default="all") == "leiden"


def test_falls_back_to_caller_supplied_default_when_neither_present():
    a = _adata(other=["x", "y", "z"])
    assert resolve_group_by(a, None, default="all") == "all"
    assert resolve_group_by(a, None, default="cell_type") == "cell_type"
