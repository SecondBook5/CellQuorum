"""QC MAD groupby columns are carried from obs onto the cell-metric table.

Regression: per-sample MAD (mad.groupby: [sample_id]) failed with
"cell_metrics is missing required column(s): sample_id" because
calculate_qc_metrics built cell_metrics from the matrix only and never copied
the grouping column from obs.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.qc.config import QCConfig
from cellquorum.qc.metrics import calculate_qc_metrics


def _counts_adata(n=40):
    rng = np.random.default_rng(0)
    x = rng.poisson(3.0, size=(n, 6)).astype(np.float32)
    a = ad.AnnData(X=x, var=pd.DataFrame(index=[f"G{i}" for i in range(6)]))
    a.layers["counts"] = x.copy()
    a.obs_names = [f"c{i}" for i in range(n)]
    # Two samples, so a groupby has real groups.
    a.obs["sample_id"] = ["S1"] * (n // 2) + ["S2"] * (n - n // 2)
    return a


def test_groupby_column_attached_to_cell_metrics():
    a = _counts_adata()
    cfg = QCConfig.model_validate(
        {
            "metrics": {"layer": "counts"},
            "mad": {"enabled": True, "groupby": ["sample_id"]},
        }
    )
    result = calculate_qc_metrics(a, cfg)
    # The grouping column must be present in the metric table for group-wise MAD.
    assert "sample_id" in result.cell_metrics.columns
    # And aligned to the obs it came from.
    assert list(result.cell_metrics["sample_id"]) == list(a.obs["sample_id"])


def test_no_groupby_leaves_metrics_untouched():
    a = _counts_adata()
    cfg = QCConfig.model_validate(
        {"metrics": {"layer": "counts"}, "mad": {"enabled": True, "groupby": []}}
    )
    result = calculate_qc_metrics(a, cfg)
    assert "sample_id" not in result.cell_metrics.columns


def test_missing_groupby_column_fails_loud():
    # A configured groupby column absent from obs must fail loud (not silently
    # collapse to a single global threshold). Input validation catches it first.
    from cellquorum.qc.validation import QCInputValidationError

    a = _counts_adata()
    del a.obs["sample_id"]
    cfg = QCConfig.model_validate(
        {
            "metrics": {"layer": "counts"},
            "mad": {"enabled": True, "groupby": ["sample_id"]},
        }
    )
    with pytest.raises(QCInputValidationError):
        calculate_qc_metrics(a, cfg)
