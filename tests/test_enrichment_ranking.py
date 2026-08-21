from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.comparative.enrichment.ranking import de_table_to_ranking


def test_signed_neglog10p_metric():
    df = pd.DataFrame(
        {"gene": ["A", "B", "C"], "logFC": [2.0, -1.0, 0.5], "PValue": [0.01, 0.001, 0.1]}
    )
    r = de_table_to_ranking(df)
    assert list(r.index) == ["contrast"]
    # A: +1 * -log10(0.01)=2 ; B: -1 * -log10(0.001)=-3 ; C: +1 * -log10(0.1)=1
    assert r.loc["contrast", "A"] == 2.0
    assert r.loc["contrast", "B"] == -3.0
    assert np.isclose(r.loc["contrast", "C"], 1.0)


def test_duplicate_gene_collapsed_by_max_abs():
    df = pd.DataFrame({"gene": ["A", "A"], "logFC": [1.0, -1.0], "PValue": [0.5, 0.001]})
    r = de_table_to_ranking(df)
    # second row has larger |metric| (|-3| > |0.3|) -> kept
    assert r.loc["contrast", "A"] == -3.0


def test_nan_and_zero_pvalue_dropped():
    df = pd.DataFrame(
        {"gene": ["A", "B", "C"], "logFC": [1.0, np.nan, 1.0], "PValue": [0.0, 0.01, np.nan]}
    )
    r = de_table_to_ranking(df)
    # A has p=0 -> -log10(0)=inf -> dropped; B has nan logFC -> dropped; C nan p -> dropped
    assert "A" not in r.columns and "B" not in r.columns and "C" not in r.columns
