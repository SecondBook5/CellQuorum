"""Pseudobulk PCA of samples: one point per sample, variance per axis, donor pairs joined."""

from __future__ import annotations

import re

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cellquorum.visualization.pseudobulk_pca import pseudobulk_pca  # noqa: E402


def _data(n_donors: int = 5, n_genes: int = 60):
    rng = np.random.default_rng(0)
    rows, meta = [], []
    for donor in range(n_donors):
        for cond in ("Normal", "Lymphedema"):
            shift = 3.0 if cond == "Lymphedema" else 0.0
            rows.append(
                rng.normal(0, 0.2, n_genes) + shift * np.r_[np.ones(10), np.zeros(n_genes - 10)]
            )
            meta.append(
                {
                    "sample": f"P{donor}_{cond}",
                    "donor": f"P{donor}",
                    "condition": cond,
                    "state": "A",
                }
            )
    meta = pd.DataFrame(meta).set_index("sample")
    return pd.DataFrame(rows, index=meta.index, columns=[f"g{i}" for i in range(n_genes)]), meta


@pytest.fixture(autouse=True)
def _close():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _draw(expression, meta, **kwargs):
    base = dict(condition_col="condition", group_col="state", pair_col="donor", n_top=40)
    base.update(kwargs)
    return pseudobulk_pca(expression, meta, **base)


def _points(fig) -> int:
    return sum(len(c.get_offsets()) for c in fig.axes[0].collections if c.get_gid() == "samples")


def test_sample_pca_shows_all_samples_and_variance_from_variable_genes() -> None:
    expression, meta = _data()
    padded = expression.assign(**{f"flat{i}": 5.0 for i in range(200)})
    fig = _draw(expression, meta)

    assert _points(fig) == len(meta)

    values = expression.to_numpy()
    top = values[:, np.argsort(values.var(axis=0))[::-1][:40]]  # the n_top genes the PCA uses
    s = np.linalg.svd(top - top.mean(axis=0), compute_uv=False)
    expected = 100 * s[0] ** 2 / (s**2).sum()
    shown = float(re.search(r"PC1 \((\d+)%\)", fig.axes[0].get_xlabel()).group(1))
    assert abs(shown - expected) <= 0.51

    a = fig.axes[0].get_xlabel()
    b = _draw(padded, meta).axes[0].get_xlabel()
    assert a == b


def test_donor_pairs_are_joined_only_when_both_conditions_exist() -> None:
    expression, meta = _data()
    meta = meta.drop(index="P0_Lymphedema")
    expression = expression.loc[meta.index]
    fig = _draw(expression, meta)
    joined = [ln for ln in fig.axes[0].lines if ln.get_gid() == "pair"]
    assert len(joined) == 4


def test_fewer_than_three_samples_is_an_error() -> None:
    expression, meta = _data(n_donors=1)
    with pytest.raises(ValueError, match="at least three"):
        _draw(expression, meta)


def _legend_labels(fig):
    return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]


def test_legend_follows_the_order_of_the_colours_and_states_asked_for() -> None:
    """Real ask: every figure puts Normal first and lymphedema second; the legend listed the levels
    in the order the samples happened to appear (Lymphedema first on the LEC pseudobulk)."""
    expression, meta = _data()
    meta = meta.iloc[::-1]  # lymphedema samples first, as in a table sorted by sample name
    meta["state"] = ["B", "A"] * (len(meta) // 2)
    expression = expression.loc[meta.index]
    fig = _draw(
        expression,
        meta,
        condition_colors={"Normal": "#24608F", "Lymphedema": "#C41E3A"},
        group_order=["A", "B"],
    )
    labels = _legend_labels(fig)
    assert labels.index("A") < labels.index("B")
    assert labels.index("Normal") < labels.index("Lymphedema")
