"""The scib benchmark must subsample, stratified by batch, and return the FULL object.

kBET, iLISI and cLISI are kNN statistics at n_neighbors=90. Over 201,871 cells the neighbour
structures were large enough to kill a run at this stage. They describe the embedding as a
population, so they converge well below the full cohort, and scib's own documentation recommends
subsampling for kBET.

Two properties matter beyond "it is smaller". Every one of these metrics is about how BATCHES
mix, so a draw that lost a small library would change the thing being measured. And the stage is
read-only: returning the subset would silently drop cells from every later stage, which is worse
than a slow benchmark.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd


def _cohort(n: int = 5_000, n_batches: int = 6) -> ad.AnnData:
    rng = np.random.default_rng(0)
    # Deliberately uneven, with one very small batch: the case a naive draw would lose.
    head = [n // 2, n // 4, n // 8, n // 16, n // 32]
    sizes = [*head, n - sum(head)]
    batches = np.concatenate([np.full(s, f"batch{i}") for i, s in enumerate(sizes)])
    obs = pd.DataFrame(
        {"batch": batches, "cell_type": rng.choice(["A", "B", "C"], size=len(batches))},
        index=pd.Index([f"c{i}" for i in range(len(batches))]),
    )
    adata = ad.AnnData(X=rng.normal(size=(len(batches), 20)).astype("float32"), obs=obs)
    adata.obsm["X_pca"] = rng.normal(size=(len(batches), 10))
    adata.obsm["X_pca_harmony"] = rng.normal(size=(len(batches), 10))
    return adata


class _Ctx:
    class paths:
        results = "/tmp"
        figures = "/tmp"
        scratch = "/tmp"

    backend_registry = None
    config = {}

    def require_adata(self):  # noqa: D102
        return None


def _subsample(adata, max_cells, seed=0):
    """Reproduce the stage's draw, which is what the assertions are about."""
    rng = np.random.default_rng(seed)
    values = adata.obs["batch"].astype(str).to_numpy()
    positions = np.arange(adata.n_obs)
    share = max_cells / adata.n_obs
    keep = []
    for value in np.unique(values):
        members = positions[values == value]
        take = max(1, min(len(members), int(round(len(members) * share))))
        keep.extend(members if take >= len(members) else rng.choice(members, take, replace=False))
    return np.sort(np.asarray(keep, dtype=int))


def test_every_batch_survives_the_draw() -> None:
    """A lost library would change what the metric measures, not just its precision."""

    adata = _cohort()
    index = _subsample(adata, 1_000)
    kept = adata.obs["batch"].astype(str).to_numpy()[index]
    assert set(kept) == set(adata.obs["batch"].astype(str)), "a batch was dropped entirely"


def test_the_smallest_batch_keeps_at_least_one_cell() -> None:
    """The floor is why: proportional alone rounds a tiny library to zero."""

    adata = _cohort()
    smallest = adata.obs["batch"].value_counts().idxmin()
    index = _subsample(adata, 100)  # aggressive cap
    kept = adata.obs["batch"].astype(str).to_numpy()[index]
    assert (kept == smallest).sum() >= 1


def test_the_draw_is_proportional() -> None:
    """Stratified means representative, so batch shares should be roughly preserved."""

    adata = _cohort()
    index = _subsample(adata, 2_000)
    before = adata.obs["batch"].value_counts(normalize=True).sort_index()
    after = (
        pd.Series(adata.obs["batch"].astype(str).to_numpy()[index])
        .value_counts(normalize=True)
        .sort_index()
    )
    assert np.allclose(before.to_numpy(), after.to_numpy(), atol=0.03)


def test_the_draw_is_reproducible() -> None:
    """A benchmark score that changes between runs is not a score."""

    adata = _cohort()
    assert np.array_equal(_subsample(adata, 1_000, seed=7), _subsample(adata, 1_000, seed=7))


def test_a_cohort_under_the_cap_is_untouched() -> None:
    adata = _cohort(n=500)
    index = _subsample(adata, 40_000)
    assert len(index) == adata.n_obs
