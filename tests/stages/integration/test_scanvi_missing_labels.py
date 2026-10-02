"""scANVI must treat missing labels as unlabeled, not as a literal "nan" cell type.

The natural way to represent "this cell has no annotation yet" in a pandas/AnnData obs
column is NaN, not the literal string configured in `unlabeled_category`. Building
`_scanvi_labels` via a bare `.astype(str)` turns NaN into the string "nan" -- a bogus,
distinct category that scANVI would then try to learn a real embedding cluster for,
instead of recognizing those cells as the unlabeled ones it is semi-supervised to predict.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.integration.scanvi_methods import ScANVIMethod


class _Registry:
    def available(self, name: str) -> bool:
        return name == "gpu"


class _Context:
    backend_registry = _Registry()


def _cohort_with_missing_labels(n: int = 60, n_genes: int = 40, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=(n, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=counts.copy())
    adata.layers["counts"] = counts
    adata.obs["patient_id"] = ["a"] * (n // 2) + ["b"] * (n // 2)
    # A third of cells have no annotation yet -- NaN, the natural pandas representation.
    cell_type = ["t1"] * 20 + ["t2"] * 20 + [np.nan] * 20
    adata.obs["cell_type"] = pd.array(cell_type, dtype="object")
    return adata


class _CapturingSCVI:
    captured_labels: list[list[str]] = []

    def __init__(self, adata, n_latent, **kwargs):
        pass

    @staticmethod
    def setup_anndata(adata, batch_key=None, **kwargs):
        _CapturingSCVI.captured_labels.append(adata.obs["_scanvi_labels"].tolist())
        return None

    def train(self, max_epochs=None, **kwargs):
        return None


class _StubSCANVI:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_scvi_model(cls, vae, unlabeled_category=None, labels_key=None, **kwargs):
        return cls()

    def train(self, max_epochs=None, **kwargs):
        return None

    def get_latent_representation(self, adata=None):
        n = adata.n_obs if adata is not None else 60
        rng = np.random.default_rng(1)
        return rng.normal(size=(n, 10)).astype(np.float32)


def test_missing_labels_become_the_configured_unlabeled_category_not_the_string_nan(
    monkeypatch,
):
    import scvi

    _CapturingSCVI.captured_labels = []
    monkeypatch.setattr(scvi.model, "SCVI", _CapturingSCVI)
    monkeypatch.setattr(scvi.model, "SCANVI", _StubSCANVI)

    adata = _cohort_with_missing_labels()

    ScANVIMethod()._run(
        adata,
        {"batch_key": "patient_id", "label_key": "cell_type", "n_latent": 10},
        context=_Context(),
    )

    labels = _CapturingSCVI.captured_labels[0]
    assert "nan" not in labels, labels
    # The 20 originally-NaN cells (last third) must carry the unlabeled sentinel.
    assert labels[40:] == ["Unknown"] * 20
    assert labels[:20] == ["t1"] * 20
    assert labels[20:40] == ["t2"] * 20


def test_missing_labels_work_when_the_column_is_a_pandas_categorical(monkeypatch):
    """Cell-type obs columns are routinely Categorical after an h5ad round-trip.

    Assigning "Unknown" straight into a Categorical whose categories don't include it
    raises `TypeError: Cannot setitem on a Categorical with a new category` -- this is
    the more realistic path than a plain object-dtype column.
    """
    import scvi

    _CapturingSCVI.captured_labels = []
    monkeypatch.setattr(scvi.model, "SCVI", _CapturingSCVI)
    monkeypatch.setattr(scvi.model, "SCANVI", _StubSCANVI)

    adata = _cohort_with_missing_labels()
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    ScANVIMethod()._run(
        adata,
        {"batch_key": "patient_id", "label_key": "cell_type", "n_latent": 10},
        context=_Context(),
    )

    labels = _CapturingSCVI.captured_labels[0]
    assert "nan" not in labels, labels
    assert labels[40:] == ["Unknown"] * 20
