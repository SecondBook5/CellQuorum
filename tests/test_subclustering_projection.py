"""Projecting subcluster labels onto the parent leaves an object that still saves.

Regression: the projection widened the donor-gate columns from the analysed subset
to the whole object with a plain ``reindex``, which fills the gap with float NaN
whatever the column held. For the BOOLEAN gate verdict that produced ``object``
holding ``{True, False, nan}``, which h5py cannot encode — and so every h5ad write
downstream of subclustering raised ``TypeError: Can't implicitly convert
non-string objects to strings``. The run still reported success, because those
writers are skip-not-crash, and lost its final object, its checkpoints and its
velocity h5ads (and with them CellRank's velocity kernel).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellquorum.core.h5ad_io import write_h5ad
from cellquorum.stages.clustering.subclustering.stage import SubclusteringStage

_project = SubclusteringStage._project_labels


def _parent(n_obs: int = 10) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(1, (n_obs, 4)).astype(np.float32)))
    a.obsm["X_pca_harmony"] = rng.normal(size=(n_obs, 3))
    a.layers["counts"] = a.X.copy()
    return a


def test_boolean_gate_verdict_projects_to_nullable_boolean():
    parent = _parent()
    # Six of ten cells were analysed; the group filter dropped the rest.
    verdict = pd.Series([True] * 4 + [False] * 2, index=parent.obs_names[:6])

    projected = pd.Series(_project(verdict, parent.obs_names), index=parent.obs_names)

    assert projected.dtype == "boolean"
    assert projected.sum() == 4
    # The four unanalysed cells stay unknown. Filling them with False would assert
    # they failed a gate they were never put through.
    assert projected.isna().sum() == 4


def test_label_column_projects_to_a_categorical():
    parent = _parent()
    labels = pd.Series(
        np.array(["c1", "c2", "c1", "c2", "c1", "c2"], dtype=object).astype(str),
        index=parent.obs_names[:6],
    )

    projected = pd.Series(_project(labels, parent.obs_names), index=parent.obs_names)

    assert isinstance(projected.dtype, pd.CategoricalDtype)
    assert set(projected.dropna()) == {"c1", "c2"}
    assert projected.isna().sum() == 4


def test_projected_parent_object_round_trips_through_h5ad(tmp_path):
    parent = _parent()
    index = parent.obs_names[:6]
    parent.obs["lec_subcluster"] = _project(
        pd.Series(pd.Categorical(["c1", "c2"] * 3), index=index), parent.obs_names
    )
    parent.obs["donor_qc_qc_pass"] = _project(
        pd.Series([True] * 4 + [False] * 2, index=index), parent.obs_names
    )
    parent.obs["donor_qc_qc_reason"] = _project(
        pd.Series(["PASS"] * 4 + ["FAIL: n<3"] * 2, index=index), parent.obs_names
    )

    path = tmp_path / "parent.h5ad"
    # No sanitation notes: the projection produced writable dtypes itself, rather
    # than relying on the writer to repair them.
    assert write_h5ad(parent, path) == []

    back = ad.read_h5ad(path)
    assert back.obs["donor_qc_qc_pass"].sum() == 4
    assert set(back.obs["lec_subcluster"].dropna()) == {"c1", "c2"}
    assert set(back.obs["donor_qc_qc_reason"].dropna()) == {"PASS", "FAIL: n<3"}
    # And the object a resume would need is intact.
    assert "X_pca_harmony" in back.obsm
    assert "counts" in back.layers
