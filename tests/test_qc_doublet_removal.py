"""QC honors doublets.remove: consensus doublets are dropped when configured."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.qc.config import QCConfig
from cellquorum.stages.qc.stage import QCStage


class _Paths:
    def __init__(self, tmp):
        self.results = tmp


class _Ctx:
    def __init__(self, adata, config, paths):
        self._adata = adata
        self.config = config
        self.paths = paths
        self.backend_registry = None
        self.run_id = "test"
        self.random_seed = 0

    def require_adata(self):
        return self._adata


def _counts_adata(n=60):
    rng = np.random.default_rng(0)
    x = rng.poisson(3.0, size=(n, 8)).astype(np.float32)
    a = ad.AnnData(X=x, var=pd.DataFrame(index=[f"G{i}" for i in range(8)]))
    a.layers["counts"] = x.copy()
    a.obs["sample_id"] = "S1"
    return a


@pytest.mark.parametrize("source", ["X", "chosen", "raw"])
def test_doublet_adapters_receive_selected_qc_source(source, monkeypatch, tmp_path):
    import cellquorum.stages.qc.doublets as doublets_mod

    adata = _counts_adata()
    expected = adata.X.copy()
    metrics_config = {}
    if source == "chosen":
        adata.layers["chosen"] = expected.copy()
        adata.X[:] = 999
        metrics_config = {"layer": "chosen"}
    elif source == "raw":
        adata.raw = adata.copy()
        adata.X[:] = 999
        metrics_config = {"use_raw": True}
    adata.layers["counts"][:] = 555

    def detect(work, *args, **kwargs):
        assert kwargs["random_state"] == 79
        np.testing.assert_array_equal(work.X, expected)
        assert "counts" not in work.layers
        work.obs["predicted_doublet"] = False
        work.obs["doublet_score"] = np.arange(work.n_obs, dtype=float)
        return {"n_doublets": 0}

    monkeypatch.setattr(doublets_mod, "detect_doublets", detect)
    config = QCConfig(metrics=metrics_config, doublets={"enabled": True})
    metrics = {}
    context = _Ctx(adata, {}, _Paths(tmp_path))
    context.random_seed = 79
    result = QCStage()._score_doublets(
        adata,
        qc_config=config,
        addon_metrics=metrics,
        context=context,
    )
    np.testing.assert_array_equal(result.obs["doublet_score"], np.arange(adata.n_obs))
    assert (
        metrics["doublets"]["matrix_source"]
        == {"X": "X", "chosen": "layers[chosen]", "raw": "raw.X"}[source]
    )


def test_removed_doublets_are_not_counted_as_analysable():
    from cellquorum.stages.qc.stage import _analysable_mask

    original = _counts_adata(5)
    survivors = original[2:].copy()
    keep = pd.Series(True, index=original.obs_names)
    assert _analysable_mask(original, survivors, keep).tolist() == [False, False, True, True, True]


def test_retained_probable_doublets_are_not_counted_as_analysable():
    from cellquorum.stages.qc.stage import _analysable_mask

    original = _counts_adata(5)
    output = original.copy()
    output.obs["qc_probable_multiplet"] = [True, False, False, False, False]
    keep = pd.Series(True, index=original.obs_names)
    assert _analysable_mask(original, output, keep).tolist() == [False, True, True, True, True]


def test_empty_doublet_result_fails_at_qc(monkeypatch, tmp_path):
    import cellquorum.stages.qc.doublets as doublets_mod
    from cellquorum.stages.qc.stage import QCStageError

    adata = _counts_adata(5)

    def detect(work, *args, **kwargs):
        work.obs["predicted_doublet"] = True
        return {"n_doublets": work.n_obs}

    monkeypatch.setattr(doublets_mod, "detect_doublets", detect)
    with pytest.raises(QCStageError, match="Doublet removal left no cells"):
        QCStage()._score_doublets(
            adata,
            qc_config=QCConfig(doublets={"enabled": True, "remove": True}),
            addon_metrics={},
            context=_Ctx(adata, {}, _Paths(tmp_path)),
        )


def test_doublets_removed_when_remove_true(tmp_path):
    a = _counts_adata()
    config = {
        "qc": {
            "enabled": True,
            "metrics": {"layer": "counts"},
            # The fixture is a handful of synthetic barcodes, below any real
            # detection floor. Lifting the floors isolates the doublet decision,
            # which is what this test is about.
            "floors": {"min_genes_per_cell": None, "min_cells_per_gene": None},
            "doublets": {
                "enabled": True,
                "method": "scrublet",
                "methods": ["scrublet"],
                "consensus": "any",
                "remove": True,
            },
            "ambient": {"enabled": False, "method": "none"},
        }
    }
    ctx = _Ctx(a, config, _Paths(tmp_path))
    # Force a deterministic predicted_doublet column by pre-seeding it: the
    # remover must act on whatever detect_doublets leaves in obs. We simulate a
    # detector result by monkeypatching detect_doublets to flag the first 5 cells.
    import cellquorum.stages.qc.doublets as doublets_mod

    def _fake_detect(adata, cfg, backend, sample_key=None, n_jobs=1, random_state=0):
        flags = np.zeros(adata.n_obs, dtype=bool)
        flags[:5] = True
        adata.obs["predicted_doublet"] = flags
        adata.obs["doublet_score"] = np.linspace(0, 1, adata.n_obs)
        return {"n_doublets": 5, "method": "scrublet"}

    orig = doublets_mod.detect_doublets
    try:
        # detect_doublets is imported lazily inside the stage from the module,
        # so patch the source module attribute.
        doublets_mod.detect_doublets = _fake_detect
        result = QCStage().run(ctx)
    finally:
        doublets_mod.detect_doublets = orig

    assert result.adata.n_obs == 55
    assert bool(result.adata.obs.get("predicted_doublet", pd.Series([], dtype=bool)).any()) is False
    assert result.metrics["doublets"]["n_removed"] == 5


def test_doublets_kept_when_remove_false(tmp_path):
    a = _counts_adata()
    config = {
        "qc": {
            "enabled": True,
            "metrics": {"layer": "counts"},
            # The fixture is a handful of synthetic barcodes, below any real
            # detection floor. Lifting the floors isolates the doublet decision,
            # which is what this test is about.
            "floors": {"min_genes_per_cell": None, "min_cells_per_gene": None},
            "doublets": {
                "enabled": True,
                "method": "scrublet",
                "methods": ["scrublet"],
                "consensus": "any",
                "remove": False,
            },
            "ambient": {"enabled": False, "method": "none"},
        }
    }
    ctx = _Ctx(a, config, _Paths(tmp_path))
    import cellquorum.stages.qc.doublets as doublets_mod

    def _fake_detect(adata, cfg, backend, sample_key=None, n_jobs=1, random_state=0):
        flags = np.zeros(adata.n_obs, dtype=bool)
        flags[:5] = True
        adata.obs["predicted_doublet"] = flags
        adata.obs["doublet_score"] = np.linspace(0, 1, adata.n_obs)
        return {"n_doublets": 5, "method": "scrublet"}

    orig = doublets_mod.detect_doublets
    try:
        doublets_mod.detect_doublets = _fake_detect
        result = QCStage().run(ctx)
    finally:
        doublets_mod.detect_doublets = orig

    # remove=False: all cells retained, flag preserved.
    assert result.adata.n_obs == 60
    assert int(result.adata.obs["predicted_doublet"].sum()) == 5
    called = result.adata.obs["predicted_doublet"]
    assert result.adata.obs.loc[called, "qc_probable_multiplet"].all()
    assert not result.adata.obs.loc[called, "qc_fit_manifold"].any()
    assert not result.adata.obs.loc[called, "qc_fit_clustering"].any()
