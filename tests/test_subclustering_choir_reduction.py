"""CHOIR must receive a batch-corrected embedding, not cluster uncorrected data.

Why this is a test and not a comment: CHOIR's own docs state that batch correction
is what "ensures that clusters do not originate from a single batch". CHOIR 0.3.0
cannot batch-correct itself against harmony >= 1.0 (it calls the removed
`HarmonyMatrix()`), so the ONLY way the stage produces a trustworthy cluster count
is by passing a precomputed corrected embedding through CHOIR's `reduction`
argument. On the real LEC subset, omitting it yielded a cluster that was 98% a
single donor — certified significant by the permutation test.

The wiring has three parts, each covered here: the stage re-embeds batch-aware
BEFORE partitioning, `run_choir` forwards the embedding plus the
`highly_variable` flag CHOIR requires alongside it, and a missing embedding
degrades to CHOIR's own reduction rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.clustering.subclustering import partition
from cellquorum.stages.clustering.subclustering.config import SubclusteringConfig
from cellquorum.stages.clustering.subclustering.extract import reembed_focus_batch_aware


def _focus_subset(n_cells: int = 120, n_genes: int = 60, n_batches: int = 3):
    """Synthetic counts object with a batch column and a counts layer."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(n_cells, n_genes)).astype("float32")
    adata = ad.AnnData(X=counts.copy())
    adata.layers["counts"] = counts.copy()
    adata.obs_names = [f"cell{i}" for i in range(n_cells)]
    adata.var_names = [f"gene{j}" for j in range(n_genes)]
    adata.obs["donor_id"] = [f"D{i % n_batches}" for i in range(n_cells)]
    return adata


@dataclass
class _CapturingBackend:
    """Stands in for the Rscript backend; records args instead of running R."""

    captured: list = None
    returncode: int = 0

    def __post_init__(self):
        self.captured = []

    def _r_package_available(self, name: str) -> bool:
        return True

    def run_script(self, script, args, timeout=None):
        self.captured.append((Path(script).name, list(args)))

        # Emulate choir.R's contract: write a barcode,subcluster CSV.
        out_csv = Path(args[1])
        in_h5ad = Path(args[0])
        written = ad.read_h5ad(in_h5ad)
        labels = ["1" if i % 2 == 0 else "2" for i in range(written.n_obs)]
        pd.DataFrame({"barcode": written.obs_names, "subcluster": labels}).to_csv(
            out_csv, index=False
        )

        @dataclass
        class _Result:
            returncode: int
            stdout: str = ""
            stderr: str = ""

        return _Result(returncode=self.returncode)


def test_reembed_focus_batch_aware_writes_embedding_and_hvg_flag():
    adata = _focus_subset()
    key = reembed_focus_batch_aware(
        adata, counts_layer="counts", batch_key="donor_id", n_top_genes=40, n_comps=10
    )
    assert key is not None
    assert key in adata.obsm
    assert adata.obsm[key].shape[0] == adata.n_obs
    # CHOIR needs var_features alongside a supplied reduction.
    assert "highly_variable" in adata.var
    assert int(adata.var["highly_variable"].sum()) >= 2
    # The caller's matrices must be untouched (normalisation happens on copies).
    assert np.allclose(np.asarray(adata.X), np.asarray(adata.layers["counts"]))


def test_reembed_reports_harmony_non_convergence_to_its_caller():
    """CHOIR's permutation test is only as meaningful as the space it runs in.

    The re-embedding hands CHOIR a Harmony-corrected reduction, and CHOIR then
    declares a cluster count significant *in that space*. Harmony's own default cap
    is 10 iterations and harmonypy announces falling short of convergence at INFO —
    which this code path silences. Both lec_mechanotransduction arms hit that cap, so
    the sink is how the stage gets to say so.
    """
    adata = _focus_subset()
    sink: list[str] = []
    key = reembed_focus_batch_aware(
        adata,
        counts_layer="counts",
        batch_key="donor_id",
        n_top_genes=40,
        n_comps=10,
        max_iter_harmony=1,
        diagnostics=sink,
    )
    assert key is not None
    joined = " ".join(sink)
    assert "converge" in joined, sink
    assert "max_iter_harmony=1" in joined


def test_reembed_is_quiet_when_harmony_converges():
    adata = _focus_subset()
    sink: list[str] = []
    reembed_focus_batch_aware(
        adata,
        counts_layer="counts",
        batch_key="donor_id",
        n_top_genes=40,
        n_comps=10,
        max_iter_harmony=200,
        diagnostics=sink,
    )
    assert sink == []


def test_reembed_single_batch_falls_back_to_plain_pca_key():
    # One batch: nothing to correct, so the uncorrected key is the honest label.
    adata = _focus_subset(n_batches=1)
    key = reembed_focus_batch_aware(
        adata, counts_layer="counts", batch_key="donor_id", n_top_genes=40, n_comps=10
    )
    assert key == "X_pca"


def test_reembed_returns_none_for_degenerate_subset():
    adata = _focus_subset(n_cells=2, n_genes=2)
    assert (
        reembed_focus_batch_aware(adata, counts_layer="counts", batch_key="donor_id", n_comps=10)
        is None
    )


@pytest.fixture
def _config():
    config = SubclusteringConfig()
    config.donor_gate.group_key = "donor_id"
    return config


def test_run_choir_forwards_reduction_key_as_ninth_arg(tmp_path, _config, monkeypatch):
    monkeypatch.setattr(partition.shutil, "which", lambda name: "/usr/bin/Rscript")
    adata = _focus_subset()
    key = reembed_focus_batch_aware(
        adata, counts_layer="counts", batch_key="donor_id", n_top_genes=40, n_comps=10
    )
    backend = _CapturingBackend()

    result = partition.run_choir(adata, _config, backend, tmp_path, reduction_key=key)
    assert not isinstance(result, MethodSkip)

    script_name, args = backend.captured[0]
    assert script_name == "choir.R"
    # 9 positional args, the last naming the embedding.
    assert len(args) == 9
    assert args[8] == key

    # The written input must carry BOTH the embedding and the HVG flag, or
    # choir.R rejects the reduction ("var_features cannot be NULL").
    written = ad.read_h5ad(Path(args[0]))
    assert key in written.obsm
    assert "highly_variable" in written.var
    assert written.var["highly_variable"].dtype == bool


def test_run_choir_passes_none_when_no_embedding_available(tmp_path, _config, monkeypatch):
    # No reduction and no HVG flag: CHOIR computes its own reduction. This must
    # degrade quietly rather than send a bogus key that choir.R would reject.
    monkeypatch.setattr(partition.shutil, "which", lambda name: "/usr/bin/Rscript")
    adata = _focus_subset()
    backend = _CapturingBackend()

    partition.run_choir(adata, _config, backend, tmp_path, reduction_key=None)
    _, args = backend.captured[0]
    assert args[8] == "NONE"


def test_run_choir_passes_none_when_hvg_flag_missing(tmp_path, _config, monkeypatch):
    # An embedding without var_features is unusable by choir.R; the guard must
    # catch that here rather than letting the R script fail the whole stage.
    monkeypatch.setattr(partition.shutil, "which", lambda name: "/usr/bin/Rscript")
    adata = _focus_subset()
    adata.obsm["X_pca_harmony"] = np.zeros((adata.n_obs, 5), dtype="float32")
    backend = _CapturingBackend()

    partition.run_choir(adata, _config, backend, tmp_path, reduction_key="X_pca_harmony")
    _, args = backend.captured[0]
    assert args[8] == "NONE"


def test_run_choir_writes_plain_string_index(tmp_path, _config, monkeypatch):
    # A pandas nullable-string index (which real run objects carry) is refused by
    # the h5ad writer and misread by zellkonverter as a categorical.
    monkeypatch.setattr(partition.shutil, "which", lambda name: "/usr/bin/Rscript")
    adata = _focus_subset()
    adata.obs_names = pd.array([f"cell{i}" for i in range(adata.n_obs)], dtype="string")
    backend = _CapturingBackend()

    result = partition.run_choir(adata, _config, backend, tmp_path)
    assert not isinstance(result, MethodSkip)
    written = ad.read_h5ad(Path(backend.captured[0][1][0]))
    assert written.obs_names.dtype == object
