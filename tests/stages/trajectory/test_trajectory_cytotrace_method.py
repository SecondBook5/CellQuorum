"""CytoTraceMethod orchestration tests (heavy run mocked)."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.stage import StageResult
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.trajectory import _cytotrace
from cellquorum.stages.trajectory.cytotrace_method import CytoTraceMethod


def _adata(n=40, g=50):
    rng = np.random.default_rng(1)
    a = ad.AnnData(rng.poisson(2, size=(n, g)).astype("float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(g)]
    return a


def _ctx(a, tmp_path):
    config = types.SimpleNamespace()
    paths = types.SimpleNamespace(results=str(tmp_path))
    return types.SimpleNamespace(
        require_adata=lambda: a, config=config, paths=paths, donor_col=None
    )


def _fake_results(obs_names):
    """A CytoTRACE 2 results frame like the real entrypoint returns."""
    n = len(obs_names)
    return pd.DataFrame(
        {
            "CytoTRACE2_Score": np.linspace(0, 1, n),
            "CytoTRACE2_Potency": ["Differentiated"] * n,
            "CytoTRACE2_Relative": np.linspace(0, 1, n),
        },
        index=list(obs_names),
    )


def test_cytotrace_writes_score(tmp_path, monkeypatch):
    a = _adata()
    monkeypatch.setattr(_cytotrace, "run_cytotrace2", lambda *x, **k: _fake_results(a.obs_names))
    res = CytoTraceMethod()._run(a, {"species": "human", "seed": 14}, _ctx(a, tmp_path))
    assert isinstance(res, StageResult)
    assert "cytotrace2_score" in res.adata.obs
    assert "cytotrace2_potency" in res.adata.obs
    assert res.adata.uns["trajectory"]["cytotrace"]["n_cells_scored"] == a.n_obs


def test_cytotrace_skips_when_unavailable(tmp_path, monkeypatch):
    a = _adata()

    def _boom(*x, **k):
        raise _cytotrace.CytoTraceUnavailable("cytotrace2-py unavailable")

    monkeypatch.setattr(_cytotrace, "run_cytotrace2", _boom)
    res = CytoTraceMethod()._run(a, {"species": "human"}, _ctx(a, tmp_path))
    assert isinstance(res, MethodSkip)


def test_cytotrace_skips_on_empty_counts(tmp_path):
    a = ad.AnnData(np.zeros((0, 0), dtype="float32"))
    res = CytoTraceMethod()._run(a, {"species": "human"}, _ctx(a, tmp_path))
    assert isinstance(res, MethodSkip)


def test_cytotrace_skips_when_no_cells_align(tmp_path, monkeypatch):
    a = _adata()
    # Results indexed by cell names that do not exist in adata → nothing aligns.
    monkeypatch.setattr(
        _cytotrace,
        "run_cytotrace2",
        lambda *x, **k: _fake_results([f"other{i}" for i in range(a.n_obs)]),
    )
    res = CytoTraceMethod()._run(a, {"species": "human"}, _ctx(a, tmp_path))
    assert isinstance(res, MethodSkip)


def test_external_worker_exchange_preserves_cell_ids_and_requires_fresh_scores(
    tmp_path, monkeypatch
):
    import json
    import subprocess

    import pytest
    from scipy import sparse

    names = ["001", "NA"]
    counts = sparse.csr_matrix([[1, 0], [0, 3]])
    args = dict(
        species="mouse",
        workdir=tmp_path,
        seed=42,
        disable_parallelization=True,
        batch_size=20000,
        smooth_batch_size=1000,
    )
    monkeypatch.setenv("CELLQUORUM_CYTOTRACE2_PYTHON", "/model/python")

    def score(command, *, cwd, **kwargs):
        assert command[:2] == ["/model/python", "-I"]
        assert (sparse.load_npz(cwd / "counts.npz") != counts).nnz == 0
        genes, barcodes, settings = json.loads((cwd / "input.json").read_text())
        assert genes == ["A", "B"] and barcodes == names and settings["seed"] == 42
        output = cwd / "cytotrace2_results"
        output.mkdir()
        _fake_results(names[::-1]).to_csv(output / "cytotrace2_results.txt", sep="\t")
        (cwd / "cytotrace2-runtime.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", score)
    result = _cytotrace.run_cytotrace2(counts, ["A", "B"], names, **args)
    assert result.index.tolist() == names[::-1]
    assert (tmp_path / "cytotrace2-runtime.json").is_file()

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(_cytotrace.CytoTraceFailed, match="worker log"):
        _cytotrace.run_cytotrace2(counts, ["A", "B"], names, **args)
