# tests/test_ucell_method.py
"""Tests for UcellMethod (fake pyucell backend -- no real micromamba/pyucell)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.state_scoring.ucell_method import UcellMethod


def _adata(n=20, genes=("G1", "G2", "G3", "G4", "G5")):
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(n, len(genes))).astype("float32")
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cell_type": ["A" if i % 2 == 0 else "B" for i in range(n)]},
            index=[f"cell_{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=list(genes)),
    )
    a.layers["cellquorum_normalized"] = X.copy()
    return a


class _FakeBackend:
    def __init__(self, *, returncode=0, module_ok=True, write_output=True, score_cols=None):
        self.returncode = returncode
        self.module_ok = module_ok
        self.write_output = write_output
        self.score_cols = score_cols or ["proliferating"]
        self.calls = []

    def _py_module_available(self, name):
        return self.module_ok

    def run_script(self, script, args, *, timeout=None):
        self.calls.append((script, args))
        if self.write_output:
            out_csv = Path(args[args.index("--out-csv") + 1])
            h5ad_path = Path(args[args.index("--h5ad") + 1])
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            adata = ad.read_h5ad(h5ad_path)
            rng = np.random.default_rng(1)
            df = pd.DataFrame(
                {c: rng.uniform(0, 1, adata.n_obs) for c in self.score_cols},
                index=adata.obs_names,
            )
            df.insert(0, "cell", adata.obs_names.to_numpy())
            df.to_csv(out_csv, index=False)
        return SimpleNamespace(returncode=self.returncode, stderr="")


def _ctx(tmp_path: Path, backend):
    paths = SimpleNamespace(
        root=tmp_path,
        results=tmp_path / "results",
        figures=tmp_path / "figs",
        scratch=tmp_path / "scratch",
    )
    registry = SimpleNamespace(get=lambda name: backend)
    return SimpleNamespace(paths=paths, backend_registry=registry, config=None)


def test_skips_when_no_programs_configured(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend)
    res = UcellMethod()._run(_adata(), {"use_builtin_programs": False, "programs": {}}, ctx)
    assert isinstance(res, MethodSkip)
    assert "program" in res.reason.lower()


def test_skips_when_no_program_meets_present_gene_gate(tmp_path):
    backend = _FakeBackend()
    ctx = _ctx(tmp_path, backend)
    res = UcellMethod()._run(
        _adata(),
        {
            "use_builtin_programs": False,
            "programs": {"tiny": ["NOT_A_REAL_GENE"]},
            "min_program_genes": 3,
        },
        ctx,
    )
    assert isinstance(res, MethodSkip)


def test_skips_when_backend_unavailable(tmp_path):
    ctx = SimpleNamespace(
        paths=SimpleNamespace(
            root=tmp_path,
            results=tmp_path / "results",
            figures=tmp_path / "figs",
            scratch=tmp_path / "scratch",
        ),
        backend_registry=SimpleNamespace(get=lambda name: (_ for _ in ()).throw(KeyError(name))),
        config=None,
    )
    res = UcellMethod()._run(
        _adata(), {"use_builtin_programs": False, "programs": {"p": ["G1", "G2", "G3"]}}, ctx
    )
    assert isinstance(res, MethodSkip)
    assert "backend" in res.reason.lower()


def test_skips_when_pyucell_module_unavailable(tmp_path):
    backend = _FakeBackend(module_ok=False)
    ctx = _ctx(tmp_path, backend)
    res = UcellMethod()._run(
        _adata(), {"use_builtin_programs": False, "programs": {"p": ["G1", "G2", "G3"]}}, ctx
    )
    assert isinstance(res, MethodSkip)
    assert "pyucell" in res.reason.lower()


def test_skip_reason_names_the_signal_when_subprocess_is_killed(tmp_path):
    backend = _FakeBackend(returncode=-9, write_output=False)
    ctx = _ctx(tmp_path, backend)
    res = UcellMethod()._run(
        _adata(), {"use_builtin_programs": False, "programs": {"p": ["G1", "G2", "G3"]}}, ctx
    )
    assert isinstance(res, MethodSkip)
    assert "sigkill" in res.reason.lower()


def test_success_writes_obsm_and_uns(tmp_path):
    backend = _FakeBackend(score_cols=["proliferating"])
    ctx = _ctx(tmp_path, backend)
    adata = _adata()
    res = UcellMethod()._run(
        adata,
        {
            "use_builtin_programs": False,
            "programs": {"proliferating": ["G1", "G2", "G3"]},
            "min_program_genes": 2,
        },
        ctx,
    )
    assert not isinstance(res, MethodSkip)
    assert "X_state_ucell" in res.adata.obsm
    assert res.adata.obsm["X_state_ucell"].shape == (adata.n_obs, 1)
    assert res.adata.uns["state_ucell"]["programs"] == ["proliferating"]
    assert res.adata.uns["state_ucell"]["genes"]["proliferating"] == ["G1", "G2", "G3"]


def test_success_writes_per_celltype_mean_artifact(tmp_path):
    backend = _FakeBackend(score_cols=["proliferating"])
    ctx = _ctx(tmp_path, backend)
    res = UcellMethod()._run(
        _adata(),
        {
            "use_builtin_programs": False,
            "programs": {"proliferating": ["G1", "G2", "G3"]},
            "cell_type_col": "cell_type",
        },
        ctx,
    )
    names = {a.name for a in res.artifacts}
    assert "state_scoring_results" in names
