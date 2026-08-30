# tests/test_coexpression_hdwgcna_method.py
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.gene_regulation.coexpression.hdwgcna_method import HdwgcnaMethod
from cellquorum.methods.base import MethodSkip


def _adata(n: int = 200) -> ad.AnnData:
    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, size=(n, 30)).astype(float)
    obs = pd.DataFrame(
        {
            "cell_type": (["A"] * (n // 2)) + (["B"] * (n - n // 2)),
            "condition": (["ctrl"] * (n // 2)) + (["case"] * (n - n // 2)),
        },
        index=[f"c{i}" for i in range(n)],
    )
    a = ad.AnnData(x, obs=obs)
    a.layers["counts"] = x.copy()
    return a


def _ctx(tmp_path: Path, backend) -> SimpleNamespace:  # noqa: ANN001
    paths = SimpleNamespace(results=tmp_path / "res", scratch=tmp_path / "scr")
    reg = SimpleNamespace(get=lambda name: backend if name == "hdwgcna_r" else None)
    return SimpleNamespace(paths=paths, backend_registry=reg)


def test_skips_when_too_few_cells(tmp_path: Path) -> None:
    m = HdwgcnaMethod()
    res = m._run(_adata(10), {"min_cells_total": 100}, _ctx(tmp_path, None))
    assert isinstance(res, MethodSkip)
    assert "min" in res.reason.lower() or "cells" in res.reason.lower()


def test_skips_when_launcher_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    m = HdwgcnaMethod()
    res = m._run(_adata(), {}, _ctx(tmp_path, None))
    assert isinstance(res, MethodSkip)


def test_skips_when_backend_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/micromamba")
    m = HdwgcnaMethod()
    res = m._run(_adata(), {}, _ctx(tmp_path, None))  # backend_registry.get -> None
    assert isinstance(res, MethodSkip)


def test_success_path_with_canned_csvs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/micromamba")

    class FakeBackend:
        def _r_package_available(self, _pkg: str) -> bool:  # noqa: ANN001
            return True

        def run_script(self, _script, _args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(_args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"gene": ["G0", "G1", "G2"], "module": ["M1", "M1", "M2"]}).to_csv(
                out_dir / "modules.csv", index=False
            )
            pd.DataFrame(
                {
                    "gene": ["G0", "G2"],
                    "UMAP1": [0.1, 0.9],
                    "UMAP2": [0.1, 0.9],
                    "module": ["M1", "M2"],
                    "color": ["red", "blue"],
                    "hub": ["hub", "hub"],
                    "kME": [0.9, 0.8],
                }
            ).to_csv(out_dir / "module_umap.csv", index=False)
            return subprocess.CompletedProcess(["x"], 0, stdout="", stderr="")

    m = HdwgcnaMethod()
    res = m._run(_adata(), {}, _ctx(tmp_path, FakeBackend()))
    from cellquorum.core.stage import StageResult

    assert isinstance(res, StageResult)
    names = {a.name for a in res.artifacts}
    assert any("module" in n for n in names)
    assert res.metrics["n_modules"] == 2


def test_skips_on_sentinel_skip_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/micromamba")

    class FakeBackend:
        def _r_package_available(self, _pkg: str) -> bool:  # noqa: ANN001
            return True

        def run_script(self, _script, _args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(_args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "modules.csv").write_text("gene,module\n")
            (out_dir / "hdwgcna_SKIPPED.txt").write_text("missing R packages")
            return subprocess.CompletedProcess(["x"], 0, stdout="", stderr="")

    res = HdwgcnaMethod()._run(_adata(), {}, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)


def test_skips_on_malformed_modules_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/micromamba")

    class FakeBackend:
        def _r_package_available(self, _pkg: str) -> bool:  # noqa: ANN001
            return True

        def run_script(self, _script, _args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(_args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            # Write malformed CSV that pandas cannot read
            (out_dir / "modules.csv").write_bytes(b"\xff\xfe\x00malformed")
            return subprocess.CompletedProcess(["x"], 0, stdout="", stderr="")

    res = HdwgcnaMethod()._run(_adata(), {}, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)
    assert "could not read" in res.reason.lower()


def test_skips_on_modules_csv_missing_module_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/micromamba")

    class FakeBackend:
        def _r_package_available(self, _pkg: str) -> bool:  # noqa: ANN001
            return True

        def run_script(self, _script, _args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(_args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            # Write modules.csv with rows but no 'module' column
            pd.DataFrame({"gene": ["G0", "G1", "G2"], "foo": ["A", "B", "C"]}).to_csv(
                out_dir / "modules.csv", index=False
            )
            return subprocess.CompletedProcess(["x"], 0, stdout="", stderr="")

    res = HdwgcnaMethod()._run(_adata(), {}, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)
    assert "module" in res.reason.lower() and "column" in res.reason.lower()


def test_skips_when_r_script_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A subprocess timeout must become a MethodSkip, never raise out of _run."""
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/micromamba")

    class FakeBackend:
        def _r_package_available(self, _pkg: str) -> bool:  # noqa: ANN001
            return True

        def run_script(self, _script, _args, *, timeout=None):  # noqa: ANN001, ANN003
            raise subprocess.TimeoutExpired(cmd=["micromamba"], timeout=timeout or 1)

    res = HdwgcnaMethod()._run(_adata(), {}, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)
    assert "time" in res.reason.lower() or "fail" in res.reason.lower()


def test_metrics_exclude_grey_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The unassigned 'grey' bin must not inflate n_modules / n_genes_assigned."""
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/micromamba")

    class FakeBackend:
        def _r_package_available(self, _pkg: str) -> bool:  # noqa: ANN001
            return True

        def run_script(self, _script, _args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(_args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            # Two real modules (M1, M2) plus a grey unassigned bin.
            pd.DataFrame(
                {"gene": ["G0", "G1", "G2", "G3"], "module": ["M1", "M1", "M2", "grey"]}
            ).to_csv(out_dir / "modules.csv", index=False)
            return subprocess.CompletedProcess(["x"], 0, stdout="", stderr="")

    res = HdwgcnaMethod()._run(_adata(), {}, _ctx(tmp_path, FakeBackend()))
    from cellquorum.core.stage import StageResult

    assert isinstance(res, StageResult)
    # grey excluded: 2 real modules, 3 genes assigned to real modules.
    assert res.metrics["n_modules"] == 2
    assert res.metrics["n_genes_assigned"] == 3


def test_input_contract_does_not_require_group_by() -> None:
    """A configured-but-absent group_by must not hard-fail the contract.

    The R script falls back to a single 'all' group, so group_by is never a
    hard obs requirement; missing columns route through the R fallback / skip.
    """
    m = HdwgcnaMethod()
    contract = m.input_contract({"group_by": "cell_type"})
    assert "cell_type" not in contract.required_obs


def test_method_registered() -> None:
    import cellquorum.stages.gene_regulation.coexpression  # noqa: F401
    from cellquorum.methods.registry import METHOD_REGISTRY

    assert METHOD_REGISTRY.has("coexpression", "hdwgcna")
